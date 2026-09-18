"""Shared training utilities for the EdgeBox forecasting models."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DATASET_NAME = "forecast_w60_h60"  # Default for calls without --dataset-dir.
INPUT_WINDOW = 60  # Defaults for direct build_model() calls.
INPUT_SIZE = 6
OUTPUT_SIZE = 6
FEATURES = [
    "cpu_percent",
    "ram_percent",
    "temperature_c",
    "disk_percent",
    "latency_ms",
    "load1",
]


@dataclass(frozen=True)
class DatasetBundle:
    """Arrays and metadata loaded once for a complete experiment."""

    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    metadata: dict[str, Any]

    @property
    def input_window(self) -> int:
        return self.train_x.shape[1]

    @property
    def input_size(self) -> int:
        return self.train_x.shape[2]

    @property
    def output_size(self) -> int:
        return self.train_y.shape[1]


def project_root() -> Path:
    """Resolve the repository root without depending on the current directory."""

    source = Path(__file__).resolve()
    for candidate in source.parents:
        if (candidate / "config.py").is_file() and (candidate / "fluxos").is_dir():
            return candidate
    raise RuntimeError(f"Could not resolve the AquaFL project root from {source}")


def default_dataset_dir() -> Path:
    return project_root() / "dados" / "edgebox" / "processed" / DATASET_NAME


def default_output_dir(model_type: str, dataset_dir: Path | None = None) -> Path:
    scenario = (dataset_dir or default_dataset_dir()).name.removeprefix("forecast_")
    return project_root() / "dados" / "edgebox" / "models" / model_type / scenario


def set_seed(seed: int) -> None:
    """Seed all RNGs used by this CPU-only training pipeline."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cpu_device() -> torch.device:
    """Return the only device supported by this benchmark."""

    return torch.device("cpu")


def validate_model_input(
    inputs: torch.Tensor,
    input_window: int = INPUT_WINDOW,
    input_size: int = INPUT_SIZE,
) -> None:
    expected = (input_window, input_size)
    if inputs.ndim != 3 or tuple(inputs.shape[1:]) != expected:
        raise ValueError(
            f"Expected input shape (batch, {input_window}, {input_size}), "
            f"received {tuple(inputs.shape)}"
        )


def _read_metadata(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "metadata.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as source:
        metadata = json.load(source)
    if not isinstance(metadata, dict):
        raise ValueError(f"Dataset metadata must be a JSON object: {path}")
    return metadata


def _load_split(dataset_dir: Path, split: str, *, allow_empty: bool = False) -> tuple[np.ndarray, np.ndarray]:
    path = dataset_dir / f"{split}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        missing = {"X", "y"}.difference(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        inputs = np.asarray(archive["X"], dtype=np.float32)
        targets = np.asarray(archive["y"], dtype=np.float32)
    _validate_split(split, inputs, targets, allow_empty=allow_empty)
    return inputs, targets


def _validate_split(split: str, inputs: np.ndarray, targets: np.ndarray, *, allow_empty: bool = False) -> None:
    if inputs.ndim != 3 or inputs.shape[1] <= 0 or inputs.shape[2] <= 0:
        raise ValueError(f"{split}.npz X must have shape (N, window, features); received {inputs.shape}")
    if targets.ndim != 2 or targets.shape[1] != OUTPUT_SIZE:
        raise ValueError(
            f"{split}.npz y must have shape (N, {OUTPUT_SIZE}); received {targets.shape}"
        )
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError(
            f"{split}.npz has different sample counts: X={inputs.shape[0]}, y={targets.shape[0]}"
        )
    if inputs.shape[0] == 0 and not allow_empty:
        raise ValueError(f"{split}.npz contains no samples")


def _validate_metadata(metadata: dict[str, Any], bundle: DatasetBundle, dataset_dir: Path) -> None:
    for split, inputs, targets in (
        ("validation", bundle.validation_x, bundle.validation_y),
        ("test", bundle.test_x, bundle.test_y),
    ):
        if inputs.shape[1:] != bundle.train_x.shape[1:] or targets.shape[1:] != bundle.train_y.shape[1:]:
            raise ValueError(f"{split}.npz dimensions differ from train.npz")

    window = metadata.get("input_window", bundle.input_window)
    if type(window) is not int or window != bundle.input_window:
        raise ValueError(f"metadata.json input_window must match X shape ({bundle.input_window})")

    horizon = metadata.get("forecast_horizon")
    if horizon is None:
        match = re.search(r"(?:^|_)w(\d+)_h(\d+)(?:_|$)", dataset_dir.name)
        if match and int(match.group(1)) != bundle.input_window:
            raise ValueError("dataset directory window does not match X shape")
        horizon = int(match.group(2)) if match else None
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("forecast_horizon must be a positive integer in metadata.json or dataset directory name")

    features = metadata.get("features", FEATURES if bundle.input_size == len(FEATURES) else None)
    if not isinstance(features, list) or len(features) != bundle.input_size or not all(isinstance(name, str) and name for name in features):
        raise ValueError(f"metadata.json features must contain {bundle.input_size} names")
    output_features = metadata.get("target_features", FEATURES)
    if not isinstance(output_features, list) or len(output_features) != bundle.output_size:
        raise ValueError(f"metadata.json target_features must contain {bundle.output_size} names")
    for key, expected_value in (("forecast_type", "point"), ("dtype", "float32")):
        if key in metadata and metadata[key] != expected_value:
            raise ValueError(f"metadata.json field {key!r} must be {expected_value!r}")

    dataset_name = metadata.get("dataset") or metadata.get("dataset_name") or metadata.get("scenario") or dataset_dir.name
    if not isinstance(dataset_name, str):
        raise ValueError("metadata.json dataset name must be a string")
    metadata["dataset"] = dataset_name
    metadata["input_window"] = window
    metadata["forecast_horizon"] = horizon
    metadata["features"] = features
    metadata["target_features"] = output_features

    counts = {
        "train_samples": bundle.train_x.shape[0],
        "validation_samples": bundle.validation_x.shape[0],
        "test_samples": bundle.test_x.shape[0],
    }
    for key, actual in counts.items():
        if key in metadata and metadata[key] != actual:
            raise ValueError(
                f"metadata.json field {key!r} is {metadata[key]!r}, but the split contains {actual}"
            )


def load_dataset(dataset_dir: Path | None = None, *, train_only: bool = False) -> DatasetBundle:
    """Load and validate all splits exactly once."""

    resolved = (dataset_dir or default_dataset_dir()).resolve()
    metadata = _read_metadata(resolved)
    train_x, train_y = _load_split(resolved, "train")
    validation_x, validation_y = _load_split(resolved, "validation", allow_empty=train_only)
    test_x, test_y = _load_split(resolved, "test", allow_empty=train_only)
    bundle = DatasetBundle(
        train_x=train_x,
        train_y=train_y,
        validation_x=validation_x,
        validation_y=validation_y,
        test_x=test_x,
        test_y=test_y,
        metadata=metadata,
    )
    _validate_metadata(metadata, bundle, resolved)
    return bundle


def numpy_to_torch(inputs: np.ndarray, targets: np.ndarray) -> TensorDataset:
    """Create a zero-copy CPU TensorDataset when arrays are already float32."""

    return TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))


def create_data_loaders(
    bundle: DatasetBundle,
    batch_size: int,
    seed: int,
    *,
    train_only: bool = False,
) -> dict[str, DataLoader]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    common = {"batch_size": batch_size, "num_workers": 0, "pin_memory": False}
    loaders = {
        "train": DataLoader(
            numpy_to_torch(bundle.train_x, bundle.train_y),
            shuffle=True,
            generator=generator,
            **common,
        ),
    }
    if train_only:
        return loaders
    loaders.update({
        "validation": DataLoader(
            numpy_to_torch(bundle.validation_x, bundle.validation_y),
            shuffle=False,
            **common,
        ),
        "test": DataLoader(
            numpy_to_torch(bundle.test_x, bundle.test_y),
            shuffle=False,
            **common,
        ),
    })
    return loaders


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def calculate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    per_feature: bool = False,
    output_features: list[str] | None = None,
) -> dict[str, Any]:
    """Calculate sample-weighted regression metrics without retaining predictions."""

    model.eval()
    squared_error = 0.0
    absolute_error = 0.0
    element_count = 0
    feature_absolute_error = torch.zeros(len(output_features or FEATURES), dtype=torch.float64)
    sample_count = 0

    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            predictions = model(inputs)
            errors = predictions - targets
            squared_error += errors.square().sum().item()
            absolute_error += errors.abs().sum().item()
            element_count += errors.numel()
            sample_count += errors.shape[0]
            if per_feature:
                feature_absolute_error += errors.abs().sum(dim=0, dtype=torch.float64).cpu()

    if element_count == 0:
        raise ValueError("Cannot calculate metrics for an empty DataLoader")
    mse = squared_error / element_count
    result: dict[str, Any] = {
        "mse": mse,
        "mae": absolute_error / element_count,
        "rmse": math.sqrt(mse),
    }
    if per_feature:
        result["mae_per_feature"] = {
            feature: float(value / sample_count)
            for feature, value in zip(output_features or FEATURES, feature_absolute_error.tolist())
        }
    return result


def train_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    epochs: int,
    learning_rate: float,
    device: torch.device,
    *,
    train_only: bool = False,
) -> tuple[list[dict[str, float | int]], float]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for inputs, targets in loaders["train"]:
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * inputs.shape[0]
            sample_count += inputs.shape[0]

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": loss_sum / sample_count,
        }
        if not train_only:
            validation = calculate_metrics(model, loaders["validation"], device)
            epoch_metrics["validation_mse"] = validation["mse"]
        history.append(epoch_metrics)
        line = f"epoch={epoch}/{epochs} train_mse={epoch_metrics['train_loss']:.6f}"
        if not train_only:
            line += f" validation_mse={epoch_metrics['validation_mse']:.6f}"
        print(line, flush=True)

    return history, time.perf_counter() - started


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_experiment(
    model_type: str,
    model_factory: Callable[..., nn.Module],
    *,
    hidden_size: int | None,
    num_layers: int | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    dataset_dir: Path | None = None,
    output_dir: Path | None = None,
    train_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train and serialize one model, optionally skipping all evaluation."""

    set_seed(seed)
    device = cpu_device()
    bundle = load_dataset(dataset_dir, train_only=train_only)
    loaders = create_data_loaders(bundle, batch_size=batch_size, seed=seed, train_only=train_only)
    model = model_factory(
        input_window=bundle.input_window,
        input_size=bundle.input_size,
        output_size=bundle.output_size,
    ).to(device)
    parameter_count = count_parameters(model)
    history, training_seconds = train_model(
        model,
        loaders,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        train_only=train_only,
    )

    train_metrics = None if train_only else calculate_metrics(model, loaders["train"], device)
    validation_metrics = None if train_only else calculate_metrics(model, loaders["validation"], device)
    test_metrics = None if train_only else calculate_metrics(
        model, loaders["test"], device, per_feature=True,
        output_features=bundle.metadata["target_features"],
    )
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    model_metadata = {
        "model_type": model_type,
        "dataset": bundle.metadata["dataset"],
        "input_window": bundle.input_window,
        "forecast_horizon": bundle.metadata["forecast_horizon"],
        "input_size": bundle.input_size,
        "output_size": bundle.output_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "features": bundle.metadata["features"],
        "target_features": bundle.metadata["target_features"],
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "optimizer": "Adam",
        "loss": "MSELoss",
        "parameter_count": parameter_count,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": str(device),
        "train_only": train_only,
        "trained_at": trained_at,
    }
    metrics = {
        "model_type": model_type,
        "dataset": bundle.metadata["dataset"],
        "input_window": bundle.input_window,
        "forecast_horizon": bundle.metadata["forecast_horizon"],
        "input_size": bundle.input_size,
        "output_size": bundle.output_size,
        "features": bundle.metadata["features"],
        "target_features": bundle.metadata["target_features"],
        "train_samples": bundle.train_x.shape[0],
        "validation_samples": bundle.validation_x.shape[0],
        "test_samples": bundle.test_x.shape[0],
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "loss": "MSELoss",
        "train_only": train_only,
        "train_mse": None if train_only else train_metrics["mse"],
        "train_mae": None if train_only else train_metrics["mae"],
        "train_rmse": None if train_only else train_metrics["rmse"],
        "validation_mse": None if train_only else validation_metrics["mse"],
        "validation_mae": None if train_only else validation_metrics["mae"],
        "validation_rmse": None if train_only else validation_metrics["rmse"],
        "test_mse": None if train_only else test_metrics["mse"],
        "test_mae": None if train_only else test_metrics["mae"],
        "test_rmse": None if train_only else test_metrics["rmse"],
        "test_mae_per_feature": None if train_only else test_metrics["mae_per_feature"],
        "final_epoch_train_loss": history[-1]["train_loss"],
        "training_seconds": training_seconds,
        "parameter_count": parameter_count,
        "trained_at": trained_at,
        "history": history,
    }

    destination = (output_dir or default_output_dir(model_type, dataset_dir)).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination / "model.pt")
    write_json(destination / "model.json", model_metadata)
    write_json(destination / "metrics.json", metrics)
    print(f"saved={destination}", flush=True)
    return model_metadata, metrics


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def parse_training_args(model_type: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Train the EdgeBox {model_type} forecaster on CPU.")
    parser.add_argument("--epochs", type=_positive_int, default=25)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--learning-rate", type=_positive_float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-only", action="store_true", help="Benchmark training without validation or test metrics.")
    return parser.parse_args()


def run_model_cli(
    model_type: str,
    model_factory: Callable[..., nn.Module],
    *,
    hidden_size: int | None,
    num_layers: int | None,
) -> None:
    args = parse_training_args(model_type)
    run_experiment(
        model_type,
        model_factory,
        hidden_size=hidden_size,
        num_layers=num_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        train_only=args.train_only,
    )
