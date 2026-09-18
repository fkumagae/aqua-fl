"""CPU-only tests for the supervised EdgeBox forecasting models."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from fluxos.edgebox.models.common import (
    FEATURES,
    count_parameters,
    default_dataset_dir,
    default_output_dir,
    load_dataset,
    parse_training_args,
    run_experiment,
    run_model_cli,
)
from fluxos.edgebox.models.gru import build_model as build_gru
from fluxos.edgebox.models.linear import build_model as build_linear
from fluxos.edgebox.models.lstm import build_model as build_lstm
from fluxos.edgebox.models.mlp import build_model as build_mlp
from fluxos.edgebox.models.rnn import build_model as build_rnn


MODEL_CASES = {
    "linear": (build_linear, 2_166, None, None),
    "mlp": (build_mlp, 25_382, None, None),
    "rnn": (build_rnn, 1_478, 32, 1),
    "gru": (build_gru, 4_038, 32, 1),
    "lstm": (build_lstm, 5_318, 32, 1),
}


class ForecastModelTests(unittest.TestCase):
    def make_dataset(
        self,
        parent: Path,
        window: int,
        horizon: int,
        input_size: int = 6,
        metadata_present: bool = True,
    ) -> Path:
        dataset_dir = parent / f"forecast_w{window}_h{horizon}"
        dataset_dir.mkdir()
        generator = np.random.default_rng(42)
        counts = {"train": 12, "validation": 8, "test": 8}
        for split, samples in counts.items():
            inputs = generator.normal(size=(samples, window, input_size)).astype(np.float32)
            targets = generator.normal(size=(samples, 6)).astype(np.float32)
            np.savez_compressed(dataset_dir / f"{split}.npz", X=inputs, y=targets)

        if metadata_present:
            metadata = {
                "features": FEATURES if input_size == 6 else [f"feature_{i}" for i in range(input_size)],
                "sampling_seconds": 10,
                "input_window": window,
                "forecast_horizon": horizon,
                "forecast_type": "point",
                "dtype": "float32",
                "train_samples": counts["train"],
                "validation_samples": counts["validation"],
                "test_samples": counts["test"],
            }
            (dataset_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return dataset_dir

    def test_cli_accepts_explicit_paths_and_preserves_defaults(self) -> None:
        with patch("sys.argv", ["linear"]):
            defaults = parse_training_args("linear")
        self.assertIsNone(defaults.dataset_dir)
        self.assertIsNone(defaults.output_dir)
        self.assertEqual((defaults.epochs, defaults.batch_size, defaults.learning_rate, defaults.seed), (25, 32, 0.001, 42))
        self.assertEqual(default_dataset_dir().name, "forecast_w60_h60")
        self.assertEqual(default_output_dir("linear").name, "w60_h60")

        arguments = ["linear", "--dataset-dir", "forecast_w60_h30", "--output-dir", "linear/w60_h30_e10", "--epochs", "10"]
        with patch("sys.argv", arguments):
            parsed = parse_training_args("linear")
        self.assertEqual(parsed.dataset_dir, Path("forecast_w60_h30"))
        self.assertEqual(parsed.output_dir, Path("linear/w60_h30_e10"))
        self.assertEqual(parsed.epochs, 10)

        with tempfile.TemporaryDirectory(prefix="aquafl_cli_test_") as temp_name:
            temp_dir = Path(temp_name)
            dataset_dir = self.make_dataset(temp_dir, 60, 30)
            output_dir = temp_dir / "linear" / "w60_h30_e1"
            arguments = [
                "linear", "--dataset-dir", str(dataset_dir),
                "--output-dir", str(output_dir), "--epochs", "1",
            ]
            with patch("sys.argv", arguments):
                run_model_cli("linear", build_linear, hidden_size=None, num_layers=None)
            saved = json.loads((output_dir / "model.json").read_text(encoding="utf-8"))
            self.assertEqual((saved["input_window"], saved["forecast_horizon"]), (60, 30))

    def test_dynamic_input_shapes_for_all_models(self) -> None:
        for window, input_size in ((30, 6), (120, 8)):
            inputs = torch.randn(2, window, input_size)
            for model_name, (factory, _, _, _) in MODEL_CASES.items():
                with self.subTest(model=model_name, window=window, input_size=input_size):
                    model = factory(input_window=window, input_size=input_size, output_size=6)
                    predictions = model(inputs)
                    self.assertEqual(tuple(predictions.shape), (2, 6))
                    predictions.sum().backward()
                    self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
                    if model_name == "linear":
                        self.assertEqual(model.output.in_features, window * input_size)
                    if model_name == "mlp":
                        self.assertEqual(model.network[1].in_features, window * input_size)
                    if model_name in ("rnn", "gru", "lstm"):
                        self.assertEqual(model.recurrent.input_size, input_size)

    def test_forward_backward_state_dict_parameters_and_cpu(self) -> None:
        inputs = torch.randn(8, 60, 6, device="cpu")
        targets = torch.randn(8, 6, device="cpu")

        with tempfile.TemporaryDirectory(prefix="aquafl_model_test_") as temp_name:
            temp_dir = Path(temp_name)
            for model_name, (factory, expected_parameters, _, _) in MODEL_CASES.items():
                with self.subTest(model=model_name):
                    model = factory()
                    self.assertTrue(all(parameter.device.type == "cpu" for parameter in model.parameters()))
                    predictions = model(inputs)
                    self.assertEqual(tuple(predictions.shape), (8, 6))
                    torch.nn.functional.mse_loss(predictions, targets).backward()
                    self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
                    self.assertEqual(count_parameters(model), expected_parameters)

                    weights_path = temp_dir / f"{model_name}.pt"
                    torch.save(model.state_dict(), weights_path)
                    restored = factory()
                    restored.load_state_dict(
                        torch.load(weights_path, map_location="cpu", weights_only=True)
                    )
                    for key, value in model.state_dict().items():
                        self.assertTrue(torch.equal(value, restored.state_dict()[key]))

    def test_one_epoch_smoke_training_for_every_model(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aquafl_training_test_") as temp_name:
            temp_dir = Path(temp_name)
            dataset_dir = self.make_dataset(temp_dir, 60, 60)

            for model_name, (factory, expected_parameters, hidden_size, num_layers) in MODEL_CASES.items():
                with self.subTest(model=model_name):
                    output_dir = temp_dir / "outputs" / model_name
                    model_metadata, metrics = run_experiment(
                        model_name,
                        factory,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        epochs=1,
                        batch_size=4,
                        learning_rate=0.001,
                        seed=42,
                        dataset_dir=dataset_dir,
                        output_dir=output_dir,
                    )
                    self.assertEqual(model_metadata["device"], "cpu")
                    self.assertEqual(model_metadata["dataset"], "forecast_w60_h60")
                    self.assertEqual(model_metadata["input_window"], 60)
                    self.assertEqual(model_metadata["forecast_horizon"], 60)
                    self.assertEqual(model_metadata["parameter_count"], expected_parameters)
                    self.assertEqual(set(metrics["test_mae_per_feature"]), set(FEATURES))
                    self.assertTrue((output_dir / "model.pt").is_file())
                    self.assertTrue((output_dir / "model.json").is_file())
                    self.assertTrue((output_dir / "metrics.json").is_file())

    def test_datasets_and_custom_outputs_for_multiple_scenarios(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aquafl_scenarios_test_") as temp_name:
            temp_dir = Path(temp_name)
            for window, horizon in ((60, 60), (60, 30), (30, 60), (120, 60)):
                with self.subTest(window=window, horizon=horizon):
                    dataset_dir = self.make_dataset(temp_dir, window, horizon)
                    bundle = load_dataset(dataset_dir)
                    self.assertEqual(bundle.input_window, window)
                    self.assertEqual(bundle.metadata["forecast_horizon"], horizon)
                    self.assertEqual(bundle.metadata["dataset"], dataset_dir.name)
                    self.assertEqual(default_output_dir("linear", dataset_dir).name, f"w{window}_h{horizon}")

                    output_dir = temp_dir / "outputs" / f"w{window}_h{horizon}_e10"
                    model_metadata, metrics = run_experiment(
                        "linear",
                        build_linear,
                        hidden_size=None,
                        num_layers=None,
                        epochs=1,
                        batch_size=4,
                        learning_rate=0.001,
                        seed=42,
                        dataset_dir=dataset_dir,
                        output_dir=output_dir,
                    )
                    expected_parameters = (window * 6 + 1) * 6
                    for result in (model_metadata, metrics):
                        self.assertEqual(result["dataset"], dataset_dir.name)
                        self.assertEqual(result["input_window"], window)
                        self.assertEqual(result["forecast_horizon"], horizon)
                        self.assertEqual(result["input_size"], 6)
                        self.assertEqual(result["output_size"], 6)
                        self.assertEqual(result["parameter_count"], expected_parameters)
                        self.assertEqual(result["epochs"], 1)
                        self.assertEqual(result["batch_size"], 4)
                        self.assertEqual(result["learning_rate"], 0.001)
                    self.assertEqual(json.loads((output_dir / "model.json").read_text())["input_window"], window)
                    self.assertEqual(json.loads((output_dir / "metrics.json").read_text())["forecast_horizon"], horizon)
                    self.assertTrue((output_dir / "model.pt").is_file())

    def test_metadata_fallback_and_dimension_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aquafl_metadata_test_") as temp_name:
            temp_dir = Path(temp_name)
            fallback_dir = self.make_dataset(temp_dir, 30, 60, metadata_present=False)
            bundle = load_dataset(fallback_dir)
            self.assertEqual(bundle.metadata["forecast_horizon"], 60)
            self.assertEqual(bundle.input_window, 30)

            named_dir = self.make_dataset(temp_dir, 60, 30)
            named_metadata_path = named_dir / "metadata.json"
            named_metadata = json.loads(named_metadata_path.read_text(encoding="utf-8"))
            named_metadata["scenario"] = "benchmark_five_minutes"
            named_metadata_path.write_text(json.dumps(named_metadata), encoding="utf-8")
            self.assertEqual(load_dataset(named_dir).metadata["dataset"], "benchmark_five_minutes")

            mismatched_dir = self.make_dataset(temp_dir, 120, 60)
            metadata_path = mismatched_dir / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["input_window"] = 30
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input_window"):
                load_dataset(mismatched_dir)


if __name__ == "__main__":
    unittest.main()
