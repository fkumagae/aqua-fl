# EdgeBox forecasting models

All models run on CPU. Without explicit paths they retain the original
`forecast_w60_h60` dataset and `w60_h60` output directory. From the repository
root, the original commands still work:

```bash
python -m fluxos.edgebox.models.linear --epochs 25 --batch-size 32 --learning-rate 0.001
python -m fluxos.edgebox.models.mlp --epochs 25 --batch-size 32 --learning-rate 0.001
python -m fluxos.edgebox.models.rnn --epochs 25 --batch-size 32 --learning-rate 0.001
python -m fluxos.edgebox.models.gru --epochs 25 --batch-size 32 --learning-rate 0.001
python -m fluxos.edgebox.models.lstm --epochs 25 --batch-size 32 --learning-rate 0.001
```

The optional `--seed` argument defaults to `42`. Outputs are written to
`dados/edgebox/models/<model>/w60_h60/` as `model.pt`, `model.json`, and
`metrics.json`. The weights file contains only the model `state_dict`.

To select a prepared dataset and keep each experiment separate, pass both paths:

```bash
python -m fluxos.edgebox.models.linear \
  --dataset-dir dados/edgebox/processed/forecast_w60_h30 \
  --output-dir dados/edgebox/models/linear/w60_h30_e10 \
  --epochs 10

python -m fluxos.edgebox.models.gru \
  --dataset-dir dados/edgebox/processed/forecast_w60_h60 \
  --output-dir dados/edgebox/models/gru/w60_h60_e25 \
  --epochs 25
```

The input window, forecast horizon, and feature names come from the selected
dataset's `metadata.json`; the input shape is checked against its `.npz` files.
When metadata is absent, the window comes from the arrays and the horizon from
the `forecast_w<window>_h<horizon>` directory name. The output still contains
six variables. Results for different epoch counts must use distinct output
directories to avoid overwriting earlier runs.

For a CPU-capacity benchmark when validation or test has no usable samples,
add `--train-only`. This still requires valid `train.npz`, `validation.npz`,
`test.npz`, and `metadata.json` files, but only the training split is used for
computation. For example, on the TV Box:

```bash
python -m fluxos.edgebox.models.linear \
  --dataset-dir /root/aqua-fl/dados/edgebox/processed/forecast_w60_h30 \
  --output-dir /root/aqua-fl-final/dados/edgebox/models/linear/w60_h30_e1_compute \
  --epochs 1 --train-only
```

The saved files are marked `train_only: true`. `training_seconds` and
`final_epoch_train_loss` are available, while post-training accuracy metrics
(`train_mse`, validation, and test metrics) are `null`. The per-epoch `train_loss`
is the average loss during parameter updates, not a held-out accuracy score.
These runs measure training capacity, not forecast quality.

`requirements-benchmark.txt` is platform-neutral. On the Linux ARM64 TV Box,
install the CPU wheel explicitly before installing the remaining requirements:

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-benchmark.txt
```

No CUDA device is selected or required.
