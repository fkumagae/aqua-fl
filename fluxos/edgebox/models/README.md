# EdgeBox forecasting models

All models use the same normalized dataset for the selected horizon and run explicitly on CPU. The input window remains fixed at 60 samples.
From the repository root, train them with:

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

For a five-minute-ahead forecast with the same 60-sample input window, prepare
`forecast_w60_h30` first, then pass `--horizon 30` to each model:

```bash
python -m fluxos.edgebox.benchmark.prepare_dataset --window 60 --horizon 30
python -m fluxos.edgebox.models.linear --horizon 30 --epochs 25 --batch-size 32 --learning-rate 0.001
```

Repeat the second command for `mlp`, `rnn`, `gru`, and `lstm`. Outputs go to
`dados/edgebox/models/<model>/w60_h30/`; the existing `w60_h60` outputs are not
overwritten. Omitting `--horizon` retains the original 60-step default.

Progress is shown once per epoch by default. Use `--verbose` to also show every
batch or `--quiet` when an automated benchmark should minimize terminal I/O.

`requirements-benchmark.txt` is platform-neutral. On the Linux ARM64 TV Box,
install the CPU wheel explicitly before installing the remaining requirements:

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-benchmark.txt
```

No CUDA device is selected or required.
