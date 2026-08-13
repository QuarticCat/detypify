# Detypify Model

This directory contains the Python package and entry scripts for data preprocessing,
model training, checkpoint evaluation, and frontend metadata generation.

## Project Structure

- `proc_data.py`: Data conversion, upload, metadata, mapping-digest, and preview CLI.
- `train.py`: Model training and post-training evaluation CLI.
- `test.py`: Standalone checkpoint evaluation and diagnostic logging CLI.
- `detypify/config.py`: Shared dataset and model configuration.
- `detypify/types.py`: Shared stroke aliases and msgspec structs.
- `detypify/data/`: Raw source parsing, Polars transforms, rendering, metadata, and path config.
- `detypify/training/`: Lightning data module, model definitions, and training callbacks.
- `detypify/tools/`: Local dataset inspection tools.
- `detypify/assets/tex_to_typ_sup.yaml`: Manual mapping overrides for LaTeX to Typst symbol names.

## Development

### Prerequisites

This project requires Python 3.13 or newer and uses `uv` for dependency management.
Run commands from the repository root unless noted otherwise.

>[!WARNING]
> On Linux, plain `uv run` still installs PyTorch indirectly through Lightning
> and timm. The default PyPI build can be CUDA-enabled, and training automatically
> uses CUDA when a compatible GPU is available. Select exactly one accelerator
> extra to pin the intended PyTorch build: `cpu`, `cuda12`, `cuda13`, or `rocm`.
> Pass it to `uv run`, for example `uv run --extra cuda13 python/train.py`.

### Data Preprocessing

Training reads `build/raw/_converted/data.parquet` with Polars, using the output of
`convert-raw` directly when available. If the file is absent, it is downloaded
from Hugging Face through `fsspec`. Polars handles label mapping, filtering,
sampling, and deterministic splitting. Split rows are cached as Arrow IPC files
under `build/train/_dataset_splits`, while the PyTorch data loader rasterizes
strokes on demand.

#### Preparing Raw Data

Raw data conversion writes a reusable cache to
`build/raw/_converted/data.parquet`. This file is also the input used by metadata
generation, preview, training, testing, and `upload-raw`, so the entire pipeline
can run locally without Hugging Face. If this file already exists, skip the
source downloads and conversion below.

Download the original Detexify PostgreSQL dump and symbol metadata from the
[Detexify data archive](https://github.com/kirel/detexify-data):

```bash
curl -fL --create-dirs \
  'https://drive.usercontent.google.com/download?id=0ByuYordD0JBRV01NM2pmNlpfNUE&export=download&authuser=0&confirm=t&resourcekey=0-CZHt-PBM7v0hty25FF5wsg' \
  -o build/raw/detexify/detexify.sql.gz
curl -fL --create-dirs \
  'https://drive.usercontent.google.com/download?id=0ByuYordD0JBRU1Y3Q3VSNk9kdE0&export=download&authuser=0&confirm=t&resourcekey=0-V2m8tmPfD8eyNe4GGrhSxw' \
  -o build/raw/detexify/symbols.json
```

Download the original MathWriting 2024 archive. The conversion command reads
the individual-symbol InkML files directly from this archive, so it does not
need to be extracted:

```bash
curl -fL --create-dirs \
  https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz \
  -o build/raw/mathwriting/mathwriting-2024.tgz
```

Verify both gzip archives:

```bash
gzip --test build/raw/detexify/detexify.sql.gz
gzip --test build/raw/mathwriting/mathwriting-2024.tgz
```

Convert the original sources into the local Parquet cache:

```bash
uv run python/proc_data.py convert-raw --datasets detexify --datasets mathwriting
```

To optionally publish the cached Parquet dataset, authenticate with a token that
has write access to `Cloud0310/detypify-datasets`, then run:

```bash
uv run hf auth login
uv run python/proc_data.py upload-raw
```

`upload-raw` only reads `build/raw/_converted/data.parquet`; it does not access the
original gz/tgz files. It uploads that file to `raw/data.parquet` in the dataset
repository.

To generate frontend inference metadata:

```bash
uv run python/proc_data.py gen-metadata
```

Generated frontend metadata is written to `build/raw/_metadata`:

- `infer.json`: model output symbol metadata.
- `contrib.json`: Typst symbol-name to character mapping for contribution UI.
- `unmapped_latex_symbols.json`: unmapped source labels for review.

To print the digest of the effective LaTeX-to-Typst mapping:

```bash
uv run python/proc_data.py digest
```

To browse mapped dataset samples locally, including truth labels, source, sample
index, pagination, and search:

```bash
uv run python/proc_data.py preview
```

The browser is served at `http://127.0.0.1:8000` by default. Use
`--datasets`, `--port`, `--page-size`, and `--image-size` to change what is shown.

See available subcommands and options with:

```bash
uv run python/proc_data.py --help
uv run python/proc_data.py preview --help
```

### Model Training

>[!NOTE]
> EMA decay and warmup are step-dependent. Their defaults are tuned around the
> default batch size of 128; revisit them when the effective batch size changes
> substantially.

To train the default model with the default settings:

```bash
uv run python/train.py
```

This trains `mobilenet_v4_035` for 40 epochs with 224x224 inputs and an initial
batch size of 128.

You can specify models to be trained:

```bash
uv run python/train.py --models mobilenet_v4_035 --models mobilenet_v4_050
```

Model names use `mobilenet_{v4|v5}_{size}`. The size suffix is divided by 100,
so `mobilenet_v4_035` uses a `0.35` channel multiplier. MobileNetV4 uses a
scaled conv-small model. MobileNetV5 uses a scaled custom trimmed V5
architecture with a compact multi-scale fusion head.

> [!WARNING]
> MobileNetV5 support is experimental and still in development. Use smaller
> size suffixes (e.g. `005`, `010`) to keep the V5 architecture close to
> the V4-small budget. If V5 validation loss becomes NaN, use full fp32 precision
> with `--amp-precision 32-true`. Consider `--no-ema` for shorter V5 training runs:

```bash
uv run python/train.py --models mobilenet_v5_035 --no-ema --no-find-lr --learning-rate 5e-4 --amp-precision 32-true
```

The script will:

1. Read the local raw Parquet data, downloading it only when absent.
2. Build deterministic 80/10/10 Polars splits and cache their vector rows under `build/train/_dataset_splits`.
3. Optionally find the largest batch size when `--find-batch-size` is set.
4. Find a learning rate unless `--no-find-lr` is set.
5. Train each requested model and retain the best and last Lightning checkpoints.
6. Export the best checkpoint to ONNX, then evaluate the best checkpoint on the test split.

Each run writes to `build/train/{model_name}/version_*`. Its `ckpts`
directory contains checkpoints and the exported ONNX model; `training_args.yaml`
records the final batch size and effective learning rate.

**Key Options:**

- `--out-dir`: Output directory (default: `build/train`).
- `--init-batch-size`: Initial batch size (default: `128`).
- `--total-epochs`: Total training epochs (default: `40`).
- `--warmup-epochs`: Linear learning-rate warmup epochs (default: `3`).
- `--learning-rate`: Optimizer learning rate used when LR finder is disabled.
- `--amp-precision`: Training precision (default: `bf16-mixed`; use `32-true` if V5 loss becomes NaN).
- `--find-batch-size`: Enable Lightning batch-size scaling.
- `--no-find-lr`: Skip Lightning learning-rate finder before training.
- `--num-workers`: Override the DataLoader worker count.
- `--ema` / `--no-ema`: Enable or disable EMA weight averaging (enabled by default).
- `--ema-warmup` / `--no-ema-warmup`: Enable or disable inverse-gamma EMA warmup.
- `--compile`: Enable `torch.compile` for the final training run (disabled by default).
- `--log-pred` / `--no-log-pred`: Enable or disable test prediction diagnostics (enabled by default).

CUDA training automatically uses fused AdamW. The default `bf16-mixed` precision
falls back to `16-mixed` when the GPU does not provide native BF16 instructions;
emulated BF16 support is not used for training.

To view the training/test logs:

```bash
uv run tensorboard --logdir ./build/train
```

To run the test diagnostics for an existing checkpoint without retraining:

```bash
uv run python/test.py --ckpt-path build/train/mobilenet_v4_035/version_0/ckpts/last.ckpt
```

By default, evaluation uses both source datasets, takes the image size from the
checkpoint, and writes TensorBoard logs under
`build/train/_eval/existing_model/version_*`. Logs include prediction grids,
`test/confusion_matrix`, `test/top_false_predicted_labels`, and
`test/top_false_predicted_label_examples`.

Use `--out-dir` and `--run-name` to organize output,
`--no-log-predictions` to omit prediction grids, and
`--top-false-labels`, `--examples-per-label`, or `--max-confusion-labels` to
control error diagnostics. Dataset selection must preserve the class ordering
used to train the checkpoint.

See all current options with:

```bash
uv run python/train.py --help
uv run python/test.py --help
```
