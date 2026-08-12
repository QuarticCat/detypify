# Detypify Model

This directory contains the Python package and entry scripts for data preprocessing, model training, and frontend metadata generation.

## Project Structure

- `proc_data.py`: Compatibility entry script for raw dataset upload and metadata generation.
- `train.py`: Compatibility entry script for model training.
- `detypify/config.py`: Shared enum and remote dataset config.
- `detypify/types.py`: Shared stroke aliases and msgspec structs.
- `detypify/data/`: Raw source parsing, Typst mapping, rendering, Hugging Face dataset transforms, metadata, and path config.
- `detypify/training/`: Lightning data module, model definitions, and training callbacks.
- `detypify/tools/`: Maintainer tools.
- `detypify/assets/tex_to_typ_sup.yaml`: Manual mapping overrides for LaTeX to Typst symbol names.

## Development

### Prerequisites

This project uses `uv` for dependency management. Run commands from the repository root unless noted otherwise.

For training only, install dependencies with:

```bash
uv sync
```

>[!WARNING]
> Plain `uv run` can resolve and install a PyTorch build that does not match your
> hardware. Select the correct accelerator extra before training, for example
> `uv run --extra cpu ...`, `uv run --extra cuda ...`, or `uv run --extra rocm ...`.

If you're interested in processing data:

```bash
uv sync
```

### Data Preprocessing

Training loads the raw LaTeX-annotated dataset from Hugging Face and uses the
`datasets` local cache for label mapping, rasterization, and splitting. Processed
`ClassLabel` splits are built locally instead of uploaded, which avoids CI/CD
failures when labels change.

#### Preparing Raw Data

Raw data conversion writes a reusable cache to
`build/datasets/raw/data.parquet`. If this file already exists, skip the source
downloads and conversion below and proceed directly to `upload-raw`.

To create the cache from the original Detexify and MathWriting data, install
the Python dependencies and create the source directories first:

```bash
uv sync
mkdir -p build/raw/detexify build/raw/mathwriting
```

Download the original Detexify PostgreSQL dump and symbol metadata from the
[Detexify data archive](https://github.com/kirel/detexify-data):

```bash
curl -fL \
  'https://drive.usercontent.google.com/download?id=0ByuYordD0JBRV01NM2pmNlpfNUE&export=download&authuser=0&confirm=t&resourcekey=0-CZHt-PBM7v0hty25FF5wsg' \
  -o build/raw/detexify/detexify.sql.gz

curl -fL \
  'https://drive.usercontent.google.com/download?id=0ByuYordD0JBRU1Y3Q3VSNk9kdE0&export=download&authuser=0&confirm=t&resourcekey=0-V2m8tmPfD8eyNe4GGrhSxw' \
  -o build/raw/detexify/symbols.json
```

Download the original MathWriting 2024 archive. The conversion command reads
the individual-symbol InkML files directly from this archive, so it does not
need to be extracted:

```bash
curl -fL \
  https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz \
  -o build/raw/mathwriting/mathwriting-2024.tgz
```

The resulting layout must be:

```text
build/raw/
├── detexify/
│   ├── detexify.sql.gz
│   └── symbols.json
└── mathwriting/
    └── mathwriting-2024.tgz
```

Verify both gzip archives:

```bash
gzip --test build/raw/detexify/detexify.sql.gz
gzip --test build/raw/mathwriting/mathwriting-2024.tgz
```

The MathWriting dataset is licensed under CC BY-NC-SA 4.0. The Detexify
database is licensed under ODbL 1.0. Review those terms before redistributing
the combined dataset.

Generated frontend metadata is written to `build/generated`:
- `infer.json`: model output symbol metadata.
- `contrib.json`: Typst symbol-name to character mapping for contribution UI.
- `unmapped_latex_symbols.json`: unmapped source labels for review.

To generate frontend inference metadata:

```bash
uv run python/proc_data.py gen-metadata
```

Convert the original sources into the local Parquet cache:

```bash
uv run python/proc_data.py convert-raw --datasets detexify --datasets mathwriting
```

The conversion scans `detexify.sql.gz` directly; no PostgreSQL server or
intermediate `detexify.json` is needed. It also reads MathWriting symbol files
directly from `mathwriting/mathwriting-2024.tgz` without extracting them.

To upload the cached Parquet dataset, authenticate with a token that has write
access to `Cloud0310/detypify-datasets`, then run:

```bash
uv run hf auth login
uv run python/proc_data.py upload-raw
```

`upload-raw` only reads `build/datasets/raw/data.parquet`; it does not access the
original gz/tgz files. It uploads the `raw` configuration with the `data` split.

To print the digest of the effective LaTeX-to-Typst mapping:

```bash
uv run python/proc_data.py digest
```

To browse mapped dataset samples locally, including truth labels, source, sample
index, pagination, and search:

```bash
uv run --extra cpu python/proc_data.py preview
```

The browser is served at `http://127.0.0.1:8000` by default. Use
`--datasets`, `--port`, `--page-size`, and `--image-size` to change what is shown.

See more options with:

```bash
uv run python/proc_data.py --help
```

### Model Training

>[!NOTE]
> The ema gamma and decay params are crucial things to change if you're meeting with
> accuracy low problem.
> By default, these options are tuned for batch size 128 as default.

To train the default MobileNet comparison set:

```bash
uv run --extra <accelerator> python/train.py --total-epochs 35 --image-size 224
```

This trains `mobilenet_v4_035`.

You can specify models to be trained:

```bash
uv run --extra <accelerator> python/train.py --models mobilenet_v4_035 --models mobilenet_v4_050
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
uv run --extra <accelerator> python/train.py --models mobilenet_v5_035 --no-ema --no-find-lr --learning-rate 5e-4 --amp-precision 32-true
```

The script will:
1. Load raw dataset data from Hugging Face and build cached rendered splits locally.
2. Optionally find the largest batch size when `--find-batch-size` is set.
3. Find a learning rate for non-debug, non-`--dev-run` training unless `--no-find-lr` is set.
4. Train each requested model.
5. Export best checkpoints to ONNX under `build/train/{model_name}/version_*/ckpts`.

**Key Options:**
- `--out-dir`: Output directory (default: `build/train`).
- `--debug --dev-run`: Use a small CPU-only fast dev run.
- `--learning-rate`: Optimizer learning rate used when LR finder is disabled.
- `--amp-precision`: Training precision (default: `bf16-mixed`; use `32-true` if V5 loss becomes NaN).
- `--find-batch-size`: Enable Lightning batch-size scaling.
- `--no-find-lr`: Skip Lightning learning-rate finder before training.
- `--num-workers`: Override DataLoader and dataset mapping worker count.
- `--log-pred`: Enable logging of predictions (default: True).

To view the training/test logs:

```bash
uv run tensorboard --logdir ./build/train
```

To run the test diagnostics for an existing checkpoint without retraining:

```bash
uv run --extra <accelerator> python/test.py --ckpt-path build/train/mobilenet_v4_035/version_0/ckpts/best-epoch=00-val_acc=0.0000.ckpt
```

This writes TensorBoard logs under `build/train/eval/existing_model/version_*`,
including `test/confusion_matrix`, `test/top_false_predicted_labels`, and
`test/top_false_predicted_label_examples`.

See more tunable options with: `uv run python/train.py --help`
