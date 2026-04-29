# Detypify Model

This directory contains the Python package and entry scripts for data preprocessing, model training, contribution review, and frontend metadata generation.

## Project Structure

- `proc_data.py`: Compatibility entry script for raw dataset upload and metadata generation.
- `train.py`: Compatibility entry script for model training.
- `review_contrib.py`: Compatibility entry script for maintainer contribution review.
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
uv sync --extra data
```

### Data Preprocessing

Training loads the raw LaTeX-annotated dataset from Hugging Face and uses the
`datasets` local cache for label mapping, rasterization, and splitting. Processed
`ClassLabel` splits are built locally instead of uploaded, which avoids CI/CD
failures when labels change.

Generated frontend metadata is written to `build/generated`:
- `infer.json`: model output symbol metadata.
- `contrib.json`: Typst symbol-name to character mapping for contribution UI.
- `unmapped_latex_symbols.json`: unmapped source labels for review.

To generate frontend inference metadata:

```bash
uv run --extra data python/proc_data.py --gen-metadata
```

To compose the raw dataset (Detexify + MathWriting) and upload it to Hugging Face:

```bash
uv run --extra data python/proc_data.py --upload-raw --datasets detexify --datasets mathwriting
```

The raw upload also writes a local copy to `build/datasets/raw/data.parquet`.

To include the contributed dataset in the raw upload, first review D1 samples:

```bash
uv run --extra data python/review_contrib.py
```

The review command reads the fetched D1 dump from `build/raw/contrib/dataset.json`,
renders images into `build/review/contrib`, and writes accepted samples to
`build/raw/contrib/accepted.json`. The upload command requires that accepted file
when `--datasets contrib` is present:

```bash
uv run --extra data python/proc_data.py --upload-raw --datasets detexify --datasets mathwriting --datasets contrib
```

To print the digest used by CI to detect effective LaTeX-to-Typst mapping changes:

```bash
uv run --extra data python/proc_data.py --print-tex-typ-map-digest
```

See more options with:

```bash
uv run --extra data python/proc_data.py --help
```

### Model Training

>[!NOTE]
> The ema gamma and decay params are crucial things to change if you're meeting with
> accuracy low problem.
> By default, these options are tuned for batch size 128 as default.

To train the default model:

```bash
uv run python/train.py --total-epochs 35 --image-size 224
```

You can specify models to be trained:

```bash
uv run python/train.py --models mobilenetv4_conv_small_035 --models mobilenetv4_conv_small_050
```

The script will:
1. Load raw dataset data from Hugging Face and build cached rendered splits locally.
2. Optionally find the largest batch size when `--find-batch-size` is set.
3. Find a learning rate for non-debug, non-`--dev-run` training.
4. Train each requested model.
5. Export best checkpoints to ONNX under `build/train/{model_name}/version_*/ckpts`.

**Key Options:**
- `--out-dir`: Output directory (default: `build/train`).
- `--debug --dev-run`: Use a small CPU-only fast dev run.
- `--find-batch-size`: Enable Lightning batch-size scaling.
- `--ema-start-epoch`: Epoch to start EMA (default: 5).
- `--log-pred`: Enable logging of predictions (default: True).

To view the training/test logs:

```bash
uv run tensorboard --logdir ./build/train
```

See more tunable options with: `uv run python/train.py --help`
