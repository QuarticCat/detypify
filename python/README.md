# Detypify Model

This directory contains scripts for data preprocessing, model training, and asset generation for Detypify.

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

This project uses `uv` for dependency management.

For training only, install dependencies with:

```bash
uv sync
```

If you're interested in processing data:

```bash
uv sync --extra=data
```

### Data Preprocessing

Training loads the raw LaTeX-annotated dataset from Hugging Face and uses the
`datasets` local cache for label mapping, rasterization, and splitting. Processed
`ClassLabel` splits are not uploaded, which avoids CI/CD failures when labels
change.

To generate frontend inference metadata:

```bash
uv run --extra=data proc_data.py --gen-metadata
```

To compose the raw dataset (Detexify + MathWriting) and upload it to Hugging Face:

```bash
uv run --extra=data proc_data.py --upload-raw --datasets detexify --datasets mathwriting
```

To include the contributed dataset in the raw upload (requires `build/raw/contrib/dataset.json`):

```bash
uv run --extra=data proc_data.py --upload-raw --datasets detexify --datasets mathwriting --datasets contrib
```

See more options with:

```bash
uv run --extra=data proc_data.py --help
```

### Model Training

>[!NOTE]
> The ema gamma and decay params are crucial things to change if you're meeting with
> accuracy low problem.
> By default, these options are tuned for batch size 128 as default.

To train the default models (defined in `train.py`):

```bash
uv run train.py --total-epochs 35 --image-size 224
```

You can specify models to be trained:
```bash
uv run train.py --models mobilenetv4_conv_small_035 --models mobilenetv4_conv_small_050
```

The script will:
1. Automatically find the optimal batch size (can be disabled with `--no-find-batch-size`).
2. Find the optimal learning rate.
3. Train the models.
4. Export checkpoints them to `build/train/{model_name}/ckpts`.

**Key Options:**
- `--out-dir`: Output directory (default: `build/train`).
- `--ema-start-epoch`: Epoch to start EMA (default: 5).
- `--log-pred`: Enable logging of predictions (default: True).

To view the training/test logs:

```bash
uv run tensorboard --logdir ./build/{train,test}
```

See more tunable options with: `uv run python/train.py --help`

### Model Testing

To test a trained model checkpoint:

```bash
uv run python/test_model.py path/to/checkpoint.ckpt --model-type timm --model-name mobilenetv4_conv_small_050
```

For CNN models:

```bash
uv run python/test_model.py path/to/checkpoint.ckpt --model-type cnn
```
