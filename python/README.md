# Detypify Model

This directory contains scripts for data preprocessing, model training, and asset generation for Detypify.

## Project Structure

- `proc_data.py`: Main data preprocessing script. Handles:
  - Scraping symbol information from Typst documentation (downloads `typ_sym.html` if missing).
  - Processing raw datasets (Detexify and MathWriting).
  - Mapping LaTeX commands to Typst symbols.
  - Creating and uploading sharded datasets to Hugging Face.
  - Generating inference metadata (`infer.json`) and contribution metadata (`contrib.json`).
- `train.py`: Training script using PyTorch Lightning.
  - Supports multiple MobileNetV4 variants from the `timm` library and a custom CNN.
  - Includes automatic batch size and learning rate finding.
  - Exports the trained model to ONNX format.
  - Uses TensorBoard for logging.
- `model.py`: Neural network architectures.
  - `TimmModel`: Wrapper for `timm` models optimized for grayscale math symbol recognition.
  - `CNNModel`: A simple custom CNN for comparison or smaller tasks.
- `dataset.py`: Data loading and augmentation.
  - `MathSymbolDataModule`: Handles downloading from Hugging Face, rasterizing strokes, and applying real-time augmentations (rotation, affine transforms).
- `proc_font.py`: Font subsetting utility.
  - Subsets the `NewCMMath` font to include only the characters used by Detypify, reducing bundle size for the web interface.
  - Requires `external/NewCMMath-Regular.otf` to be present.
- `review_contrib.py`: Utility to review and incorporate community-contributed symbol samples from the D1 database (Maintainer only).
- `tex_to_typ.json`: Manual mapping overrides for LaTeX to Typst symbol names.
- `callbacks.py`: Custom callbacks for model training:
  - `EMAWeightAveraging`: EMA implementation with performance optimization and warmup, similar to `timm`'s EMAv3.
  - `LogPredictCallback`: Logs sample images with their ground truth and predicted labels to TensorBoard for visual performance review.
- `test_model.py`: Script for testing pre-trained model performance and logging wrong guesses.

## Development

### Prerequisites

This project uses `uv` for dependency management.

For training only, install dependencies with:

```bash
uv sync
```

If you're interested in processing data (includes data processing, font processing):

```bash
uv sync --extra=data
```

### Data Preprocessing

To compose the dataset (Detexify + MathWriting) and upload it to Hugging Face:

```bash
uv run --extra=data proc_data.py --datasets detexify mathwriting
```

To prepare data locally without uploading (useful for debugging):

```bash
uv run --extra=data proc_data.py --no-upload --split-parts
```

To generate data for worker/frontend review without full dataset processing:

```bash
uv run --extra=data proc_data.py --skip-convert-data
```

To include the contributed dataset (requires `build/dataset.json`):

```bash
uv run --extra=data proc_data.py --include-contrib
```

See more options with:

```bash
uv run --extra=data proc_data.py --help
```

### Model Training

To train the default models (defined in `train.py`):

```bash
uv run train.py --total-epochs 35 --image-size 224
```

You can specify models to be trained:

```bash
uv run train.py --models mobilenetv4_conv_small_050 --models mobilenetv4_conv_small
```

The script will:
1. Automatically find the optimal batch size (can be disabled with `--no-find-batch-size`).
2. Find the optimal learning rate.
3. Train the models.
4. Export them to `build/train/onnx`.

**Key Options:**
- `--out-dir`: Output directory (default: `build/train`).
- `--ema-start-epoch`: Epoch to start EMA (default: 10).
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

### Font Subsetting

To generate the subsetted font for the web interface:

```bash
# Ensure external/NewCMMath-Regular.otf exists
uv run --extra=data proc_font.py
```

This will create `build/font/NewCMMath-Detypify.woff2`.
