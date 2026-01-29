# Detypify Model

This directory contains scripts for data preprocessing, model training, and asset generation for Detypify.

## Project Structure

- `proc_data.py`: Main data preprocessing script. Handles:
  - Scraping symbol information from Typst documentation.
  - Processing raw datasets (Detexify and MathWriting).
  - Mapping LaTeX commands to Typst symbols.
  - Creating and uploading sharded datasets to Hugging Face.
- `train.py`: Training script using PyTorch Lightning.
  - Supports multiple MobileNetV4 variants from the `timm` library.
  - Includes automatic batch size and learning rate finding.
  - Exports the trained model to ONNX format.
- `model.py`: Neural network architectures.
  - `TimmModel`: Wrapper for `timm` models optimized for grayscale math symbol recognition.
  - `CNNModel`: A simpler custom CNN for comparison or smaller tasks.
- `dataset.py`: Data loading and augmentation.
  - `MathSymbolDataModule`: Handles downloading from Hugging Face, rasterizing strokes, and applying real-time augmentations (rotation, affine transforms).
- `proc_font.py`: Font subsetting utility.
  - Subsets the `NewCMMath` font to include only the characters used by Detypify, reducing bundle size for the web interface.
- `review_contrib.py`: Utility to review and incorporate community-contributed symbol samples from the D1 database.
- `tex_to_typ.json`: Manual mapping overrides for LaTeX to Typst symbol names.

## Development

### Prerequisites

Ensure you have the following installed:
- Python 3.13+
- `torch`, `torchvision`, `lightning`, `timm`
- `datasets`, `polars`, `msgspec`, `lxml`, `opencv-python`
- `fontTools` (for font subsetting)

### Data Preprocessing

To prepare the dataset and upload it to Hugging Face:

```bash
python proc_data.py --datasets detexify mathwriting
```

To prepare data locally without uploading:

```bash
python proc_data.py --no-upload --split-parts
```

### Model Training

To train the default models:

```bash
python train.py --total-epochs 35 --image-size 128
```

You can specify different models from `timm`:

```bash
python train.py --timm-models mobilenetv4_conv_small_050 mobilenetv4_conv_small
```

The script will automatically find the optimal batch size and learning rate, train the models, and export them to `build/train/onnx`.

### Font Subsetting

To generate the subsetted font for the web interface:

```bash
python proc_font.py
```

This will create `build/font/NewCMMath-Detypify.woff2`.
