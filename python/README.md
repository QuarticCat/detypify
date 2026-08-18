# Detypify model pipeline

This directory contains the data conversion, symbol mapping, metadata, training,
and evaluation tools for Detypify.

## Requirements

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/)
- `curl` for downloading the source datasets

Sync the Python dependencies once, choosing the PyTorch extra for the machine
you are using:

```bash
uv sync --extra=cuda13  # or cpu, cuda12, rocm
```

The commands below use `uv run --no-sync` to preserve the selected PyTorch
environment. Run `uv sync --extra=<extra>` again after changing `pyproject.toml`
or `uv.lock`.

## Development

Run these commands from the repository root.

### 1. Download the source datasets

```bash
curl -fL --create-dirs \
  'https://drive.usercontent.google.com/download?id=0ByuYordD0JBRV01NM2pmNlpfNUE&export=download&authuser=0&confirm=t&resourcekey=0-CZHt-PBM7v0hty25FF5wsg' \
  -o build/data/detexify/detexify.sql.gz
curl -fL --create-dirs \
  'https://drive.usercontent.google.com/download?id=0ByuYordD0JBRU1Y3Q3VSNk9kdE0&export=download&authuser=0&confirm=t&resourcekey=0-V2m8tmPfD8eyNe4GGrhSxw' \
  -o build/data/detexify/symbols.json
curl -fL --create-dirs \
  'https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz' \
  -o build/data/mathwriting/mathwriting-2024.tgz
```

### 2. Prepare the data and metadata

```bash
uv run --no-sync python/data.py prepare
```

`prepare` runs the three required steps in order:

1. Convert the raw datasets into `build/data/_converted/data.parquet`.
2. Generate the pinned Typst catalog at `build/data/_converted/typst_symbols.json`.
3. Apply the mapping and generate frontend metadata.

The Parquet file keeps the source name, original LaTeX label, and vector strokes.
The metadata files are:

```text
build/data/_metadata/infer.json
build/data/_metadata/contrib.json
```

The mapping combines:

```text
UnicodeIt 0.7.5
    + Typst symbol catalog
    + python/detypify/assets/tex_to_typ_sup.yaml
```

`infer.json` contains the trained class order. `contrib.json` contains all
Typst aliases for the contribution UI.

### 3. Inspect the converted data (optional)

```bash
uv run --no-sync python/data.py preview
```

Open <http://127.0.0.1:8000>. The preview renders vector strokes on demand and
does not create another dataset file.

### 4. Train a model

```bash
uv run --no-sync python/train.py
```

The default is `mobilenet_v4_035`, 40 epochs, 224px input, and batch size 128.
Training creates deterministic train/test/validation Arrow caches and writes
the run under:

```text
build/train/mobilenet_v4_035/version_*/
```

Important outputs include:

```text
training_args.yaml
ckpts/best-*.ckpt
ckpts/last.ckpt
ckpts/best-*.onnx
events.out.tfevents.*
```

To train more than one model:

```bash
uv run --no-sync python/train.py \
  --models mobilenet_v4_035 \
  --models mobilenet_v4_050
```

For a GPU, use the matching extra and optionally enable CUDA-friendly mixed
precision, batch-size search, or compilation. See:

```bash
uv run --no-sync python/train.py --help
```

### 5. Evaluate an existing checkpoint

```bash
uv run --no-sync python/test.py \
  --ckpt-path build/train/mobilenet_v4_035/version_0/ckpts/last.ckpt
```

Evaluation logs are written under:

```text
build/train/_eval/existing_model/version_*/
```

The checkpoint must be evaluated with the same dataset selection and symbol
mapping used for training, because the sorted class order defines model output
indices.
