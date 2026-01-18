# Detypify Model

Data preprocessing and model training scripts for Detypify.

## Development

In project root (not this folder):

```console
# extract mathwriting dataset
$ tar xavf external/dataset/mathwriting-2024-symbols.tar.zst -C external/dataset

$ uv sync                     # install dependencies
$ uv run python/proc_data.py  # preprocess data
$ uv run python/train.py      # train model
```
