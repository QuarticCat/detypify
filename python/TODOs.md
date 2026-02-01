# TODOs for python code

1. [x] Implement torch vision datasets and split, loading logic in `datasets.py`
2. [x] Split model details from `train.py` into `model.py`
3. [x] Use torch lightning's LightningDataModule for better split
4. [x] Adding arg parser for `train.py`
5. [x] Fix dataset's `__getitem__`
6. [ ] Remove `proc_font.py` (also `review_contrib.py`?) @QuaticCat
7. [ ] Add a custom validation callback hook to visualize wrong guesses. (needs test)
8. [ ] Fix infer always the same result problem.
9. [ ] Weight averaging function fixes. (Write a `get_ema_func` or just use the
   default implementation of torch lightning's with finetuned start epoch).
   Ref: `EMAv3` in timm and torch `get_ema_func`
