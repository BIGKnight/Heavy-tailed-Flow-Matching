# Data

All loaders look under `./data` (relative to the repository root). Only the
fixed CIFAR10-LT subset index is versioned; everything else is downloaded or
built locally and ignored by git.

```
data/
├── splits/cifar10_lt_max5000_ratio0.01_classes10_seed42.json   # versioned
├── cifar10/        # torchvision CIFAR-10 train split (auto-download)
├── cifar10_test/   # torchvision CIFAR-10 test split (auto-download)
└── hrrr/vil/       # HRRR VIL cache built with tools/build_hrrr_cache.py
```

## 2D α-stable mixtures

Generated on the fly from the `data:` block of the config (`dataset: sas_grid`,
`n_mixture: 9`, `grid_spacing: 2.0`, `std: 0.1`, `data_alpha`, `weights`).
Nothing to download. Training draws `nsamples` points; evaluation draws
`eval.real_data` fresh reference points.

## CIFAR10-LT

`dataset: cifar10_lt` downloads CIFAR-10 through torchvision into
`data/cifar10` and `data/cifar10_test` on first use, then selects the
long-tailed subset described by `data.long_tail`
(`imbalance_ratio: 0.01`, `max_count: 5000`, `num_classes: 10`, `seed: 42`).
The per-class indices are read from `data/splits/…seed42.json`
(12,406 images), so every run and every baseline sees the identical subset.
If the JSON is missing it is rebuilt deterministically from the same seed and
validated against the expected per-class counts.

## HRRR VIL

Hourly Vertically Integrated Liquid analyses from NOAA's High-Resolution Rapid
Refresh archive, cropped to a 128×128 window over the Central US following
Pandey et al. (2024). Build the cache once (requires the optional
[`herbie-data`](https://github.com/blaylockbk/Herbie) package and network
access to the NOAA Open Data bucket on AWS):

```bash
pip install herbie-data
python tools/build_hrrr_cache.py --start 2019-01-01 --end 2022-01-01 \
    --output_dir ./data/hrrr --workers 16
```

The script is resumable and writes per-year `.npy` arrays plus train-year
z-score statistics under `data/hrrr/vil/`. Configs use
`train_years: [2019, 2020]` (≈17,400 valid hourly fields) and
`test_years: [2021]` (≈8,700). Expect roughly 15 GB on disk and several hours
of download time for the full range; use a one-week range first to check the
setup.
