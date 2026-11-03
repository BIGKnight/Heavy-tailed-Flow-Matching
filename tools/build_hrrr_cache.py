"""
Build a local HRRR VIL cache for Phase-1 unconditional flow-matching experiments.

Streams hourly HRRR analysis (f00) GRIB2 from NOAA Open Data on AWS via Herbie,
extracts a single GRIB2 message (default: VIL = Vertically Integrated Liquid),
crops to a fixed 128x128 patch over the Central US (REGEN-compatible offset),
and writes per-year `.npy` arrays plus train-set z-score statistics.

Pandey et al. 2024 (Heavy-Tailed Diffusion Models) recipe:
  - source: HRRR analysis f00, hourly
  - product: wrfsfc (for VIL); GRIB message regex `:VIL:entire atmosphere:anl:`
  - region: 128x128 over Central US (we adopt REGEN's Oklahoma offset y0=834, x0=353)
  - normalization: per-channel z-score from train years only

Usage examples:
  # Dry run: one week
  python tools/build_hrrr_cache.py --start 2020-06-01 --end 2020-06-08 \
      --output_dir ./data/hrrr --workers 8

  # Full Phase-1 build (run overnight/in background; resumable)
  python tools/build_hrrr_cache.py --start 2019-01-01 --end 2022-01-01 \
      --output_dir ./data/hrrr --workers 16

The script is resumable: per-year arrays and a `progress.json` index are saved
incrementally. Re-running with the same args picks up where it left off.
"""

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import numpy as np


# Default region: REGEN/Manshausen 2024 Oklahoma window.
DEFAULT_Y0, DEFAULT_X0, DEFAULT_SIZE = 834, 353, 128

# Variable spec table. Add other channels by extending this dict.
VAR_SPECS = {
    'vil': dict(
        product='sfc',
        search=':VIL:entire atmosphere:anl:',
        units='kg m^-2',
    ),
    # w20: Pandey et al.'s "Vertical Wind Velocity at hybrid model level 20".
    # In HRRR GRIB2, this is the VVEL message at "20 hybrid level" inside the
    # wrfnat (native hybrid levels) product.
    'w20': dict(
        product='nat',
        search=':VVEL:20 hybrid level:anl:',
        units='Pa s^-1',
    ),
}


def _hourly_range(start_dt, end_dt):
    """Yield UTC datetimes hourly in [start, end)."""
    cur = start_dt
    while cur < end_dt:
        yield cur
        cur = cur + timedelta(hours=1)


def _ts_key(dt):
    return dt.strftime('%Y%m%d_%H')


def _ensure_year_array(year_path, n_hours, size, dtype=np.float32):
    """Open or create a memmap-backed per-year array of shape (n_hours, size, size).

    NaN-fill is used as 'not yet written' sentinel.
    """
    shape = (n_hours, size, size)
    if os.path.exists(year_path):
        arr = np.load(year_path, mmap_mode='r+')
        assert arr.shape == shape, f"shape mismatch at {year_path}: {arr.shape} != {shape}"
        return arr
    # Create.
    os.makedirs(os.path.dirname(year_path), exist_ok=True)
    arr = np.full(shape, np.nan, dtype=dtype)
    np.save(year_path, arr)
    # Reopen as memmap so writes persist.
    return np.load(year_path, mmap_mode='r+')


def _hours_in_year(year):
    """Number of hours in given year (handles leap years)."""
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    return int((end - start).total_seconds() // 3600)


def _hour_index_in_year(dt):
    year_start = datetime(dt.year, 1, 1, tzinfo=timezone.utc)
    return int((dt - year_start).total_seconds() // 3600)


def fetch_one(dt, var_spec, y0, x0, size, herbie_save_dir=None):
    """Download + extract + crop one hour. Returns (patch_np or None, err_msg or None)."""
    from herbie import Herbie

    try:
        H = Herbie(
            dt.strftime('%Y-%m-%d %H:%M'),
            model='hrrr',
            product=var_spec['product'],
            fxx=0,
            verbose=False,
            save_dir=herbie_save_dir,
        )
    except Exception as e:
        return None, f'herbie_init:{type(e).__name__}:{e}'

    if H.grib is None:
        return None, 'not_found_on_aws'

    try:
        ds = H.xarray(var_spec['search'], verbose=False, remove_grib=True)
    except Exception as e:
        return None, f'parse:{type(e).__name__}:{e}'

    var_names = list(ds.data_vars)
    if not var_names:
        return None, 'empty_dataset'
    arr = ds[var_names[0]].values
    if arr.ndim != 2:
        return None, f'bad_ndim:{arr.shape}'
    if arr.shape[0] < y0 + size or arr.shape[1] < x0 + size:
        return None, f'too_small:{arr.shape}'
    patch = arr[y0:y0 + size, x0:x0 + size].astype(np.float32, copy=False)
    if not np.isfinite(patch).all():
        return None, 'nonfinite_in_patch'
    return patch.copy(), None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', default='2019-01-01', help='inclusive UTC date YYYY-MM-DD')
    p.add_argument('--end', default='2022-01-01', help='exclusive UTC date YYYY-MM-DD')
    p.add_argument('--variable', default='vil', choices=list(VAR_SPECS.keys()))
    p.add_argument('--y0', type=int, default=DEFAULT_Y0)
    p.add_argument('--x0', type=int, default=DEFAULT_X0)
    p.add_argument('--size', type=int, default=DEFAULT_SIZE)
    p.add_argument('--output_dir', default='./data/hrrr')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--max_samples', type=int, default=None,
                   help='cap total fetched hours (for dry runs)')
    p.add_argument('--herbie_cache', default=None,
                   help='temporary scratch dir for Herbie GRIB partial files; '
                        'cleared periodically. Defaults to <output_dir>/_herbie_cache')
    p.add_argument('--train_years', default='2019,2020',
                   help='comma-separated years for stats computation')
    p.add_argument('--skip_stats', action='store_true',
                   help='skip post-build stats computation')
    p.add_argument('--stats_only', action='store_true',
                   help='skip download, only compute stats from existing arrays')
    args = p.parse_args()

    var_spec = VAR_SPECS[args.variable]
    var_dir = os.path.join(args.output_dir, args.variable)
    os.makedirs(var_dir, exist_ok=True)
    progress_path = os.path.join(var_dir, 'progress.json')
    missing_path = os.path.join(var_dir, 'missing.json')
    if args.herbie_cache is None:
        args.herbie_cache = os.path.join(args.output_dir, '_herbie_cache')
    os.makedirs(args.herbie_cache, exist_ok=True)

    # Load progress (set of completed timestamp keys) and missing index.
    progress = set()
    if os.path.exists(progress_path):
        with open(progress_path) as f:
            progress = set(json.load(f))
    missing = {}
    if os.path.exists(missing_path):
        with open(missing_path) as f:
            missing = json.load(f)

    start_dt = datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(args.end, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    assert start_dt < end_dt

    # Build per-year arrays lazily.
    years = sorted(set(range(start_dt.year, end_dt.year + 1)))
    year_arrays = {}
    for y in years:
        yp = os.path.join(var_dir, f'{y}.npy')
        year_arrays[y] = _ensure_year_array(yp, _hours_in_year(y), args.size)

    if args.stats_only:
        compute_stats(args, var_dir)
        return

    # Build worklist.
    work = [dt for dt in _hourly_range(start_dt, end_dt) if _ts_key(dt) not in progress]
    if args.max_samples is not None:
        work = work[:args.max_samples]
    print(f'[hrrr] var={args.variable} range={args.start}..{args.end} '
          f'workers={args.workers} pending={len(work)} done={len(progress)} '
          f'crop=(y0={args.y0}, x0={args.x0}, size={args.size})', flush=True)

    if not work:
        print('[hrrr] nothing to do.')
        if not args.skip_stats:
            compute_stats(args, var_dir)
        return

    t_start = time.time()
    completed = 0
    last_flush = time.time()
    failures = 0

    def _save_progress():
        with open(progress_path, 'w') as f:
            json.dump(sorted(progress), f)
        with open(missing_path, 'w') as f:
            json.dump(missing, f, indent=2)
        # Force-flush memmaps.
        for arr in year_arrays.values():
            if hasattr(arr, 'flush'):
                arr.flush()

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_to_dt = {
                ex.submit(fetch_one, dt, var_spec, args.y0, args.x0, args.size,
                          args.herbie_cache): dt
                for dt in work
            }
            for fut in as_completed(future_to_dt):
                dt = future_to_dt[fut]
                key = _ts_key(dt)
                try:
                    patch, err = fut.result()
                except Exception as e:
                    patch, err = None, f'unexpected:{type(e).__name__}:{e}'
                if patch is not None:
                    idx = _hour_index_in_year(dt)
                    year_arrays[dt.year][idx] = patch
                    progress.add(key)
                else:
                    missing[key] = err
                    failures += 1
                completed += 1

                # Periodic flush + progress log.
                now = time.time()
                if completed % 200 == 0 or now - last_flush > 60:
                    rate = completed / max(now - t_start, 1e-9)
                    eta = (len(work) - completed) / max(rate, 1e-9)
                    print(f'[hrrr] {completed}/{len(work)} done '
                          f'(failed={failures}) rate={rate:.2f}/s '
                          f'eta={eta/60:.1f}min', flush=True)
                    _save_progress()
                    # Periodic Herbie cache wipe to avoid disk bloat.
                    if completed % 2000 == 0:
                        _maybe_wipe(args.herbie_cache)
                    last_flush = now
    finally:
        _save_progress()
        _maybe_wipe(args.herbie_cache)

    print(f'[hrrr] finished. completed={completed} failed={failures} '
          f'elapsed={(time.time() - t_start)/60:.1f}min', flush=True)

    if not args.skip_stats:
        compute_stats(args, var_dir)


def _maybe_wipe(cache_dir):
    if not os.path.isdir(cache_dir):
        return
    try:
        for sub in os.listdir(cache_dir):
            full = os.path.join(cache_dir, sub)
            if os.path.isdir(full):
                shutil.rmtree(full, ignore_errors=True)
            else:
                try:
                    os.remove(full)
                except OSError:
                    pass
    except Exception:
        traceback.print_exc()


def compute_stats(args, var_dir):
    """Compute per-channel z-score stats from train years and save to stats.npz.

    Train-only (Pandey: stats computed on 2019-2020). Skips any NaN slot.
    Also writes a small summary of percentiles for tail QA.
    """
    train_years = [int(y) for y in args.train_years.split(',')]
    print(f'[hrrr] computing stats from years {train_years}', flush=True)
    chunks = []
    for y in train_years:
        yp = os.path.join(var_dir, f'{y}.npy')
        assert os.path.exists(yp), f'missing year array {yp}'
        arr = np.load(yp, mmap_mode='r')
        valid = np.isfinite(arr).reshape(arr.shape[0], -1).all(axis=-1)
        nv = int(valid.sum())
        if nv == 0:
            print(f'  year {y}: 0 valid hours, skipping')
            continue
        # Use a streaming-friendly running stats accumulator to avoid loading
        # the full year into RAM.
        s = 0.0
        s2 = 0.0
        n = 0
        for i in np.where(valid)[0]:
            x = arr[i].astype(np.float64)
            s += x.sum()
            s2 += (x * x).sum()
            n += x.size
        chunks.append((y, nv, s, s2, n))
        print(f'  year {y}: {nv} valid hours, {n} cells')

    if not chunks:
        print('[hrrr] no train data found, stats skipped.')
        return

    # Aggregate.
    n_total = sum(c[4] for c in chunks)
    s_total = sum(c[2] for c in chunks)
    s2_total = sum(c[3] for c in chunks)
    mean = s_total / n_total
    var = s2_total / n_total - mean * mean
    std = float(np.sqrt(max(var, 1e-20)))
    mean = float(mean)

    # Percentile QA on a random subsample of train data (for tail comparison).
    qa_vals = []
    rng = np.random.default_rng(0)
    sample_target = 4_000_000  # ~4M cells = ~250 frames; cheap
    per_chunk = sample_target // max(len(chunks), 1)
    for (y, nv, *_rest) in chunks:
        yp = os.path.join(var_dir, f'{y}.npy')
        arr = np.load(yp, mmap_mode='r')
        valid_idx = np.where(np.isfinite(arr).reshape(arr.shape[0], -1).all(axis=-1))[0]
        n_pick = min(len(valid_idx), max(per_chunk // (args.size * args.size), 1))
        pick = rng.choice(valid_idx, size=n_pick, replace=False)
        qa_vals.append(arr[pick].reshape(-1).astype(np.float32))
    qa = np.concatenate(qa_vals)
    qs = [50, 90, 99, 99.9, 99.99]
    qvals = np.percentile(qa, qs)
    print(f'[hrrr] mean={mean:.4f} std={std:.4f} '
          f'percentiles {dict(zip(qs, qvals.round(4).tolist()))}', flush=True)

    stats_path = os.path.join(var_dir, 'stats.npz')
    np.savez(
        stats_path,
        mean=np.float32(mean),
        std=np.float32(std),
        train_years=np.array(train_years, dtype=np.int32),
        percentiles=np.array(qs, dtype=np.float32),
        percentile_values=qvals.astype(np.float32),
        n_cells=np.int64(n_total),
    )
    print(f'[hrrr] saved stats -> {stats_path}', flush=True)


if __name__ == '__main__':
    main()
