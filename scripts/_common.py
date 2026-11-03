"""Shared plumbing for the reproduction orchestrators in scripts/.

Nothing here is experiment-specific: timestamps, GPU discovery, a FIFO worker pool with
``lanes`` concurrent jobs per GPU, and the environment handed to child processes.
"""
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable


def ts():
    return datetime.now().strftime('%m-%d %H:%M:%S')


def log(msg):
    print(f'[{ts()}] {msg}', flush=True)


def default_gpus():
    """All visible GPUs (torch if importable, else nvidia-smi, else [0])."""
    try:
        import torch
        n = torch.cuda.device_count()
        if n:
            return list(range(n))
    except Exception:
        pass
    try:
        out = subprocess.check_output(['nvidia-smi', '-L'], text=True)
        n = len([l for l in out.splitlines() if l.startswith('GPU ')])
        if n:
            return list(range(n))
    except Exception:
        pass
    return [0]


def add_gpu_args(parser, lanes_default=1):
    parser.add_argument('--gpus', type=int, nargs='+', default=None,
                        help='GPU ids to use (default: all visible)')
    parser.add_argument('--lanes', type=int, default=lanes_default,
                        help='concurrent jobs per GPU')


def resolve_gpus(args):
    return list(args.gpus) if args.gpus else default_gpus()


def child_env(gpu, **extra):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = str(REPO) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env.update({k: str(v) for k, v in extra.items()})
    return env


def free_mb(gpu):
    """Free memory of a GPU in MB via nvidia-smi; None if unavailable."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '-i', str(gpu)],
            text=True)
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def run_pool(jobs, gpus, lanes, run_job):
    """Run ``jobs`` FIFO over ``len(gpus) * lanes`` worker threads.

    ``run_job(job, gpu, lane, launch_lock)`` does the work for one job; ``launch_lock`` is a
    per-GPU lock the callee may hold while starting a process (same-GPU launch stagger).
    Returns [(job, result)] in completion order.
    """
    q = Queue()
    for j in jobs:
        q.put(j)
    locks = {g: threading.Lock() for g in gpus}
    results, res_lock = [], threading.Lock()

    def worker(gpu, lane):
        while True:
            try:
                job = q.get_nowait()
            except Empty:
                return
            try:
                r = run_job(job, gpu, lane, locks[gpu])
            except Exception as e:          # keep the pool alive, record the failure
                r = f'EXCEPTION {e!r}'
                log(f'worker gpu{gpu} lane{lane}: {r}')
            with res_lock:
                results.append((job, r))

    threads = [threading.Thread(target=worker, args=(g, l), daemon=True)
               for g in gpus for l in range(lanes)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results
