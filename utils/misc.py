"""Seeding, config loading, padding and environment description."""

import os
import platform
import random
import subprocess

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def load_config(path: str) -> dict:
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def pad_to_multiple(x: torch.Tensor, multiple: int = 16):
    """Reflect-pad (B,C,H,W) so that H and W are multiples of `multiple`."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='reflect')
    return x, (h, w)


def describe_environment(device: str = 'cpu') -> dict:
    env = {'python': platform.python_version(), 'torch': torch.__version__,
           'platform': platform.platform(), 'cpu': platform.processor() or platform.machine(),
           'threads': torch.get_num_threads()}
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    env['cpu'] = line.split(':', 1)[1].strip()
                    break
    except OSError:
        pass
    if device.startswith('cuda') and torch.cuda.is_available():
        env['gpu'] = torch.cuda.get_device_name(0)
        env['cuda'] = torch.version.cuda
        env['cudnn'] = torch.backends.cudnn.version()
    try:
        env['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).decode().strip()
    except Exception:
        pass
    return env
