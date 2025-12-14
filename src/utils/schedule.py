# src.utils.schedule.py

import torch
import numpy as np


def get_default_device():
    """
    Prefer CUDA, but safely fall back to CPU if CUDA is unavailable or the build is CPU-only.
    """
    try:
        if torch.cuda.is_available() and torch.version.cuda is not None:
            return torch.device("cuda")
    except Exception:
        pass

    # 2. MPS (Apple Silicon)
    # try:
    #     if torch.backends.mps.is_available():
    #         return torch.device("mps")
    # except Exception:
    #     pass

    # 3. CPU
    return torch.device("cpu")


# get torch device
device = get_default_device()

# helper functions
def f(t, T, s=0.0001):
    return np.cos((np.pi * (t / T + s)) / (2 * (1 + s))) ** 2

def g(t, T):
    return f(t, T) / f(0, T)

def h(t, T):
    return 1 - (g(t, T)) / (g(t - 1, T))

def make_schedule(beta0, betaT, T, schedule, schedule_exponent, device):

    if schedule == 'linear': betas = np.insert(np.linspace(beta0, betaT, T+1), 0, 0)
    elif schedule == 'cosine': betas = np.clip(h(np.linspace(0, T, T+1)**schedule_exponent, T), 0., 0.999)

    alphas_bar = np.cumprod(1-betas)

    return torch.tensor(betas).float().to(device), torch.tensor(alphas_bar).float().to(device)
