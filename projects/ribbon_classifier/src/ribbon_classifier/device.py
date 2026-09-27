"""Choose where tensors live: an NVIDIA GPU (CUDA), an Apple GPU (MPS), or the CPU."""

from __future__ import annotations

import random

import numpy as np
import torch


def pick_device(preference: str = "auto") -> torch.device:
    """``auto`` picks the fastest available device; otherwise ``cuda``/``mps``/``cpu``."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return f"cuda ({props.name}, {props.total_memory / 2**30:.1f} GB)"
    if device.type == "mps":
        return "mps (Apple GPU)"
    return f"cpu ({torch.get_num_threads()} threads) - fine for the probe and small runs; see docs for GPU options"


def seed_everything(seed: int) -> None:
    """Make runs repeatable (bit-exact only on the same hardware and library versions)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
