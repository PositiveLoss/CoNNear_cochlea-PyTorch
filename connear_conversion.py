"""Utilities for converting original CoNNear Keras weights to PyTorch."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import Tensor


ArrayLikePath = Union[str, Path]

KERAS_TO_TORCH_WEIGHT_MAP = {
    "enc1.weight": ("model_1/conv1d_1/kernel:0", (2, 1, 0)),
    "enc2.weight": ("model_1/conv1d_2/kernel:0", (2, 1, 0)),
    "enc3.weight": ("model_1/conv1d_3/kernel:0", (2, 1, 0)),
    "enc4.weight": ("model_1/conv1d_4/kernel:0", (2, 1, 0)),
    "dec1.weight": ("model_1/conv2d_transpose_1/kernel:0", (2, 1, 0)),
    "dec2.weight": ("model_1/conv2d_transpose_2/kernel:0", (2, 1, 0)),
    "dec3.weight": ("model_1/conv2d_transpose_3/kernel:0", (2, 1, 0)),
    "dec4.weight": ("model_1/conv2d_transpose_4/kernel:0", (2, 1, 0)),
}


def keras_h5_to_state_dict(h5_path: ArrayLikePath) -> dict[str, Tensor]:
    """Convert the original Keras ``Gmodel.h5`` weights into a PyTorch state dict."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to convert Keras .h5 weights") from exc

    state_dict = {}
    with h5py.File(Path(h5_path), "r") as h5_file:
        for torch_name, (keras_name, axes) in KERAS_TO_TORCH_WEIGHT_MAP.items():
            weights = np.asarray(h5_file[keras_name])
            if weights.ndim == 4:
                weights = weights[:, 0, :, :]
            weights = np.transpose(weights, axes).copy()
            state_dict[torch_name] = torch.from_numpy(weights)
    return state_dict


def save_converted_weights(
    h5_path: ArrayLikePath = "connear/Gmodel.h5",
    output_path: ArrayLikePath = "connear/Gmodel.pt",
) -> Path:
    """Write a converted PyTorch state dict from the original Keras weights."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(keras_h5_to_state_dict(h5_path), output_path)
    return output_path
