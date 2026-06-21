"""PyTorch implementation of the pretrained CoNNear cochlea model."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


ArrayLikePath = Union[str, Path]
CONTEXT_SAMPLES = 256
KERNEL_SIZE = 64
STRIDE = 2
HIDDEN_CHANNELS = 128
OUTPUT_CHANNELS = 201

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


def _same_pad_1d(x: Tensor, kernel_size: int, stride: int) -> Tensor:
    """Apply TensorFlow/Keras ``padding='same'`` padding for Conv1D."""
    input_length = x.shape[-1]
    output_length = (input_length + stride - 1) // stride
    pad_total = max((output_length - 1) * stride + kernel_size - input_length, 0)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return F.pad(x, (pad_left, pad_right))


class KerasSameConv1d(nn.Conv1d):
    """Conv1d with TensorFlow/Keras ``padding='same'`` semantics."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(_same_pad_1d(x, self.kernel_size[0], self.stride[0]))


class CoNNear(nn.Module):
    """Pretrained CoNNear architecture ported from the original Keras model.

    The PyTorch ``forward`` method accepts tensors in either Keras layout
    ``(batch, samples, channels)`` or native PyTorch layout
    ``(batch, channels, samples)``. It returns Keras layout by default so the
    original notebooks can keep using ``connear.predict(...)``-style arrays.
    """

    def __init__(self, output_channels: int = OUTPUT_CHANNELS):
        super().__init__()
        self.enc1 = KerasSameConv1d(1, HIDDEN_CHANNELS, KERNEL_SIZE, STRIDE)
        self.enc2 = KerasSameConv1d(
            HIDDEN_CHANNELS, HIDDEN_CHANNELS, KERNEL_SIZE, STRIDE
        )
        self.enc3 = KerasSameConv1d(
            HIDDEN_CHANNELS, HIDDEN_CHANNELS, KERNEL_SIZE, STRIDE
        )
        self.enc4 = KerasSameConv1d(
            HIDDEN_CHANNELS, HIDDEN_CHANNELS, KERNEL_SIZE, STRIDE
        )

        self.dec1 = nn.ConvTranspose1d(
            HIDDEN_CHANNELS,
            HIDDEN_CHANNELS,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=31,
            bias=False,
        )
        self.dec2 = nn.ConvTranspose1d(
            HIDDEN_CHANNELS * 2,
            HIDDEN_CHANNELS,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=31,
            bias=False,
        )
        self.dec3 = nn.ConvTranspose1d(
            HIDDEN_CHANNELS * 2,
            HIDDEN_CHANNELS,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=31,
            bias=False,
        )
        self.dec4 = nn.ConvTranspose1d(
            HIDDEN_CHANNELS * 2,
            output_channels,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=31,
            bias=False,
        )

    def forward(self, x: Tensor, channels_last: bool = True) -> Tensor:
        if channels_last:
            x = x.transpose(1, 2)

        c1 = self.enc1(x)
        a1 = torch.tanh(c1)
        c2 = self.enc2(a1)
        a2 = torch.tanh(c2)
        c3 = self.enc3(a2)
        a3 = torch.tanh(c3)
        c4 = self.enc4(a3)
        x = torch.tanh(c4)

        x = torch.tanh(self.dec1(x))
        x = torch.cat([x, c3], dim=1)
        x = torch.tanh(self.dec2(x))
        x = torch.cat([x, c2], dim=1)
        x = torch.tanh(self.dec3(x))
        x = torch.cat([x, c1], dim=1)
        x = self.dec4(x)
        x = x[..., CONTEXT_SAMPLES:-CONTEXT_SAMPLES]

        if channels_last:
            x = x.transpose(1, 2)
        return x

    @torch.no_grad()
    def predict(
        self,
        x: np.ndarray,
        verbose: int = 0,
        batch_size: int | None = None,
        device: str | torch.device | None = None,
    ) -> np.ndarray:
        """Keras-compatible NumPy prediction helper."""
        del verbose
        if len(x) == 0:
            raise ValueError("predict() requires at least one input sample")
        was_training = self.training
        self.eval()
        target_device = (
            torch.device(device)
            if device is not None
            else next(self.parameters()).device
        )
        batch_size = batch_size or len(x)
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        outputs = []
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(
                x[start : start + batch_size], dtype=torch.float32, device=target_device
            )
            outputs.append(self(batch).cpu().numpy())
        if was_training:
            self.train()
        return np.concatenate(outputs, axis=0)

    def summary(self) -> None:
        """Small notebook-friendly replacement for Keras ``model.summary()``."""
        total = sum(param.numel() for param in self.parameters())
        trainable = sum(
            param.numel() for param in self.parameters() if param.requires_grad
        )
        print(self)
        print(f"Total params: {total:,}")
        print(f"Trainable params: {trainable:,}")


def keras_h5_to_state_dict(h5_path: ArrayLikePath) -> dict[str, Tensor]:
    """Convert the original Keras ``Gmodel.h5`` weights into a PyTorch state dict."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required to convert Keras .h5 weights") from exc

    h5_path = Path(h5_path)
    state_dict = {}
    with h5py.File(h5_path, "r") as h5_file:
        for torch_name, (keras_name, axes) in KERAS_TO_TORCH_WEIGHT_MAP.items():
            weights = np.asarray(h5_file[keras_name])
            if weights.ndim == 4:
                weights = weights[:, 0, :, :]
            weights = np.transpose(weights, axes).copy()
            state_dict[torch_name] = torch.from_numpy(weights)
    return state_dict


def load_connear(
    weights_path: ArrayLikePath = "connear/Gmodel.pt",
    map_location: str | torch.device = "cpu",
) -> CoNNear:
    """Load CoNNear from a converted PyTorch state dict or the original Keras HDF5 file."""
    weights_path = Path(weights_path)
    model = CoNNear()

    if weights_path.suffix == ".h5":
        state_dict = keras_h5_to_state_dict(weights_path)
    else:
        try:
            state_dict = torch.load(
                weights_path, map_location=map_location, weights_only=True
            )
        except TypeError:
            state_dict = torch.load(weights_path, map_location=map_location)

    model.load_state_dict(state_dict)
    model.to(map_location)
    model.eval()
    return model


def save_converted_weights(
    h5_path: ArrayLikePath = "connear/Gmodel.h5",
    output_path: ArrayLikePath = "connear/Gmodel.pt",
) -> Path:
    """Write a converted PyTorch state dict from the original Keras weights."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(keras_h5_to_state_dict(h5_path), output_path)
    return output_path
