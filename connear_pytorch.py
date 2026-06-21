"""Reusable PyTorch CoNNear cochlea model and feature extractor."""

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


def load_connear(
    weights_path: ArrayLikePath = "connear/Gmodel.pt",
    map_location: str | torch.device = "cpu",
) -> CoNNear:
    """Load CoNNear from a converted PyTorch state dict."""
    weights_path = Path(weights_path)
    if weights_path.suffix == ".h5":
        raise ValueError(
            "load_connear() expects converted PyTorch weights. "
            "Run convert_keras_to_pytorch.py first or use connear_conversion.py."
        )

    model = CoNNear()

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


@torch.no_grad()
def extract_features(
    audio: np.ndarray | Tensor,
    model: CoNNear | None = None,
    weights_path: ArrayLikePath = "connear/Gmodel.pt",
    device: str | torch.device | None = None,
    batch_size: int | None = None,
) -> np.ndarray | Tensor:
    """Extract CoNNear basilar-membrane features from audio.

    Input shape can be ``(samples,)``, ``(batch, samples)``, or
    ``(batch, samples, 1)``. NumPy input returns NumPy output; tensor input
    returns tensor output. Output shape is ``(batch, samples - 512, 201)``.
    """
    return_numpy = isinstance(audio, np.ndarray)
    target_device = torch.device(device) if device is not None else None
    if model is None:
        model = load_connear(weights_path, map_location=target_device or "cpu")
    elif target_device is not None:
        model = model.to(target_device)

    if return_numpy:
        return model.predict(
            _as_batched_channels_last(audio),
            batch_size=batch_size,
            device=target_device,
        )

    tensor = _as_batched_channels_last(audio)
    tensor = tensor.to(
        device=target_device or next(model.parameters()).device,
        dtype=next(model.parameters()).dtype,
    )
    was_training = model.training
    model.eval()
    outputs = []
    batch_size = batch_size or len(tensor)
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    for start in range(0, len(tensor), batch_size):
        outputs.append(model(tensor[start : start + batch_size]))
    if was_training:
        model.train()
    return torch.cat(outputs, dim=0)


def _as_batched_channels_last(audio: np.ndarray | Tensor) -> np.ndarray | Tensor:
    if audio.ndim == 1:
        audio = audio[None, :, None]
    elif audio.ndim == 2:
        audio = audio[:, :, None]
    elif audio.ndim != 3:
        raise ValueError(
            "audio must have shape "
            "(samples,), (batch, samples), or (batch, samples, 1)"
        )
    if audio.shape[-1] != 1:
        raise ValueError("audio must have exactly one channel in the last dimension")
    return audio
