"""DNA sequence ↔ one-hot encoding conversions and transforms.

Provides canonical numpy implementations (``sequence_to_onehot``,
``onehot_to_sequence``) and thin torch wrappers (``sequence_to_onehot_tensor``,
``onehot_tensor_to_sequence``) for use throughout the package.

Encoding mapping::

    A → [1, 0, 0, 0]
    C → [0, 1, 0, 0]
    G → [0, 0, 1, 0]
    T → [0, 0, 0, 1]
    N / other → [0, 0, 0, 0]   (all-zeros, matching JAX reference)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

# A=0, C=1, G=2, T=3
_BASES = "ACGT"

# Build lookup table once (128 entries covers ASCII)
_ENCODE_LOOKUP = np.full(128, -1, dtype=np.int8)
for _i, _ch in enumerate(_BASES):
    _ENCODE_LOOKUP[ord(_ch)] = _i
    _ENCODE_LOOKUP[ord(_ch.lower())] = _i


def sequence_to_onehot(sequence: str) -> np.ndarray:
    """Convert a DNA sequence string to a one-hot encoded numpy array.

    Handles both upper- and lower-case nucleotides.
    Ambiguous / unknown bases (e.g. ``N``) are encoded as all-zeros.

    Args:
        sequence: DNA sequence string (``ACGTN``).

    Returns:
        One-hot encoded ``uint8`` array of shape ``(len(sequence), 4)``.
    """
    seq_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    onehot = np.zeros((len(seq_bytes), 4), dtype=np.uint8)
    # clip(0, 127) prevents crash on non-ASCII if present
    indices = _ENCODE_LOOKUP[seq_bytes.clip(0, 127)]
    mask = indices >= 0
    onehot[np.where(mask)[0], indices[mask]] = 1
    return onehot


def onehot_to_sequence(onehot: np.ndarray) -> str:
    """Convert a one-hot encoded array back to a DNA sequence string.

    All-zero rows (ambiguous bases) are decoded as ``N``.

    Args:
        onehot: Array of shape ``(L, 4)`` with one-hot encoding.

    Returns:
        DNA sequence string of length ``L``.
    """
    if onehot.ndim != 2 or onehot.shape[1] != 4:
        raise ValueError(f"Expected shape (L, 4), got {onehot.shape}")

    bases = np.array(list(_BASES + "N"))  # index 4 → 'N'
    # All-zero rows → argmax returns 0, but we want 'N'
    is_valid = onehot.any(axis=1)
    indices = onehot.argmax(axis=1)
    indices = np.where(is_valid, indices, 4)
    return "".join(bases[indices])


def reverse_complement_onehot(onehot: np.ndarray) -> np.ndarray:
    """Reverse complement a channels-last one-hot encoded DNA sequence.

    Args:
        onehot: Array of shape ``(L, 4)`` or ``(B, L, 4)`` in A/C/G/T order.

    Returns:
        Reverse-complemented array with the same shape and dtype.
    """
    if onehot.ndim == 2:
        return onehot[::-1, [3, 2, 1, 0]].copy()
    if onehot.ndim == 3:
        return onehot[:, ::-1, :][:, :, [3, 2, 1, 0]].copy()
    raise ValueError(f"Expected shape (L, 4) or (B, L, 4), got {onehot.shape}")


def shift_onehot(
    onehot: np.ndarray,
    shift: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Shift a channels-last one-hot DNA sequence along the length axis.

    This mirrors Baskerville's sequence-shift semantics: positive shifts move
    sequence content to the right and pad the left; negative shifts move content
    to the left and pad the right.

    Args:
        onehot: Array of shape ``(L, 4)`` or ``(B, L, 4)``.
        shift: Signed shift in bp.
        pad_value: Value used for padded positions. AlphaGenome unknown bases
            are represented as all-zeros, so the default is ``0.0``.

    Returns:
        Shifted array with the same shape and dtype.
    """
    if onehot.ndim not in (2, 3):
        raise ValueError(f"Expected shape (L, 4) or (B, L, 4), got {onehot.shape}")
    if onehot.shape[-1] != 4:
        raise ValueError(f"Expected last dimension of size 4, got {onehot.shape}")
    if shift == 0:
        return onehot

    length_axis = 0 if onehot.ndim == 2 else 1
    length = onehot.shape[length_axis]
    if abs(shift) >= length:
        return np.full_like(onehot, pad_value)

    shifted = np.full_like(onehot, pad_value)
    if onehot.ndim == 2:
        if shift > 0:
            shifted[shift:, :] = onehot[:-shift, :]
        else:
            shifted[:shift, :] = onehot[-shift:, :]
    else:
        if shift > 0:
            shifted[:, shift:, :] = onehot[:, :-shift, :]
        else:
            shifted[:, :shift, :] = onehot[:, -shift:, :]
    return shifted


# ---------------------------------------------------------------------------
# Torch wrappers
# ---------------------------------------------------------------------------


def sequence_to_onehot_tensor(
    sequence: str,
    dtype: "torch.dtype | None" = None,
    device: "torch.device | str | None" = None,
) -> "torch.Tensor":
    """Convert DNA sequence string to a one-hot encoded torch tensor.

    Thin wrapper around :func:`sequence_to_onehot` that converts the result
    to a :class:`torch.Tensor` with the requested dtype and device.

    Args:
        sequence: DNA sequence string (``ACGTN``).
        dtype: Output tensor dtype. Defaults to ``torch.float32``.
        device: Output tensor device.

    Returns:
        One-hot encoded tensor of shape ``(len(sequence), 4)``.
    """
    import torch as _torch

    np_onehot = sequence_to_onehot(sequence)
    tensor = _torch.from_numpy(np_onehot.astype(np.float32))
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def onehot_tensor_to_sequence(onehot: "torch.Tensor") -> str:
    """Convert a one-hot encoded torch tensor back to a DNA sequence string.

    Accepts tensors of shape ``(L, 4)`` or ``(B, L, 4)`` (takes first batch
    element).

    Args:
        onehot: One-hot tensor of shape ``(L, 4)`` or ``(B, L, 4)``.

    Returns:
        DNA sequence string of length ``L``.
    """
    if onehot.dim() == 3:
        onehot = onehot[0]
    return onehot_to_sequence(onehot.detach().cpu().numpy())


def reverse_complement_onehot_tensor(onehot: "torch.Tensor") -> "torch.Tensor":
    """Torch equivalent of :func:`reverse_complement_onehot`."""
    import torch as _torch

    if onehot.dim() == 2:
        return _torch.flip(onehot, dims=[0]).index_select(
            -1,
            _torch.tensor([3, 2, 1, 0], device=onehot.device),
        )
    if onehot.dim() == 3:
        return _torch.flip(onehot, dims=[1]).index_select(
            -1,
            _torch.tensor([3, 2, 1, 0], device=onehot.device),
        )
    raise ValueError(f"Expected shape (L, 4) or (B, L, 4), got {tuple(onehot.shape)}")


def shift_onehot_tensor(
    onehot: "torch.Tensor",
    shift: int,
    pad_value: float = 0.0,
) -> "torch.Tensor":
    """Torch equivalent of :func:`shift_onehot`."""
    import torch as _torch

    if onehot.dim() not in (2, 3) or onehot.shape[-1] != 4:
        raise ValueError(f"Expected shape (L, 4) or (B, L, 4), got {tuple(onehot.shape)}")
    if shift == 0:
        return onehot

    length_axis = 0 if onehot.dim() == 2 else 1
    length = onehot.shape[length_axis]
    shifted = _torch.full_like(onehot, pad_value)
    if abs(shift) >= length:
        return shifted

    if onehot.dim() == 2:
        if shift > 0:
            shifted[shift:, :] = onehot[:-shift, :]
        else:
            shifted[:shift, :] = onehot[-shift:, :]
    else:
        if shift > 0:
            shifted[:, shift:, :] = onehot[:, :-shift, :]
        else:
            shifted[:, :shift, :] = onehot[:, -shift:, :]
    return shifted


__all__ = [
    "sequence_to_onehot",
    "onehot_to_sequence",
    "reverse_complement_onehot",
    "shift_onehot",
    "sequence_to_onehot_tensor",
    "onehot_tensor_to_sequence",
    "reverse_complement_onehot_tensor",
    "shift_onehot_tensor",
]
