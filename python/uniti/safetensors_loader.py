"""
Native safetensors loader — zero external dependencies beyond numpy.

The safetensors format (https://huggingface.co/docs/safetensors) is simple:
  [8 bytes: header_size (uint64 LE)]
  [header_size bytes: JSON metadata]
  [remaining bytes: raw tensor data]

The JSON header maps each tensor name to:
  {"dtype": str, "shape": list[int], "data_offsets": [begin, end]}

This module reads the file directly and returns numpy float32 arrays,
eliminating the need for the `safetensors` and `torch` packages.
"""

import struct
import json
import numpy as np
from typing import Dict


# safetensors dtype string → (numpy dtype, element size in bytes)
_DTYPE_MAP = {
    "F64":  (np.float64, 8),
    "F32":  (np.float32, 4),
    "F16":  (np.float16, 2),
    "BF16": (None, 2),          # no native numpy BF16; handled specially
    "I64":  (np.int64, 8),
    "I32":  (np.int32, 4),
    "I16":  (np.int16, 2),
    "I8":   (np.int8, 1),
    "U8":   (np.uint8, 1),
    "BOOL": (np.bool_, 1),
}


def _bf16_to_f32(raw: bytes, shape: tuple) -> np.ndarray:
    """Convert raw BF16 bytes to float32 numpy array.

    BF16 is the upper 16 bits of IEEE 754 float32, so we just
    left-shift each uint16 by 16 bits into a uint32 view of float32.
    """
    arr_u16 = np.frombuffer(raw, dtype=np.uint16)
    arr_u32 = arr_u16.astype(np.uint32) << 16
    arr_f32 = arr_u32.view(np.float32)
    return arr_f32.reshape(shape)


def load_safetensors(filepath: str, to_float32: bool = True) -> Dict[str, np.ndarray]:
    """Load all tensors from a single safetensors file.

    Parameters
    ----------
    filepath : str
        Path to the ``.safetensors`` file.
    to_float32 : bool
        If True (default), cast every tensor to float32.
        This mirrors the behaviour of the old torch-based loader.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from tensor name to numpy array.
    """
    tensors: Dict[str, np.ndarray] = {}

    with open(filepath, "rb") as f:
        # ── 1. Read header size (8 bytes, little-endian uint64) ──────────
        header_size = struct.unpack("<Q", f.read(8))[0]

        # ── 2. Parse JSON header ────────────────────────────────────────
        header = json.loads(f.read(header_size))

        # Byte offset where tensor data begins
        data_start = 8 + header_size

        # ── 3. Read each tensor ─────────────────────────────────────────
        for name, info in header.items():
            if name == "__metadata__":
                continue

            dtype_str: str = info["dtype"]
            shape = tuple(info["shape"])
            begin, end = info["data_offsets"]

            raw_size = end - begin
            f.seek(data_start + begin)
            raw = f.read(raw_size)

            # --- BF16 special path ---
            if dtype_str == "BF16":
                tensors[name] = _bf16_to_f32(raw, shape)
                continue

            # --- Normal dtypes ---
            np_dtype, _ = _DTYPE_MAP.get(dtype_str, (np.float32, 4))
            arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)

            if to_float32 and arr.dtype != np.float32:
                arr = arr.astype(np.float32)

            # Ensure array is writable (frombuffer returns read-only views)
            if not arr.flags.writeable:
                arr = arr.copy()

            tensors[name] = arr

    return tensors


def load_safetensors_sharded(model_path: str, to_float32: bool = True) -> Dict[str, np.ndarray]:
    """Load tensors from a potentially sharded safetensors model directory.

    If ``model.safetensors`` exists, load that single file.
    Otherwise, look for ``model.safetensors.index.json`` and load
    all shards referenced therein.

    Parameters
    ----------
    model_path : str
        Directory containing the model files.
    to_float32 : bool
        If True, cast all tensors to float32.

    Returns
    -------
    dict[str, np.ndarray]
    """
    import os

    single = os.path.join(model_path, "model.safetensors")
    if os.path.isfile(single):
        return load_safetensors(single, to_float32=to_float32)

    # Sharded model
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(index_file):
        raise FileNotFoundError(
            f"Neither model.safetensors nor model.safetensors.index.json "
            f"found in {model_path}"
        )

    with open(index_file) as f:
        index = json.load(f)

    weight_map = index["weight_map"]  # name → shard filename
    shard_files = sorted(set(weight_map.values()))

    tensors: Dict[str, np.ndarray] = {}
    for shard_name in shard_files:
        shard_path = os.path.join(model_path, shard_name)
        shard_tensors = load_safetensors(shard_path, to_float32=to_float32)
        tensors.update(shard_tensors)

    return tensors
