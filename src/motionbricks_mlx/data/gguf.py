"""Minimal, read-only GGUF v3 reader for MotionBricks bundles.

Written for this repository because the reference runtime's bundle format is
GGUF v3 and we need the *exact* tensor names, shapes and bytes to cross-check
the MLX conversion (see ``docs/decisions/0003-downstream-runtime.md``).

Only what the MotionBricks bundles actually use is implemented:

* GGUF v3, little-endian;
* all scalar metadata value types, plus arrays;
* F32 / F16 / BF16 / I32 tensor payloads (the bundle writer emits only these).

The GGML linear layout of a tensor is ``i0 + ne0*(i1 + ne1*(i2 + ...))``, i.e.
the *last* GGML axis varies fastest.  A C-contiguous NumPy array of shape
``tuple(reversed(ne))`` therefore has exactly the same memory layout, which is
also PyTorch's layout for the same weights (``nn.Linear.weight`` is
``[out, in]`` while GGML stores ``ne = (in, out)``).
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

# ggml_type enum values (ggml v0.20.x, the revision pinned by the reference).
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_I32 = 26
GGML_TYPE_BF16 = 30

_TENSOR_DTYPES: dict[int, np.dtype] = {
    GGML_TYPE_F32: np.dtype("<f4"),
    GGML_TYPE_F16: np.dtype("<f2"),
    GGML_TYPE_I32: np.dtype("<i4"),
    # BF16 has no NumPy dtype; handled explicitly in `tensor`.
    GGML_TYPE_BF16: np.dtype("<u2"),
}

TENSOR_TYPE_NAMES: dict[int, str] = {
    GGML_TYPE_F32: "F32",
    GGML_TYPE_F16: "F16",
    GGML_TYPE_I32: "I32",
    GGML_TYPE_BF16: "BF16",
}

# gguf_metadata_value_type enum values.
_UINT8, _INT8, _UINT16, _INT16 = 0, 1, 2, 3
_UINT32, _INT32, _FLOAT32, _BOOL = 4, 5, 6, 7
_STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 8, 9, 10, 11, 12

_SCALAR_STRUCT: dict[int, str] = {
    _UINT8: "<B",
    _INT8: "<b",
    _UINT16: "<H",
    _INT16: "<h",
    _UINT32: "<I",
    _INT32: "<i",
    _FLOAT32: "<f",
    _BOOL: "<?",
    _UINT64: "<Q",
    _INT64: "<q",
    _FLOAT64: "<d",
}


class GgufError(RuntimeError):
    """Raised when a file is not a readable GGUF v3 container."""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dimensions: tuple[int, ...]  # GGML order: ne[0] first.
    type_id: int
    offset: int  # byte offset relative to the start of the data section.

    @property
    def type_name(self) -> str:
        return TENSOR_TYPE_NAMES.get(self.type_id, f"type{self.type_id}")

    @property
    def shape(self) -> tuple[int, ...]:
        """NumPy/C-contiguous shape (GGML axes reversed)."""
        return tuple(reversed(self.dimensions))

    @property
    def element_count(self) -> int:
        result = 1
        for value in self.dimensions:
            result *= value
        return result

    @property
    def byte_size(self) -> int:
        if self.type_id == GGML_TYPE_BF16 or self.type_id == GGML_TYPE_F16:
            return self.element_count * 2
        if self.type_id == GGML_TYPE_F32 or self.type_id == GGML_TYPE_I32:
            return self.element_count * 4
        raise GgufError(f"unsupported ggml type {self.type_id} for {self.name}")


def _bf16_to_float32(raw: np.ndarray) -> np.ndarray:
    """BF16 bit pattern -> float32 (exact widening, no rounding involved)."""
    widened = raw.astype(np.uint32) << 16
    return widened.view(np.float32)


class GgufFile:
    """Read-only GGUF v3 container."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._stream = self.path.open("rb")
        raw = self.path.read_bytes()
        self._data = memoryview(raw)
        self._parse()
        self.sha256 = hashlib.sha256(raw).hexdigest()

    # -- parsing ---------------------------------------------------------

    def _parse(self) -> None:
        data = self._data
        if len(data) < 24 or bytes(data[:4]) != GGUF_MAGIC:
            raise GgufError(f"{self.path}: not a GGUF file")
        position = 4
        version, tensor_count, kv_count = struct.unpack_from("<IQQ", data, position)
        position += 20
        if version != GGUF_VERSION:
            raise GgufError(f"{self.path}: unsupported GGUF version {version}")
        self.version = version
        self.metadata: dict[str, Any] = {}
        for _ in range(kv_count):
            key, position = self._read_string(position)
            self.metadata[key] = self._read_value(position)
            position = self._value_end
        infos: list[TensorInfo] = []
        for _ in range(tensor_count):
            name, position = self._read_string(position)
            (n_dims,) = struct.unpack_from("<I", data, position)
            position += 4
            dimensions = struct.unpack_from("<" + "Q" * n_dims, data, position)
            position += 8 * n_dims
            type_id, offset = struct.unpack_from("<IQ", data, position)
            position += 12
            infos.append(TensorInfo(name, tuple(int(v) for v in dimensions), int(type_id), int(offset)))
        self.alignment = int(self.metadata.get("general.alignment", 32))
        self._tensor_infos = {info.name: info for info in infos}
        if len(self._tensor_infos) != len(infos):
            raise GgufError(f"{self.path}: duplicate tensor names")
        self.data_offset = (position + self.alignment - 1) // self.alignment * self.alignment
        self._order = [info.name for info in infos]

    def _read_string(self, position: int) -> tuple[str, int]:
        (length,) = struct.unpack_from("<Q", self._data, position)
        position += 8
        raw = bytes(self._data[position : position + length])
        self._value_end = position + length
        return raw.decode("utf-8"), position + length

    def _read_value(self, position: int) -> Any:
        (value_type,) = struct.unpack_from("<I", self._data, position)
        position += 4
        if value_type == _STRING:
            value, position = self._read_string(position)
            self._value_end = position
            return value
        if value_type == _ARRAY:
            (element_type,) = struct.unpack_from("<I", self._data, position)
            (count,) = struct.unpack_from("<Q", self._data, position + 4)
            position += 12
            values = []
            for _ in range(count):
                if element_type == _STRING:
                    value, position = self._read_string(position)
                else:
                    fmt = _SCALAR_STRUCT[element_type]
                    value = struct.unpack_from(fmt, self._data, position)[0]
                    position += struct.calcsize(fmt)
                values.append(value)
            self._value_end = position
            return values
        fmt = _SCALAR_STRUCT[value_type]
        value = struct.unpack_from(fmt, self._data, position)[0]
        self._value_end = position + struct.calcsize(fmt)
        return value

    # -- tensors ---------------------------------------------------------

    @property
    def tensor_names(self) -> list[str]:
        return list(self._order)

    def info(self, name: str) -> TensorInfo:
        try:
            return self._tensor_infos[name]
        except KeyError as error:
            raise GgufError(f"{self.path}: no tensor named {name!r}") from error

    def __contains__(self, name: object) -> bool:
        return name in self._tensor_infos

    def __iter__(self) -> Iterator[str]:
        return iter(self._order)

    def raw_bytes(self, name: str) -> bytes:
        info = self.info(name)
        start = self.data_offset + info.offset
        return bytes(self._data[start : start + info.byte_size])

    def tensor(self, name: str) -> np.ndarray:
        """Return the tensor as a NumPy array in C-contiguous (PyTorch) order."""
        info = self.info(name)
        if info.type_id not in _TENSOR_DTYPES:
            raise GgufError(f"{self.path}: unsupported ggml type {info.type_id} for {name}")
        raw = np.frombuffer(self.raw_bytes(name), dtype=_TENSOR_DTYPES[info.type_id])
        if info.type_id == GGML_TYPE_BF16:
            return _bf16_to_float32(raw).reshape(info.shape).copy()
        return raw.reshape(info.shape).copy()

    def inventory(self) -> list[tuple[str, tuple[int, ...], str, int]]:
        """(name, shape, type name, byte size) for every tensor, in file order."""
        return [
            (info.name, info.shape, info.type_name, info.byte_size)
            for info in (self._tensor_infos[name] for name in self._order)
        ]

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> "GgufFile":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
