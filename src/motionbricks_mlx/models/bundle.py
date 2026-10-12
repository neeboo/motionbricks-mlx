"""Weight bundles for the MLX port.

The reference runtime resolves a weight name in two steps
(``src/neural_runtime.cpp:265-278``)::

    exact match; if that fails and len(name) >= 64 and name contains
    "self_attn.", replace the first occurrence with "attn." and retry.

That fallback exists because ``scripts/convert_to_gguf.py`` shortens names at
the ``GGML_MAX_NAME`` boundary.  :class:`Weights` reproduces the rule exactly
and records which lookups needed it, so the port cannot silently read a
different tensor than the reference does.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..data.gguf import GgufFile

MLX_BUNDLE_FORMAT = "motionbricks-mlx-bundle-v1"
COMPONENTS = ("pose", "root", "vq-decoder", "support")

# Reference bundle inventory (generated/g1-f32/manifest.json).
EXPECTED_INVENTORY = {
    "pose": (209, 136_588_272),
    "root": (150, 34_122_833),
    "vq-decoder": (51, 12_437_277),
    "support": (4, 972),
}


class BundleError(RuntimeError):
    pass


@dataclass
class Resolution:
    """How one C++-side weight name resolved against the GGUF tensor list."""

    requested: str
    resolved: str
    via_fallback: bool


@dataclass
class Weights:
    """Name -> array mapping with the reference's compaction fallback."""

    component: str
    tensors: dict[str, mx.array]
    resolutions: dict[str, Resolution] = field(default_factory=dict)

    @staticmethod
    def _fallback(name: str) -> str | None:
        if len(name.encode("utf-8")) < 64:
            return None
        needle = "self_attn."
        position = name.find(needle)
        if position < 0:
            return None
        return name[:position] + "attn." + name[position + len(needle):]

    def resolve(self, name: str) -> str:
        if name in self.resolutions:
            return self.resolutions[name].resolved
        resolved, fallback = name, False
        if name not in self.tensors:
            candidate = self._fallback(name)
            if candidate is not None and candidate in self.tensors:
                resolved, fallback = candidate, True
        if resolved not in self.tensors:
            raise BundleError(f"{self.component}: missing weight {name!r}")
        self.resolutions[name] = Resolution(name, resolved, fallback)
        return resolved

    def __getitem__(self, name: str) -> mx.array:
        return self.tensors[self.resolve(name)]

    def get(self, name: str) -> mx.array | None:
        try:
            return self[name]
        except BundleError:
            return None

    def __contains__(self, name: object) -> bool:
        try:
            self.resolve(str(name))
            return True
        except BundleError:
            return False

    @property
    def fallback_names(self) -> list[str]:
        return sorted(key for key, value in self.resolutions.items() if value.via_fallback)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


class Bundle:
    """The four inference components plus the support tensors.

    ``source`` may be either the upstream ``g1-f32`` GGUF directory or a
    converted MLX directory produced by ``scripts/mlx_convert_gguf.py``.
    """

    def __init__(self, source: str | Path, device: str | None = None,
                 verify: bool = False):
        self.source = Path(source)
        self.device = device
        if device is not None:
            mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
        manifest_path = self.source / "manifest.json"
        if not manifest_path.is_file():
            raise BundleError(f"{self.source}: no manifest.json")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fmt = self.manifest.get("format")
        self.converted = fmt == MLX_BUNDLE_FORMAT
        if not self.converted and fmt != "motionbricks-gguf-bundle-v1":
            raise BundleError(f"{self.source}: unsupported bundle format {fmt!r}")
        if verify:
            self.verify_hashes()
        self._cache: dict[str, Weights] = {}

    # -- verification ----------------------------------------------------

    def verify_hashes(self) -> None:
        key = "files" if self.converted else "components"
        for name, entry in self.manifest[key].items():
            path = self.source / entry["path"]
            actual = _sha256(path)
            if actual != entry["sha256"]:
                raise BundleError(f"{path}: sha256 {actual} != manifest {entry['sha256']}")

    # -- components ------------------------------------------------------

    def weights(self, component: str) -> Weights:
        if component in self._cache:
            return self._cache[component]
        tensors: dict[str, mx.array] = {}
        if self.converted:
            path = self.source / self.manifest["files"][component]["path"]
            with np.load(path, allow_pickle=False) as archive:
                for name in archive.files:
                    tensors[name] = mx.array(archive[name])
        else:
            path = self.source / self.manifest["components"][component]["path"]
            with GgufFile(path) as gguf:
                for name in gguf.tensor_names:
                    tensors[name] = mx.array(gguf.tensor(name))
        mx.eval(list(tensors.values()))
        weights = Weights(component, tensors)
        self._cache[component] = weights
        return weights

    @property
    def root(self) -> Weights:
        return self.weights("root")

    @property
    def pose(self) -> Weights:
        return self.weights("pose")

    @property
    def decoder(self) -> Weights:
        return self.weights("vq-decoder")

    @property
    def support(self) -> Weights:
        return self.weights("support")

    # -- support tensors -------------------------------------------------

    @property
    def joint_parents(self) -> np.ndarray:
        return np.asarray(self.support["joint_parents"])

    @property
    def neutral_joints(self) -> np.ndarray:
        return np.asarray(self.support["neutral_joints"])

    @property
    def motion_mean(self) -> np.ndarray:
        return np.asarray(self.support["motion_mean"])

    @property
    def motion_std(self) -> np.ndarray:
        return np.asarray(self.support["motion_std"])

    @property
    def joint_names(self) -> list[str]:
        raw = self.manifest.get("metadata", {}).get("joint_names")
        if raw is None:
            raise BundleError("bundle manifest has no joint_names")
        return raw.split(",")

    @property
    def vq_codebook(self) -> mx.array:
        return self.decoder["quantizer.vq._codebook.embed"]

    def resolutions(self) -> list[Resolution]:
        result: list[Resolution] = []
        for component in COMPONENTS:
            result.extend(self.weights(component).resolutions.values())
        return result

    def release(self, component: str | None = None) -> None:
        if component is None:
            self._cache.clear()
        else:
            self._cache.pop(component, None)
