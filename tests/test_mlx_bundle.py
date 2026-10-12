"""Bundle, GGUF reader and schema-alignment tests.

These run without the reference checkout for everything except the byte-parity
check against the raw GGUF payloads, which is skipped when the upstream bundle
is not present.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motionbricks_mlx.data.gguf import GgufFile  # noqa: E402
from motionbricks_mlx.models.bundle import Bundle, Weights  # noqa: E402
from motionbricks_mlx.models.schema import WEIGHT_SPECS, check_bundle  # noqa: E402

MLX_BUNDLE = ROOT / "checkpoints/g1-f32-mlx"
UPSTREAM_BUNDLE = Path.home() / "dev/motion-bricks-cpp/repo/generated/g1-f32"

pytestmark = pytest.mark.skipif(not (MLX_BUNDLE / "manifest.json").is_file(),
                                reason="run scripts/mlx_convert_gguf.py first")


@lru_cache(maxsize=1)
def bundle() -> Bundle:
    return Bundle(MLX_BUNDLE)


def test_inventory_and_parameters_match_reference():
    """Tensor/parameter counts are the ones src/model.cpp:40-53 enforces."""
    expected = {"pose": (209, 136_588_272), "root": (150, 34_122_833),
                "vq-decoder": (51, 12_437_277), "support": (4, 972)}
    for component, (tensors, parameters) in expected.items():
        entry = bundle().manifest["files"][component]
        assert entry["tensor_count"] == tensors
        assert entry["parameter_count"] == parameters


def test_every_reference_weight_name_resolves_with_expected_shape():
    """The transcribed reference schema must fully resolve (models/schema.py)."""
    assert check_bundle(bundle()) == []


def test_documented_name_compaction_fallback_is_the_only_one_used():
    """Only the three >=64-character attention names use the runtime fallback."""
    used = {item.requested for item in bundle().resolutions() if item.via_fallback}
    assert used == {f"_root_token_transformer_model.layers.{layer}.self_attn.out_proj.weight"
                    for layer in range(3)}


def test_fallback_rule_matches_reference_lookup():
    """src/neural_runtime.cpp:272-276: >=64 chars and contains 'self_attn.'."""
    compacted = "_root_token_transformer_model.layers.0.attn.out_proj.weight"
    weights = Weights("root", {compacted: np.zeros(1, dtype=np.float32)})
    long = "_root_token_transformer_model.layers.0.self_attn.out_proj.weight"
    assert len(long) >= 64
    assert weights.resolve(long) == compacted
    assert weights.resolutions[long].via_fallback
    # 63 characters: below the reference's boundary, so no fallback is attempted.
    short = "_root_token_transformer_model.layers.0.self_attn.in_proj_weight"
    assert len(short) < 64
    with pytest.raises(Exception):
        weights.resolve(short)


def test_missing_weight_raises():
    weights = Weights("root", {})
    with pytest.raises(Exception):
        weights["not_a_weight"]


def test_gguf_payload_bytes_are_identical_to_the_converted_arrays():
    """Byte parity: the arrays we run are the bytes the reference reads."""
    if not (UPSTREAM_BUNDLE / "manifest.json").is_file():
        pytest.skip("reference bundle not present")
    import json
    manifest = json.loads((UPSTREAM_BUNDLE / "manifest.json").read_text())
    for component, entry in bundle().manifest["files"].items():
        mismatched = [item["name"] for item in entry["tensors"]
                      if not item["byte_identical_to_gguf_payload"]]
        assert mismatched == []
        assert entry["tensor_count"] == manifest["components"][component]["tensor_count"]


def test_support_values_match_reference_runtime_trace():
    """motion_mean/motion_std are copied straight out of the runtime's trace."""
    trace = ROOT / "out/ref/trace/transition-0.json"
    if not trace.is_file():
        pytest.skip("reference trace not captured (run the reference driver first)")
    import json
    tensors = json.loads(trace.read_text())["tensors"]
    for key, name in (("stats.mean", "motion_mean"), ("stats.std", "motion_std")):
        reference = np.asarray(tensors[key], dtype=np.float32)
        mine = np.asarray(bundle().support[name], dtype=np.float32).reshape(-1)
        assert reference.shape == mine.shape
        assert np.array_equal(reference, mine)


def test_gguf_reader_rejects_non_gguf(tmp_path: Path):
    path = tmp_path / "bogus.gguf"
    path.write_bytes(b"not a gguf file at all")
    with pytest.raises(Exception):
        GgufFile(path)
