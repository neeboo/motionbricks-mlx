#!/usr/bin/env python3
"""Convert a MotionBricks GGUF bundle into an MLX-readable bundle.

    ./.venv/bin/python scripts/mlx_convert_gguf.py \
        --bundle ~/dev/motion-bricks-cpp/repo/generated/g1-f32 \
        --out checkpoints/g1-f32-mlx --list

What it guarantees (see the report in ``docs/status-deepseek.md``):

1. **Inventory parity** -- tensor count and parameter count per component must
   equal the numbers the reference runtime enforces (``src/model.cpp:40-53``).
2. **Byte parity** -- for every F32/I32 tensor the converted array's bytes must
   equal the raw GGUF payload the reference reads.  Mismatches are reported.
3. **Schema alignment** -- every weight name the reference C++ asks for must
   resolve, with the same shape (``models/schema.py``, transcribed from the
   reference with file:line references).
4. **Support-value parity** -- ``motion_mean`` / ``motion_std`` are compared
   against a reference runtime trace when ``--trace`` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motionbricks_mlx.data.gguf import GgufFile, GGML_TYPE_F32, GGML_TYPE_I32
from motionbricks_mlx.models.bundle import EXPECTED_INVENTORY, MLX_BUNDLE_FORMAT, Bundle
from motionbricks_mlx.models.schema import WEIGHT_SPECS, check_bundle

DEFAULT_BUNDLE = Path.home() / "dev/motion-bricks-cpp/repo/generated/g1-f32"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "checkpoints/g1-f32-mlx"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def convert_component(path: Path, destination: Path, list_tensors: bool) -> dict:
    entries = []
    arrays = {}
    with GgufFile(path) as gguf:
        for name in gguf.tensor_names:
            info = gguf.info(name)
            value = gguf.tensor(name)
            raw = gguf.raw_bytes(name)
            contiguous = np.ascontiguousarray(value)
            digest = sha256_bytes(contiguous.tobytes())
            raw_digest = sha256_bytes(raw)
            byte_parity = (info.type_id in (GGML_TYPE_F32, GGML_TYPE_I32)
                           and digest == raw_digest)
            arrays[name] = contiguous
            entries.append({
                "name": name,
                "shape": list(contiguous.shape),
                "dtype": str(contiguous.dtype),
                "ggml_type": info.type_name,
                "bytes": int(contiguous.nbytes),
                "sha256": digest,
                "raw_payload_sha256": raw_digest,
                "byte_identical_to_gguf_payload": bool(byte_parity),
            })
            if list_tensors:
                marker = "==" if byte_parity else "!="
                print(f"  {name:66s} {str(contiguous.shape):20s} "
                      f"{info.type_name:4s} {digest[:16]} {marker}")
    np.savez(destination, **arrays)
    return {
        "path": destination.name,
        "sha256": sha256_file(destination),
        "tensor_count": len(entries),
        "parameter_count": int(sum(item["bytes"] // 4 for item in entries)),
        "tensors": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--list", action="store_true", help="print every tensor name/shape/sha256")
    parser.add_argument("--trace", type=Path, default=None,
                        help="reference transition trace JSON for support-value cross-check")
    parser.add_argument("--skeleton", type=Path, default=None,
                        help="reference plan JSON (Go driver) for skeleton cross-check")
    args = parser.parse_args()

    if not (args.bundle / "manifest.json").is_file():
        print(f"error: {args.bundle} is not a bundle directory", file=sys.stderr)
        return 2
    source_manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    files: dict[str, dict] = {}
    problems: list[str] = []
    for component in ("pose", "root", "vq-decoder", "support"):
        path = args.bundle / source_manifest["components"][component]["path"]
        destination = args.out / f"{component.replace('-', '_')}.npz"
        if args.list:
            print(f"=== {component} ({path.name})")
        entry = convert_component(path, destination, args.list)
        expected = EXPECTED_INVENTORY[component]
        actual = (entry["tensor_count"], entry["parameter_count"])
        if actual != expected:
            problems.append(f"{component}: inventory {actual} != reference {expected}")
        mismatched = [item["name"] for item in entry["tensors"]
                      if not item["byte_identical_to_gguf_payload"]]
        if mismatched:
            problems.append(f"{component}: {len(mismatched)} tensors not byte-identical: "
                            f"{mismatched[:4]}")
        files[component] = entry
        print(f"{component:11s} -> {destination.name}: {entry['tensor_count']} tensors, "
              f"{entry['parameter_count']:,} values, sha256 {entry['sha256'][:16]}")

    manifest = {
        "format": MLX_BUNDLE_FORMAT,
        "source_bundle": str(args.bundle),
        "source_format": source_manifest.get("format"),
        "source_manifest_sha256": sha256_file(args.bundle / "manifest.json"),
        "upstream_revision": source_manifest.get("upstream_revision"),
        "skeleton": source_manifest.get("skeleton"),
        "joint_count": source_manifest.get("joint_count"),
        "files": files,
        "metadata": {},
    }

    # joint names live in the support GGUF metadata (src/model.cpp:205-215).
    with GgufFile(args.bundle / source_manifest["components"]["support"]["path"]) as gguf:
        manifest["metadata"]["joint_names"] = gguf.metadata["motionbricks.joint_names"]
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # -- cross-checks against the reference runtime ----------------------
    bundle = Bundle(args.out)
    schema_problems = check_bundle(bundle)
    problems += schema_problems
    used_fallback = bundle.resolutions()
    fallback = sorted(item.requested for item in used_fallback if item.via_fallback)
    print(f"\nschema: {len(WEIGHT_SPECS['root']) + len(WEIGHT_SPECS['pose']) + len(WEIGHT_SPECS['vq-decoder']) + len(WEIGHT_SPECS['support'])} "
          f"reference weight names checked, {len(schema_problems)} problem(s)")
    print(f"name-compaction fallback used by {len(fallback)} name(s): {fallback}")

    if args.trace is not None:
        trace = json.loads(args.trace.read_text(encoding="utf-8"))
        for key, tensor in (("stats.mean", "motion_mean"), ("stats.std", "motion_std")):
            reference = np.asarray(trace["tensors"][key], dtype=np.float32)
            mine = np.asarray(bundle.support[tensor], dtype=np.float32).reshape(-1)
            if reference.shape != mine.shape:
                problems.append(f"{tensor}: trace {reference.shape} vs bundle {mine.shape}")
                continue
            delta = float(np.max(np.abs(reference - mine))) if reference.size else 0.0
            print(f"support:{tensor:12s} vs trace {key:10s} max|diff| = {delta:.3e}")
            if delta != 0.0:
                problems.append(f"{tensor}: differs from reference trace by {delta}")

    if args.skeleton is not None:
        document = json.loads(args.skeleton.read_text(encoding="utf-8"))
        step = document["steps"][0]
        skeleton = step["skeleton"]
        parents = np.asarray(bundle.support["joint_parents"]).reshape(-1)
        reference_parents = np.asarray([joint["parent"] for joint in skeleton], dtype=np.int32)
        positions = np.asarray([joint["position"] for joint in skeleton], dtype=np.float32)
        neutral = np.asarray(bundle.support["neutral_joints"], dtype=np.float32)
        parent_delta = int(np.max(np.abs(parents - reference_parents)))
        position_delta = float(np.max(np.abs(neutral - positions)))
        print(f"support:joint_parents vs reference skeleton max|diff| = {parent_delta}")
        print(f"support:neutral_joints vs reference skeleton max|diff| = {position_delta:.3e}")
        if parent_delta != 0:
            problems.append("joint_parents differ from the reference skeleton")
        if position_delta > 1e-6:
            problems.append(f"neutral_joints differ from the reference skeleton by {position_delta}")

    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nconversion verified: inventory, bytes, schema and reference values agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
