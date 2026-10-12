"""Contract-derived layout: derivation, round-trip, and rejection tests."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motionbricks_mlx.training import layout


def test_block_order_matches_contract_table():
    # docs/motion-representation.md v0 table order — a reorder is a schema bump
    assert layout.BLOCK_ORDER == (
        "root_position", "root_yaw", "root_velocity", "joint_rotations",
        "joint_velocities", "contacts", "phase",
    )
    assert layout.OPTIONAL_BLOCKS == ("joint_velocities",)
    assert layout.CLIP_META_KEYS == (
        "style", "target", "start_root", "end_root", "fps", "skeleton_id", "provenance",
    )
    assert layout.DEFAULT_FPS == 30


@pytest.mark.parametrize("joints", [1, 4, 34])
@pytest.mark.parametrize("with_jv", [True, False])
def test_frame_dim_derivation(joints, with_jv):
    table = layout.offsets(joints, with_jv)
    blocks = layout.blocks(joints, with_jv)
    assert table[0] == 0
    assert len(table) == len(blocks) + 1
    # D is derived from the offsets table (contract wording)
    assert layout.frame_dim(joints, with_jv) == int(table[-1]) == blocks[-1].stop
    for i, block in enumerate(blocks):
        assert block.slice == slice(int(table[i]), int(table[i + 1]))
        assert block.width == table[i + 1] - table[i]
    expected_d = 11 + (9 if with_jv else 6) * joints  # 3+1+3+2+2 fixed, 6D/3D per joint
    assert layout.frame_dim(joints, with_jv) == expected_d


@pytest.mark.parametrize("joints", [1, 4, 34])
@pytest.mark.parametrize("with_jv", [True, False])
def test_offsets_roundtrip_and_inference(joints, with_jv):
    table = layout.offsets(joints, with_jv)
    assert [b.name for b in layout.blocks_from_offsets(table)] == [
        b.name for b in layout.blocks(joints, with_jv)
    ]
    assert layout.infer_joints(table) == (joints, with_jv)


def test_block_slice_selects_expected_columns():
    table = layout.offsets(4, True)
    assert layout.block_slice(table, "root_position") == slice(0, 3)
    assert layout.block_slice(table, "joint_rotations") == slice(7, 31)
    assert layout.block_slice(table, "phase") == slice(45, 47)
    with pytest.raises(layout.LayoutError):
        layout.block_slice(table, "nonexistent")


@pytest.mark.parametrize(
    "table",
    [
        [1, 3, 4, 7, 31, 43, 45, 47],  # leading entry must be 0
        [0, 3, 4, 7, 31, 43, 45, 40],  # not increasing
        [0, 3, 4],  # wrong length
        [0, 3, 4, 8, 32, 44, 46, 48],  # joint width 8 not divisible by 6
    ],
)
def test_invalid_offset_tables_rejected(table):
    with pytest.raises(layout.LayoutError):
        layout.blocks_from_offsets(np.asarray(table, dtype=np.int64))


def test_meta_validation_requires_all_contract_keys():
    layout.validate_meta({k: None for k in layout.CLIP_META_KEYS})
    incomplete = {k: None for k in layout.CLIP_META_KEYS if k != "fps"}
    with pytest.raises(layout.LayoutError, match="fps"):
        layout.validate_meta(incomplete)
