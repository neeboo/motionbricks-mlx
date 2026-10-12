"""Contract-derived frame layout (``docs/motion-representation.md``, v0 draft).

**Provenance.** Every constant in this module is a transcription of the data
contract; nothing is invented.  The contract's single-source module
``motion/representation.py`` (``FRAME_DIM`` / ``OFFSETS`` / ``CLIP_META_KEYS``)
does not exist yet, so the B-side training package carries this transcription.
When the A line lands its module, imports here should switch to it and this
file shrinks to a compatibility check (same values, else fail loudly).

**Reading of the contract that this module encodes** (both flagged in
``docs/status-claude.md`` for the A line to confirm):

1. ``offsets`` is the block-offset table for the per-frame feature vector, in
   the contract's table order.  It is an int array of length
   ``n_blocks + 1`` with ``offsets[0] == 0``; block *i* occupies
   ``[offsets[i], offsets[i+1])`` and ``D == offsets[-1]``.  This makes the
   contract sentence "D is derived from the offsets table" literal.
2. ``meta.json`` is a sibling file of ``clip.npz`` (not an array inside it),
   containing exactly the clip-level keys below.
3. Block order and widths are fixed by the contract table.  The only optional
   block in v0 is ``joint_velocities``; presence is recoverable from the
   offsets length, and the joint count from the joint block width.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CONTRACT_DOC = "docs/motion-representation.md"
CONTRACT_VERSION = "v0-draft"

#: Default capture rate (contract: "Default 30 fps").
DEFAULT_FPS = 30

#: Per-frame feature blocks, in the contract table's order.  Do not reorder.
BLOCK_ORDER: tuple[str, ...] = (
    "root_position",
    "root_yaw",
    "root_velocity",
    "joint_rotations",
    "joint_velocities",
    "contacts",
    "phase",
)

#: Fixed widths for non-joint blocks.
_FIXED_WIDTHS: dict[str, int] = {
    "root_position": 3,  # world x/y/z (training convention: Z-up, right-handed)
    "root_yaw": 1,  # yaw only in v0 (full root orientation is out of scope)
    "root_velocity": 3,
    "contacts": 2,  # per-foot boolean
    "phase": 2,  # sin/cos of cycle
}

#: Blocks a shard may omit; every other block in ``BLOCK_ORDER`` is mandatory.
OPTIONAL_BLOCKS: tuple[str, ...] = ("joint_velocities",)

#: Clip-level metadata keys (contract "Clip-level features").
CLIP_META_KEYS: tuple[str, ...] = (
    "style",
    "target",
    "start_root",
    "end_root",
    "fps",
    "skeleton_id",
    "provenance",
)


class LayoutError(ValueError):
    """A shard's layout disagrees with the contract-derived rules."""


@dataclass(frozen=True)
class Block:
    """One contiguous slice of the per-frame vector."""

    name: str
    start: int
    stop: int

    @property
    def width(self) -> int:
        return self.stop - self.start

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


def _present_blocks(with_joint_velocities: bool) -> tuple[str, ...]:
    return tuple(
        name for name in BLOCK_ORDER
        if with_joint_velocities or name != "joint_velocities"
    )


def block_width(name: str, joints: int) -> int:
    """Width of one block for a skeleton with ``joints`` joints."""
    if joints < 1:
        raise LayoutError(f"joint count must be >= 1, got {joints}")
    if name == "joint_rotations":
        return 6 * joints  # 6D rotation per joint
    if name == "joint_velocities":
        return 3 * joints  # 3D angular velocity per joint
    if name in _FIXED_WIDTHS:
        return _FIXED_WIDTHS[name]
    raise LayoutError(f"unknown block {name!r}")


def blocks(joints: int, with_joint_velocities: bool = True) -> tuple[Block, ...]:
    """The block list for a v0 frame vector, with start/stop columns."""
    out: list[Block] = []
    start = 0
    for name in _present_blocks(with_joint_velocities):
        width = block_width(name, joints)
        out.append(Block(name, start, start + width))
        start += width
    return tuple(out)


def offsets(joints: int, with_joint_velocities: bool = True) -> np.ndarray:
    """The offsets table as stored in a shard: ``[0, w0, w0+w1, ..., D]``."""
    bs = blocks(joints, with_joint_velocities)
    table = [0] + [b.stop for b in bs]
    return np.asarray(table, dtype=np.int64)


def frame_dim(joints: int, with_joint_velocities: bool = True) -> int:
    """``D`` derived from the layout (== ``offsets(...)[:-1]`` last entry)."""
    return int(offsets(joints, with_joint_velocities)[-1])


def _require_contract_shapes(table: np.ndarray) -> None:
    if table.ndim != 1:
        raise LayoutError(f"offsets must be 1-D, got shape {table.shape}")
    # Stored tables carry the leading 0: length is n_blocks + 1.
    if table.size not in (len(BLOCK_ORDER) + 1, len(BLOCK_ORDER)):
        raise LayoutError(
            "offsets length must be "
            f"{len(BLOCK_ORDER) + 1} (with joint_velocities) or {len(BLOCK_ORDER)}"
            f" (without); got {table.size}"
        )
    if table[0] != 0:
        raise LayoutError(f"offsets[0] must be 0, got {table[0]}")
    if not np.all(np.diff(table) > 0):
        raise LayoutError(f"offsets must be strictly increasing, got {table.tolist()}")


def blocks_from_offsets(table: np.ndarray) -> tuple[Block, ...]:
    """Invert :func:`offsets`: recover the block list from a stored table.

    Works because the v0 block order is fixed and only ``joint_velocities``
    is optional: the table length says whether it is present, and the first
    joint block's width gives the joint count.
    """
    table = np.asarray(table)
    _require_contract_shapes(table)
    with_jv = table.size == len(BLOCK_ORDER) + 1
    joints = int(table[4] - table[3]) // 6  # root_position+root_yaw+root_velocity = 7
    if (table[4] - table[3]) % 6 != 0 or joints < 1:
        raise LayoutError(
            f"joint_rotations width {int(table[4] - table[3])} is not 6*J for J >= 1"
        )
    bs = blocks(joints, with_jv)
    expected = offsets(joints, with_jv)
    if not np.array_equal(table.astype(np.int64), expected):
        raise LayoutError(
            "offsets do not match any v0 layout: got "
            f"{table.tolist()}, nearest v0 layout is {expected.tolist()}"
        )
    return bs


def infer_joints(table: np.ndarray) -> tuple[int, bool]:
    """``(joints, with_joint_velocities)`` for a stored offsets table."""
    for b in blocks_from_offsets(table):
        if b.name == "joint_rotations":
            return b.width // 6, len(table) == len(BLOCK_ORDER) + 1
    raise LayoutError("joint_rotations block missing from offsets table")


def block_slice(table: np.ndarray, name: str) -> slice:
    """Column slice of one block inside ``frames[..., D]`` for a stored table."""
    for b in blocks_from_offsets(table):
        if b.name == name:
            return b.slice
    raise LayoutError(f"block {name!r} is not present in this offsets table")


def validate_meta(meta: dict) -> None:
    """All clip-level contract keys present (values' shapes are clip-specific)."""
    missing = [k for k in CLIP_META_KEYS if k not in meta]
    if missing:
        raise LayoutError(f"meta.json is missing contract keys: {missing}")
