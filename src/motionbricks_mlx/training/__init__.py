"""Training scaffold for the MLX motion VQ-VAE (B side).

Owned by the B line (see ``docs/brief-claude-training.md``).  Everything here
works against the data contract in ``docs/motion-representation.md`` using
self-generated synthetic shards; no dependency on the A line's
``motion/**`` / ``data/**`` code.

Note: the layout constants in :mod:`.layout` are transcribed from the contract
document because ``motion/representation.py`` (the intended single source of
``FRAME_DIM`` / ``OFFSETS`` / ``CLIP_META_KEYS``) does not exist yet.  Once the
A line lands it, this package should import from there instead.
"""
