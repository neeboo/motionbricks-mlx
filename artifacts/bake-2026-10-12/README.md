# `artifacts/bake-2026-10-12` — what is here, and what is deliberately not committed

Round 7's bake products.  The run, the manifest, the equivalence tables, the gate
output and the honest boundaries are in
[`docs/BATCH-BAKE-RUN.md`](../../docs/BATCH-BAKE-RUN.md); this file only says
which files are tracked and which are regenerable.

**Tracked** (evidence; a few hundred kB in total):

| path | what |
|---|---|
| `diverse/`, `single-style/`, `idle-style/` | `manifest.json` (the twelve fields per `B`), `manifest-check.json`, `compare-B*.json` (per-scenario batch vs single: tokens / uniforms / duration identity, worst deviation, both arms' float32 digests), `metrics-*.json` (C-line metric verdicts on the products) |
| `single-style-r9/manifest.json` | the same single-style `B = 16` bake with `--repeats 9`, for the timing-schedule comparison |
| `logs/` | the verbatim stdout of every command, including the `load[...]` lines |
| `gifs/`, `gifs/pairs/` | the rendered animations and the side-by-side composites |
| `render/pairs*.json` | the label → style → render-document map |

**Not tracked** (regenerable, 27.7 MiB of plan documents plus 1.5 MiB of render
documents, and that is why):

| path | what |
|---|---|
| `*/plans/**` | the per-scenario plan documents (`motion.roots`, `motion.rotations`, `world_positions_xyz`, `targets`). 128 documents, 170–298 kB each because the document shape is the reference driver's flat float lists. `scripts/mlx_bake_library.py batch\|single` rebuilds them byte for byte. |
| `render/render-*.json` | the two-step documents fed to the reference renderer, assembled from the above by `pairdoc`. |
| `runs-batch-B*.json`, `runs-single-B16.json` | the raw per-scenario token / uniform / selected dumps the two arms were paired from: 2.1 MiB across nine files, one number per line. `compare-B*.json` is the distilled form of exactly these, per scenario, with both arms' digests; the raw dumps stay on disk because re-running `batch`/`single` reproduces them exactly. |

Regenerate everything with the commands in section 5 of
[`docs/BATCH-BAKE-RUN.md`](../../docs/BATCH-BAKE-RUN.md).
