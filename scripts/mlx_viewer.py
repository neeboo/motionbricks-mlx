#!/usr/bin/env python3
"""Local viewer server: a browser page that plays motion planned by *our* MLX port.

What this is
------------
A single-file, stdlib-only HTTP server that

  * serves the static page in ``viewer/`` (its own HTML/CSS/JS, the reference
    demo's three.js vendor bundle, its skeleton-rig module and its stylesheet),
  * keeps one warm :class:`motionbricks_mlx.models.transition.MlxPipeline` and
    one :class:`~motionbricks_mlx.data.agent.Agent` in memory, and
  * answers ``/api/plan`` by *planning live with MLX* and returning the clip's
    world-space joint positions, so the page never replays a recording.

This is deliberately not the reference protocol.  The reference Go demo drives
motion over a WebSocket (``/api/stream``, see ``demo/main.go`` and
``demo/stream.go`` in ``~/dev/motion-bricks-cpp/repo``) and has retired its
HTTP ``/api/plan``.  We replace the transport with plain request/response JSON:
one HTTP request is one plan, which is trivial to drive and to measure from
``curl`` as well as from the page.

Protocol
--------
``GET  /``                     the page.
``GET  /<asset>``              ``style.css``, ``app.js``, ``rig.js``,
                               ``vendor/three.module.min.js``, ... from ``viewer/``.
``GET  /api/health``           warm-up facts: device, parameter count, cold start.
``GET  /api/styles``           the real ``*.mbstyle`` inventory plus the skeleton
                               (joint names / parents / rest positions).
``GET  /api/plan``             query form of the plan call (handy for ``curl``).
``POST /api/plan``             JSON form of the plan call (what the page uses).
``POST /api/client-metrics``   the page reports its own page-load and first-pose
                               timings; ``/api/stats`` echoes the last report.
``GET  /api/stats``            rolling timing counters observed by the server.

``/api/plan`` parameters (all optional except ``style``)::

    style    one of the names in /api/styles           (default: the inventory default)
    seed     sampling seed                             (default 1)
    move     "x,z" travel direction in the world       (default 0,1)
    facing   "x,z" facing direction in the world       (default 0,1)
    reset    1 to reseed the agent from the style      (default 0)
    advance  frames of the current clip already shown  (default 0)
    argmax   1 for diagnostic argmax sampling          (default 0)
    targets  user-placed target keyframes, "x,z,deg;x,z,deg" (up to 4)
    target_space  "world" (default) or "offset" from the clip start
    pose_mode     "style" (default) or "hold" the clip's starting pose
    switch   1 to switch *into* ``style`` from the current clip, keeping the
             context (default 0).  Without it a style change reseeds the agent
             from the new style, which is the control this round measures against.
    sequence  "B,A" -- plan a joined A->B->A clip in one response (style is A)
    seams     the seam frame for each entry of ``sequence`` (default: advance)

``targets`` is the *external* half of the transition boundary: the model's target
window is boundary frames 4..7 (``src/root.cpp:366``), normally manufactured from
a style clip by ``build_constraints``, and ``targets`` replaces those 4 samples
outright.  Each entry is a world point plus a world facing in degrees; a single
target is held across all 4 samples, which is the shape that tracks best
(measured; see ``src/motionbricks_mlx/data/targets.py``).  When ``targets`` is
present the response gains two keys -- ``placed_targets`` (the window that was
asked for) and ``target_hit`` (measured root / heading / joint error against the
declared tolerances) -- and nothing else about the response changes.

``advance`` is the continuity seam and is worth stating precisely: the agent's
planning context is the *last four frames of its current motion*, and
``Agent.advance(n)`` moves that context to frame ``n``.  The returned clip's
frame 0 is therefore the world pose the page has already displayed at frame
``advance``, which is exactly how the reference client re-anchors its playhead
(``replacementPlayhead = playhead - advance``, ``demo/web/app.js``).

``switch`` is the same mechanism with the *style* changed as well: nothing is
reset, so the target window (frames 4..7) comes from the new style while the start
window (frames 0..3, and therefore its ``local_root`` velocities) and the
inherited ``apply_context_blend`` still come from the clip on screen.  The response
gains a ``switch`` block with the measured seam (position gap, and |dv| / |da| at
the seam against the in-segment p95 of the same finite difference).  Sent with
``switch=1`` on every request that follows a style change, so a live replan keeps
starting from the new style instead of falling back to a reseed.  ``sequence``
returns a whole A->B->A clip in one response with every seam measured; it is what
the page's "Play A -> B -> A" control and the recorded GIF play.

Everything here is read-only with respect to the repository: no checkpoint, no
``out/`` document and no reference file is written.  The server binds
``127.0.0.1`` only, and it is deliberately **single-threaded**: MLX's GPU stream
is thread-local, so a ``ThreadingHTTPServer`` worker aborts the process with
``There is no Stream(gpu, 0) in current thread``.  There is exactly one stateful
agent anyway, so serialising planning is the correct shape.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from motionbricks_mlx.data.agent import Agent, Command          # noqa: E402
from motionbricks_mlx.data.style import load_style              # noqa: E402
from motionbricks_mlx.data.switch import (LEG_KINDS, Switcher,  # noqa: E402
                                          seam_metrics)
from motionbricks_mlx.data.targets import (measure_hit, plan_with_targets,  # noqa: E402
                                           resolve_world_targets)
from motionbricks_mlx.export.plan_json import FPS               # noqa: E402
from motionbricks_mlx.models.bundle import Bundle               # noqa: E402
from motionbricks_mlx.models.transition import MlxPipeline      # noqa: E402

DEFAULT_BUNDLE = ROOT / "checkpoints/g1-f32-mlx"
DEFAULT_STYLES = Path.home() / "dev/motion-bricks-cpp/repo/generated/styles"
VIEWER_ROOT = ROOT / "viewer"
DEFAULT_VIEWER_STYLE = "walk"


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------
def world_positions(roots: np.ndarray, rotations_xyzw: np.ndarray,
                    parents: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """Batched equivalent of ``plan_json.world_positions_xyz``.

    The reference function walks frames and joints in two Python loops and does
    exactly the arithmetic below per joint; this version keeps the joint loop
    (the tree order requires it) and batches the frame axis. Every output element
    is the same fixed float64 expression on the same inputs, so the two agree
    bit for bit -- ``tests/test_viewer_fk.py`` asserts that on real plans.
    Batching matters: it is 15.0 ms per clip in the loop form and under 1 ms here.
    """
    frames, joints = roots.shape[0], rotations_xyzw.shape[1]
    positions = np.zeros((frames, joints, 3), dtype=np.float64)
    world = np.zeros((frames, joints, 4), dtype=np.float64)
    rotation = np.asarray(rotations_xyzw, dtype=np.float64)
    root = np.asarray(roots, dtype=np.float64)
    for joint in range(joints):
        parent = int(parents[joint])
        if parent < 0:
            positions[:, joint] = root
            world[:, joint] = rotation[:, joint]
            continue
        q = world[:, parent]
        bx, by, bz, bw = (q[:, 0], q[:, 1], q[:, 2], q[:, 3])
        rel = (rest[joint] - rest[parent]).astype(np.float64)
        vx, vy, vz, vw = rel[0], rel[1], rel[2], 0.0
        # q * (v, 0)
        tx = bw * vx + bx * vw + by * vz - bz * vy
        ty = bw * vy - bx * vz + by * vw + bz * vx
        tz = bw * vz + bx * vy - by * vx + bz * vw
        tw = bw * vw - bx * vx - by * vy - bz * vz
        # (q * (v,0)) * conj(q), vector part
        cx, cy, cz, cw = -bx, -by, -bz, bw
        rx = tw * cx + tx * cw + ty * cz - tz * cy
        ry = tw * cy - tx * cz + ty * cw + tz * cx
        rz = tw * cz + tx * cy - ty * cx + tz * cw
        positions[:, joint] = positions[:, parent] + np.stack((rx, ry, rz), axis=1)

        r = rotation[:, joint]
        ax, ay, az, aw = (q[:, 0], q[:, 1], q[:, 2], q[:, 3])
        dx, dy, dz, dw = (r[:, 0], r[:, 1], r[:, 2], r[:, 3])
        world[:, joint] = np.stack((
            aw * dx + ax * dw + ay * dz - az * dy,
            aw * dy - ax * dz + ay * dw + az * dx,
            aw * dz + ax * dy - ay * dx + az * dw,
            aw * dw - ax * dx - ay * dy - az * dz,
        ), axis=1)
    return positions


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------
@dataclass
class PlannerStats:
    plans: int = 0
    plan_ms: list = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def record(self, milliseconds: float) -> None:
        self.plans += 1
        self.plan_ms.append(round(milliseconds, 4))
        if len(self.plan_ms) > 256:
            del self.plan_ms[:-256]

    def summary(self) -> dict:
        values = sorted(self.plan_ms)
        def quantile(q: float) -> float:
            if not values:
                return 0.0
            return values[min(len(values) - 1, int(q * len(values)))]
        return {
            "plans": self.plans,
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "plan_ms_last": values[-1] if values else 0.0,
            "plan_ms_p50": round(quantile(0.50), 3),
            "plan_ms_p90": round(quantile(0.90), 3),
            "plan_ms_max": round(max(values), 3) if values else 0.0,
            "steps_per_second_p50": round(1000.0 / quantile(0.50), 2) if values else 0.0,
            "samples": len(values),
        }


class Planner:
    """One warm MLX pipeline plus one stateful agent, guarded by a lock."""

    def __init__(self, bundle_path: Path, styles_path: Path, device: str):
        started = time.perf_counter()
        self.bundle = Bundle(bundle_path, device=device)
        self.pipeline = MlxPipeline(self.bundle)
        self.device = device
        self.bundle_path = bundle_path
        self.styles_path = styles_path
        self.parents = np.asarray(self.bundle.joint_parents)
        self.rest = np.asarray(self.bundle.neutral_joints)
        self.joint_names = list(self.bundle.joint_names)
        self.skeleton = [
            {"name": self.joint_names[joint], "parent": int(self.parents[joint]),
             "position": [float(v) for v in self.rest[joint]]}
            for joint in range(self.parents.shape[0])
        ]
        self.parameters = sum(
            self.bundle.manifest["files"][name]["parameter_count"]
            for name in ("pose", "root", "vq-decoder"))
        self.styles = sorted(path.stem for path in styles_path.glob("*.mbstyle"))
        if not self.styles:
            raise RuntimeError(f"no .mbstyle files under {styles_path}")
        self.default_style = (DEFAULT_VIEWER_STYLE if DEFAULT_VIEWER_STYLE in self.styles
                              else self.styles[0])
        self._style_cache: dict[str, object] = {}
        self._agent: Agent | None = None
        self._agent_style: str | None = None
        self._switcher: Switcher | None = None
        self.lock = threading.Lock()
        self.stats = PlannerStats()
        self.client: dict | None = None
        self.cold_start_ms = (time.perf_counter() - started) * 1000.0
        # Warm the graph so the first browser request is not the slow one.
        warm = self.plan(style=self.default_style, seed=1, move=(0.0, 1.0),
                         facing=(0.0, 1.0), reset=True, advance=0)
        self.cold_start_ms = (time.perf_counter() - started) * 1000.0
        self.stats.plan_ms.clear()
        self.stats.plans = 0
        self.warm_frames = warm["frames"]

    def style(self, name: str):
        if name not in self._style_cache:
            if name not in self.styles:
                raise KeyError(f"unknown style {name!r}; have {', '.join(self.styles)}")
            self._style_cache[name] = load_style(self.styles_path / f"{name}.mbstyle")
        return self._style_cache[name]

    def switch_command(self, style: str, seed: int, move, facing, argmax: bool) -> Command:
        return Command(style=self.style(style), movement=(float(move[0]), 0.0, float(move[1])),
                       facing=(float(facing[0]), 0.0, float(facing[1])),
                       seed=int(seed), sampling_argmax=bool(argmax))

    def plan(self, style: str, seed: int, move, facing, reset: bool, advance: int,
             argmax: bool = False, targets=None, target_space: str = "world",
             pose_mode: str = "style", switch: bool = False,
             sequence=None, seams=None) -> dict:
        """One plan; ``switch``/``sequence`` are the additive A->B seam paths.

        With neither flag this method is the round-8/9 code path unchanged: the
        same branch, the same call, the same response keys.
        """
        loaded = self.style(style)
        if sequence:
            return self.plan_sequence(style, sequence, seed, move, facing, argmax, advance,
                                      seams)
        with self.lock:
            previous = self._agent.current_motion if self._agent is not None else None
            previous_style = self._agent_style
            reseeded = reset or self._agent is None or (self._agent_style != style and not switch)
            if reseeded:
                self._agent = Agent(self.pipeline, self.bundle.neutral_joints, self.parents)
                self._agent.reset(loaded)
                advance = 0
            elif advance > 0 and self._agent.current_motion is not None:
                self._agent.advance(int(advance))
            # The agent's *active* style is the one its current motion is in.  A
            # switch updates it without a reseed, which is what makes the next
            # request in the same style continue from B instead of teleporting
            # back to B's own origin.
            self._agent_style = style
            command = Command(style=loaded, movement=(float(move[0]), 0.0, float(move[1])),
                              facing=(float(facing[0]), 0.0, float(facing[1])),
                              seed=int(seed), sampling_argmax=bool(argmax))
            started = time.perf_counter()
            # Placed targets take the additive path; with none, this is exactly
            # the accepted Agent.plan call, unchanged.
            placed = resolve_world_targets(self._agent, targets, target_space) if targets \
                else []
            if placed:
                motion, _, window = plan_with_targets(self._agent, command, loaded, placed,
                                                     pose_mode)
                hit = measure_hit(motion, window, self.parents, self.rest)
                placed_payload = {
                    "count": len(window.placed),
                    "pose_mode": window.pose_mode,
                    "frames": window.frames,
                    "space": target_space,
                    "world_roots": [[float(v) for v in row] for row in window.world_roots],
                    "world_headings_deg": [float(v) for v in window.world_headings_deg],
                    "requested": [{"x": float(t.x), "z": float(t.z),
                                   "heading_deg": float(t.heading_deg)}
                                  for t in window.placed],
                }
            else:
                motion, _ = self._agent.plan(command, loaded)
                hit = None
                placed_payload = None
            plan_ms = (time.perf_counter() - started) * 1000.0
            self.stats.record(plan_ms)
            switch_payload = None
            if switch:
                # The seam is between the clip that was on screen and this one.
                # ``advance`` is the frame of the old clip the context was taken
                # from, which is where ``restore_world`` puts the new clip's
                # frame 0 -- so the two clips meet exactly there.
                switch_payload = self._live_switch_payload(
                    previous, previous_style, style, advance if not reseeded else 0,
                    motion, conditioned=bool(previous is not None and not reseeded))
            roots = np.asarray(motion.roots, dtype=np.float64)
            rotations = np.asarray(motion.rotations_xyzw, dtype=np.float64)
            target_roots = np.asarray(motion.target_roots, dtype=np.float64)
            target_rotations = np.asarray(motion.target_local_rotations_xyzw, dtype=np.float64)

        payload = self._motion_payload(style, seed, move, facing, advance, reset, roots,
                                       rotations, target_roots, target_rotations, plan_ms,
                                       "argmax" if argmax else "gumbel")
        if placed_payload is not None:
            # Additive: only a plan that placed targets carries these keys, so
            # every existing consumer sees the same response as before.
            payload["placed_targets"] = placed_payload
            payload["target_hit"] = hit
        if switch_payload is not None:
            payload["switch"] = switch_payload
        return payload

    # -- switch helpers --------------------------------------------------
    def _live_switch_payload(self, previous, previous_style, style: str, seam_at: int,
                             motion, conditioned: bool) -> dict:
        """The seam report for a live A->B switch (or for a same-style replan)."""
        report = None
        if previous is not None and conditioned:
            index = min(int(seam_at), int(previous.frames) - 1)
            if index >= 3 and motion.frames >= 4:
                report = seam_metrics(previous, index, motion, self.parents, self.rest,
                                      label=f"{previous_style}->{style}",
                                      world=world_positions)
        seams = []
        if report is not None:
            # Same nested shape the sequence path returns, so one page readout
            # handles both and the seam cannot be described two ways.
            seams.append({"joined_frame": 0, "from_style": previous_style,
                          "to_style": style, "metrics": report})
        ratios = [max(report["ratio"].values())] if report else []
        return {
            "mode": "live",
            "conditioned": bool(conditioned),
            "from_style": previous_style,
            "to_style": style,
            "chain": [previous_style, style] if previous_style else [style],
            "seam_at": int(seam_at) if conditioned else 0,
            "legs": [],
            "seams": seams,
            "joined_frames": int(motion.frames),
            "criterion": None if report is None else report["criterion"],
            "position_criterion": None if report is None else report["position_criterion"],
            "worst_seam_ratio": max(ratios) if ratios else None,
            "all_seams_within_p95": bool(report["all_within_p95"]) if report else None,
            "note": None if conditioned else
                    "no previous clip to condition on (cold start or reseed)",
        }

    def plan_sequence(self, style: str, sequence, seed: int, move, facing, argmax: bool,
                      advance: int, seams=None) -> dict:
        """A -> B -> ... -> A as **one** joined clip, every seam measured.

        The legs are chained on the live agent (``Agent.advance`` then
        ``Agent.plan`` -- no reseed between legs), so the clip the page plays is
        the same object the seam numbers describe.
        """
        names = [style] + [str(name) for name in sequence]
        if len(names) < 2:
            raise ValueError("a sequence needs at least one style after 'style'")
        loaded = [self.style(name) for name in names]
        if seams:
            seam_frames = [int(value) for value in seams]
            if len(seam_frames) != len(names) - 1:
                raise ValueError(f"{len(names)} styles need {len(seam_frames) - 1} seams")
        else:
            seam_frames = [int(advance) or 20] * (len(names) - 1)
        with self.lock:
            if self._switcher is None:
                self._switcher = Switcher(self.pipeline, self.bundle,
                                          {name: loaded[index] for index, name in enumerate(names)},
                                          seed=int(seed), move=move, facing=facing,
                                          world=world_positions)
            else:
                self._switcher.styles.update({name: loaded[index]
                                              for index, name in enumerate(names)})
                self._switcher.seed = int(seed)
                self._switcher.move = tuple(move)
                self._switcher.facing = tuple(facing)
            agent = Agent(self.pipeline, self.bundle.neutral_joints, self.parents)
            agent.reset(loaded[0])
            started = time.perf_counter()
            plan = self._switcher.sequence_with(agent, names, seam_frames,
                                                kind=LEG_KINDS[0], seed=int(seed))
            plan_ms = (time.perf_counter() - started) * 1000.0
            self.stats.record(plan_ms)
            self._agent = agent
            self._agent_style = names[-1]
            roots = np.asarray(plan.roots, dtype=np.float64)
            rotations = np.asarray(plan.rotations_xyzw, dtype=np.float64)
            target_roots = np.asarray(plan.target_roots, dtype=np.float64)
            target_rotations = np.asarray(plan.target_local_rotations_xyzw, dtype=np.float64)
            block = plan.document()
        payload = self._motion_payload(style, seed, move, facing, seam_frames[0], reset=True,
                                       roots=roots, rotations=rotations,
                                       target_roots=target_roots,
                                       target_rotations=target_rotations, plan_ms=plan_ms,
                                       sampling="argmax" if argmax else "gumbel")
        block.update({"mode": "sequence", "conditioned": True, "from_style": names[0],
                      "to_style": names[-1], "chain": names, "seam_at": seam_frames[0],
                      "note": None})
        payload["switch"] = block
        return payload

    def _motion_payload(self, style: str, seed: int, move, facing, advance: int, reset: bool,
                        roots, rotations, target_roots, target_rotations, plan_ms: float,
                        sampling: str) -> dict:
        """The response document shared by every planning path (shape unchanged)."""
        fk_started = time.perf_counter()
        positions = world_positions(roots, rotations, self.parents, self.rest)
        target_positions = world_positions(target_roots, target_rotations, self.parents, self.rest)
        fk_ms = (time.perf_counter() - fk_started) * 1000.0
        frames = int(roots.shape[0])
        travel = float(np.linalg.norm(np.diff(roots, axis=0), axis=1).sum()) if frames > 1 else 0.0
        return {
            "ok": True,
            "style": style,
            "seed": int(seed),
            "move": [float(move[0]), float(move[1])],
            "facing": [float(facing[0]), float(facing[1])],
            "advance": int(advance),
            "reset": bool(reset),
            "fps": FPS,
            "frames": frames,
            "joints": int(self.parents.shape[0]),
            "world_positions_xyz": positions.reshape(-1).tolist(),
            "targets": {
                "frames": int(target_roots.shape[0]),
                "joints": int(self.parents.shape[0]),
                "world_positions_xyz": target_positions.reshape(-1).tolist(),
            },
            "metrics": {
                "root_travel_metres": travel,
                "duration_seconds": frames / FPS,
                "root_first_frame": [float(v) for v in roots[0]],
                "root_last_frame": [float(v) for v in roots[-1]],
            },
            "timing": {"plan_ms": round(plan_ms, 4), "fk_ms": round(fk_ms, 4),
                       "server_total_ms": round(plan_ms + fk_ms, 4)},
            "sampling": sampling,
        }


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def parse_pair(raw, default):
    """``"x,z"`` (query string) or ``[x, z]`` (JSON body) -> ``[float, float]``."""
    if raw is None or raw == "":
        return list(default)
    if isinstance(raw, (list, tuple)):
        if len(raw) != 2:
            raise ValueError(f"expected two numbers, got {raw!r}")
        return [float(raw[0]), float(raw[1])]
    parts = str(raw).split(",")
    if len(parts) != 2:
        raise ValueError(f"expected 'x,z', got {raw!r}")
    return [float(parts[0]), float(parts[1])]


def parse_sequence(raw):
    """``"walk_zombie,walk"`` (query string) or ``["walk_zombie", "walk"]`` -> list.

    Used for both ``sequence`` (style names) and ``seams`` (frame indices); an
    absent/empty value is ``None`` so the caller can tell "not asked" from "asked
    for nothing".
    """
    if raw is None or raw == "":
        return None
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = [piece.strip() for piece in str(raw).split(",")]
    items = [item for item in items if item != ""]
    if not items:
        return None
    if all(isinstance(item, (int, float)) or str(item).lstrip("-").isdigit() for item in items):
        return [int(item) for item in items]
    return [str(item) for item in items]


class Handler(BaseHTTPRequestHandler):
    server_version = "mlx-viewer/1.0"
    # One connection at a time, closed after every response.  MLX's GPU stream
    # is thread-local: the warm pipeline's arrays belong to the thread that
    # loaded them, and running ``Agent.plan`` from a ThreadingHTTPServer worker
    # aborts the process with "There is no Stream(gpu, 0) in current thread".
    # A local single-user viewer wants serialised planning anyway (there is one
    # stateful agent), so this is the correct shape rather than a workaround.
    protocol_version = "HTTP/1.1"

    planner: Planner = None       # injected on the server instance
    quiet: bool = False

    # -- helpers ---------------------------------------------------------
    def log_message(self, fmt, *args):        # noqa: A003 - stdlib signature
        if not self.quiet:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _finish(self, status: int, content_type: str, body: bytes) -> None:
        self.close_connection = True
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.send_header("cache-control", "no-store")
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def send_payload(self, payload: dict, status: int = 200) -> None:
        self._finish(status, "application/json; charset=utf-8",
                     json.dumps(payload, separators=(",", ":")).encode("utf-8"))

    def send_file(self, path: Path) -> None:
        if not path.is_file():
            self.send_payload({"ok": False, "error": "not found"}, status=404)
            return
        body = path.read_bytes()
        guessed, _ = mimetypes.guess_type(path.name)
        content_type = guessed or "application/octet-stream"
        if path.suffix == ".js":
            content_type = "text/javascript; charset=utf-8"
        elif path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif path.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        self._finish(200, content_type, body)

    def static_path(self, route: str) -> Path | None:
        relative = "index.html" if route in ("", "/") else route.lstrip("/")
        candidate = (VIEWER_ROOT / relative).resolve()
        if VIEWER_ROOT.resolve() not in candidate.parents and candidate != VIEWER_ROOT.resolve():
            return None
        return candidate

    # -- routing ---------------------------------------------------------
    def do_GET(self):                          # noqa: N802 - stdlib signature
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route in ("", "/"):
                self.send_file(VIEWER_ROOT / "index.html")
            elif route == "/api/health":
                self.send_payload(self.health())
            elif route == "/api/styles":
                self.send_payload(self.styles())
            elif route == "/api/stats":
                self.send_payload({"ok": True, "client": self.planner.client,
                                   **self.planner.stats.summary()})
            elif route == "/api/plan":
                query = {key: value[0] for key, value in parse_qs(parsed.query).items()}
                self.send_payload(self.plan(query))
            else:
                target = self.static_path(route)
                if target is None:
                    self.send_payload({"ok": False, "error": "forbidden path"}, status=403)
                else:
                    self.send_file(target)
        except KeyError as error:
            self.send_payload({"ok": False, "error": str(error)}, status=404)
        except (ValueError, TypeError) as error:
            self.send_payload({"ok": False, "error": str(error)}, status=400)
        except BrokenPipeError:
            pass

    def do_POST(self):                         # noqa: N802 - stdlib signature
        parsed = urlparse(self.path)
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as error:
            self.send_payload({"ok": False, "error": f"bad json: {error}"}, status=400)
            return
        try:
            if parsed.path == "/api/plan":
                self.send_payload(self.plan(body))
            elif parsed.path == "/api/client-metrics":
                self.planner.client = {
                    key: body.get(key) for key in
                    ("first_pose_ms", "ready_ms", "nav_ms", "device", "clips",
                     "style", "frames", "fps", "playing", "measured_at")
                }
                self.planner.client["user_agent"] = self.headers.get("user-agent", "")
                self.send_payload({"ok": True, "stored": True})
            else:
                self.send_payload({"ok": False, "error": "not found"}, status=404)
        except KeyError as error:
            self.send_payload({"ok": False, "error": str(error)}, status=404)
        except (ValueError, TypeError) as error:
            self.send_payload({"ok": False, "error": str(error)}, status=400)
        except BrokenPipeError:
            pass

    # -- endpoints -------------------------------------------------------
    def health(self) -> dict:
        planner = self.planner
        return {
            "ok": True,
            "library": "motionbricks_mlx",
            "device": f"mlx-{planner.device}",
            "model_bundle": str(planner.bundle_path),
            "styles_path": str(planner.styles_path),
            "learned_parameters": planner.parameters,
            "style_count": len(planner.styles),
            "skeleton_joints": len(planner.skeleton),
            "cold_start_ms": round(planner.cold_start_ms, 3),
            "warm_clip_frames": planner.warm_frames,
            "bind": f"{self.server.server_address[0]}:{self.server.server_address[1]}",
            **planner.stats.summary(),
        }

    def styles(self) -> dict:
        planner = self.planner
        return {
            "ok": True,
            "styles": planner.styles,
            "default": planner.default_style,
            "fps": FPS,
            "skeleton": planner.skeleton,
        }

    def plan(self, request: dict) -> dict:
        planner = self.planner
        style = str(request.get("style") or planner.default_style)
        seed = int(request.get("seed", 1))
        move = parse_pair(request.get("move"), (0.0, 1.0))
        facing = parse_pair(request.get("facing"), (0.0, 1.0))
        reset = str(request.get("reset", "0")).lower() in ("1", "true", "yes", "on")
        advance = int(request.get("advance", 0) or 0)
        argmax = str(request.get("argmax", "0")).lower() in ("1", "true", "yes", "on")
        targets = request.get("targets")
        target_space = str(request.get("target_space") or "world").lower()
        pose_mode = str(request.get("pose_mode") or "style").lower()
        switch = str(request.get("switch", "0")).lower() in ("1", "true", "yes", "on")
        sequence = parse_sequence(request.get("sequence"))
        seams = parse_sequence(request.get("seams"))
        if sequence is not None:
            return planner.plan(style=style, seed=seed, move=move, facing=facing,
                                reset=True, advance=advance, argmax=argmax,
                                sequence=sequence, seams=seams)
        if switch:
            # The A->B path: keep the context, select B for frames 4..7.  Sent by
            # the page for every plan that follows a style switch, so a live
            # replan keeps starting from B instead of falling back to a reseed.
            return planner.plan(style=style, seed=seed, move=move, facing=facing,
                                reset=reset, advance=advance, argmax=argmax,
                                switch=True, targets=targets,
                                target_space=target_space, pose_mode=pose_mode)
        if not targets:
            # Untouched call shape for every request that does not place a
            # target: the response is the same document it was before, and an
            # alternative ``plan`` implementation needs no new keyword arguments.
            return planner.plan(style=style, seed=seed, move=move, facing=facing,
                                reset=reset, advance=advance, argmax=argmax)
        return planner.plan(style=style, seed=seed, move=move, facing=facing,
                            reset=reset, advance=advance, argmax=argmax,
                            targets=targets, target_space=target_space,
                            pose_mode=pose_mode)


class Server(HTTPServer):
    """Single-threaded on purpose: see the comment on ``Handler.protocol_version``."""
    allow_reuse_address = True


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address; only 127.0.0.1 is accepted (default)")
    parser.add_argument("--port", type=int, default=8848, help="TCP port (default 8848)")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--styles", type=Path, default=DEFAULT_STYLES)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--quiet", action="store_true", help="do not log every request")
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost"):
        parser.error("this viewer is local-only: --host must be 127.0.0.1")

    if not (args.bundle / "manifest.json").is_file():
        print(f"error: {args.bundle} has no manifest.json", file=sys.stderr)
        return 2

    print(f"loading {args.bundle} on mlx-{args.device} ...", file=sys.stderr, flush=True)
    planner = Planner(args.bundle, args.styles, args.device)
    print(f"ready: cold start {planner.cold_start_ms:.1f} ms, "
          f"{len(planner.styles)} styles, warm clip {planner.warm_frames} frames",
          file=sys.stderr, flush=True)

    Handler.planner = planner
    Handler.quiet = args.quiet
    server = Server((args.host, args.port), Handler)
    print(f"viewer: http://127.0.0.1:{args.port}/  (pid {os.getpid()})",
          file=sys.stderr, flush=True)
    pid_directory = ROOT / "artifacts/viewer-run"
    if pid_directory.is_dir():
        (pid_directory / "server.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
