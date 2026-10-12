"""The viewer's placed-target path, over real HTTP.

Two layers:

* a **stub planner** layer that pins the *protocol contract*: the new keywords
  are only sent when a target is actually placed (so an alternative ``plan``
  implementation keeps working, and the response of a targetless request is
  byte-for-byte the document it was before), the query-string and JSON forms of
  the target list both parse, and bad input is a 400 rather than a crash;
* a **real planner** layer that binds the actual :class:`mlx_viewer.Planner`
  (warm MLX pipeline) and drives it with ``curl``-equivalent requests, checking
  the measured hit block and the untouched key set of the default response.

The second layer is skipped when the bundle or the reference styles are absent,
exactly like the rest of the suite.
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mlx_viewer  # noqa: E402
from mlx_viewer import Handler, PlannerStats, Server  # noqa: E402

MLX_BUNDLE = ROOT / "checkpoints/g1-f32-mlx"
STYLES = Path.home() / "dev/motion-bricks-cpp/repo/generated/styles"
HAVE_MODEL = (MLX_BUNDLE / "manifest.json").is_file() and (STYLES / "walk.mbstyle").is_file()
NEEDS_MODEL = pytest.mark.skipif(not HAVE_MODEL, reason="bundle and reference styles required")

#: The keys a targetless ``/api/plan`` response had before this round added any
#: target support.  Placing a target must add to this set, never replace it.
BASE_PLAN_KEYS = {
    "ok", "style", "seed", "move", "facing", "advance", "reset", "fps", "frames",
    "joints", "world_positions_xyz", "targets", "metrics", "timing", "sampling",
}
TARGET_PLAN_KEYS = {"placed_targets", "target_hit"}


class StubPlanner:
    """A planner that records exactly which keywords the handler passed."""

    def __init__(self) -> None:
        self.device = "gpu"
        self.bundle_path = ROOT / "checkpoints/g1-f32-mlx"
        self.styles_path = ROOT / "viewer"
        self.parameters = 183148382
        self.styles = ["idle", "walk", "walk_zombie"]
        self.default_style = "walk"
        self.warm_frames = 44
        self.cold_start_ms = 317.0
        self.skeleton = [{"name": "pelvis_skel", "parent": -1, "position": [0.0, 0.0, 0.0]}]
        self.stats = PlannerStats()
        self.client: dict | None = None
        self.calls: list[dict] = []

    def plan(self, style, seed, move, facing, reset, advance, argmax=False, **extra) -> dict:
        if style not in self.styles:
            raise KeyError(f"unknown style {style!r}")
        if "targets" in extra and not extra["targets"]:
            raise ValueError("empty target list must not be forwarded")
        # The handler forwards the *raw* target payload: parsing and validation
        # are the planner's job (``resolve_world_targets``), not the transport's.
        call = {"style": style, "seed": seed, "move": list(move), "facing": list(facing),
                "reset": reset, "advance": advance, "argmax": argmax}
        call.update(extra)
        self.calls.append(call)
        self.stats.record(26.6)
        payload = {
            "ok": True, "style": style, "seed": seed, "move": list(move),
            "facing": list(facing), "advance": advance, "reset": reset,
            "fps": 30, "frames": 2, "joints": 2,
            "world_positions_xyz": [0.0] * 12,
            "targets": {"frames": 2, "joints": 2, "world_positions_xyz": [0.0] * 12},
            "metrics": {"root_travel_metres": 0.0, "duration_seconds": 2 / 30,
                        "root_first_frame": [0.0, 0.0, 0.0],
                        "root_last_frame": [0.0, 0.0, 0.0]},
            "timing": {"plan_ms": 26.6, "fk_ms": 1.3, "server_total_ms": 27.9},
            "sampling": "argmax" if argmax else "gumbel",
        }
        if "targets" in extra:
            payload["placed_targets"] = {"count": len(extra["targets"]),
                                        "pose_mode": extra["pose_mode"],
                                        "frames": 4, "space": extra["target_space"],
                                        "world_roots": [[0.0, 0.78, 1.0]] * 4,
                                        "world_headings_deg": [0.0] * 4,
                                        "requested": list(extra["targets"])}
            payload["target_hit"] = {"errors": {"root_planar_m": 0.0}, "within_tolerance": True}
        return payload


@pytest.fixture(scope="module")
def stub_server():
    class Bound(Handler):
        planner = StubPlanner()
        quiet = True

    httpd = Server(("127.0.0.1", 0), Bound)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", Bound.planner
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def get(url: str, raw: bool = False):
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read()
            return response.status, (body if raw else json.loads(body))
    except urllib.error.HTTPError as error:
        body = error.read()
        try:
            return error.code, json.loads(body)
        except json.JSONDecodeError:
            return error.code, body


def post(url: str, payload: dict):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


# ---------------------------------------------------------------------------
# protocol contract (stub planner)
# ---------------------------------------------------------------------------
def test_targetless_request_does_not_grow_the_call_or_the_response(stub_server) -> None:
    base, planner = stub_server
    status, body = get(base + "/api/plan?style=walk&seed=3&move=0,1&facing=0,1")
    assert status == 200
    # No new keyword reaches the planner, so an alternative implementation of
    # ``plan`` needs no target support to keep serving this page.
    assert planner.calls[-1] == {"style": "walk", "seed": 3, "move": [0.0, 1.0],
                                 "facing": [0.0, 1.0], "reset": False, "advance": 0,
                                 "argmax": False}
    assert set(body) == BASE_PLAN_KEYS
    assert TARGET_PLAN_KEYS.isdisjoint(body)


def test_query_string_targets_are_forwarded_with_their_options(stub_server) -> None:
    base, planner = stub_server
    query = urllib.parse.urlencode({"style": "walk", "targets": "0,3,0;1,4,90",
                                    "target_space": "offset", "pose_mode": "hold"})
    status, body = get(base + "/api/plan?" + query)
    assert status == 200
    call = planner.calls[-1]
    assert call["targets"] == "0,3,0;1,4,90"
    assert call["target_space"] == "offset" and call["pose_mode"] == "hold"
    assert body["placed_targets"]["space"] == "offset"
    assert body["placed_targets"]["pose_mode"] == "hold"
    assert body["target_hit"]["within_tolerance"] is True
    assert set(body) == BASE_PLAN_KEYS | TARGET_PLAN_KEYS


def test_json_targets_are_forwarded_positionally(stub_server) -> None:
    base, planner = stub_server
    status, body = post(base + "/api/plan", {"style": "walk", "targets": [[0.0, 3.0, 0.0]]})
    assert status == 200
    assert planner.calls[-1]["targets"] == [[0.0, 3.0, 0.0]]
    # The defaults are declared by the server, not implied by the client.
    assert planner.calls[-1]["target_space"] == "world"
    assert planner.calls[-1]["pose_mode"] == "style"
    assert set(body) == BASE_PLAN_KEYS | TARGET_PLAN_KEYS


def test_empty_target_list_is_treated_as_no_targets(stub_server) -> None:
    """An empty list from the page is "nothing placed", not an error."""
    base, planner = stub_server
    status, body = get(base + "/api/plan?style=walk&targets=")
    assert status == 200
    assert "targets" not in planner.calls[-1]
    assert set(body) == BASE_PLAN_KEYS


def test_page_ships_the_placement_controls_and_the_hud_readout(stub_server) -> None:
    base, _ = stub_server
    _, index = get(base + "/", raw=True)
    for needle in (b'id="mlx-target-place"', b'id="mlx-target-replan"',
                   b'id="mlx-target-clear"', b'id="mlx-target-here"',
                   b'id="mlx-target-list"', b'id="mlx-target-hit"',
                   b'id="hud-target"'):
        assert needle in index, needle
    _, app = get(base + "/app.js", raw=True)
    for needle in (b"targets: toTargets", b"target_space: 'world'",
                   b"targetReplan.addEventListener", b"targetPlace.addEventListener",
                   b"rebuildTargetMarkers", b"groundPointAt", b"raycaster",
                   b"addressBar.get('targets')", b"to_targets",
                   b"payload.target_hit", b"live=0"):
        assert needle in app, needle


# ---------------------------------------------------------------------------
# real planner, real HTTP
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def live_server():
    planner = mlx_viewer.Planner(MLX_BUNDLE, STYLES, "cpu")

    class Bound(Handler):
        quiet = True

    Bound.planner = planner
    httpd = Server(("127.0.0.1", 0), Bound)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", planner
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


@NEEDS_MODEL
def test_live_default_response_is_unchanged(live_server) -> None:
    base, _ = live_server
    status, body = get(base + "/api/plan?style=walk&seed=1&move=0,1&facing=0,1&reset=1")
    assert status == 200
    assert set(body) == BASE_PLAN_KEYS
    assert body["frames"] == 44


@NEEDS_MODEL
def test_live_placed_target_is_planned_and_measured(live_server) -> None:
    base, _ = live_server
    query = urllib.parse.urlencode({"style": "walk", "seed": 1, "move": "0,1", "facing": "0,1",
                                    "reset": "1", "targets": "0,3,0",
                                    "target_space": "offset"})
    status, body = get(base + "/api/plan?" + query)
    assert status == 200
    assert set(body) == BASE_PLAN_KEYS | TARGET_PLAN_KEYS
    placed, hit = body["placed_targets"], body["target_hit"]
    assert placed["count"] == 1 and placed["frames"] == 4
    assert placed["space"] == "offset" and placed["pose_mode"] == "style"
    assert hit["window_frames"] == [body["frames"] - 4, body["frames"] - 1]
    assert hit["errors"]["root_planar_m"] <= hit["tolerances"]["root_planar_m"]
    assert hit["within_tolerance"] is True
    # The ghost the page draws is the window that was asked for: the returned
    # target FK must be at the placed world point (x, z) on every sample.
    stride = body["joints"] * 3
    for slot in range(4):
        pelvis = body["targets"]["world_positions_xyz"][slot * stride:slot * stride + 3]
        assert pelvis[0] == pytest.approx(placed["requested"][0]["x"], abs=1e-3)
        assert pelvis[2] == pytest.approx(placed["requested"][0]["z"], abs=1e-3)


@NEEDS_MODEL
def test_live_impossible_target_is_reported_not_refused(live_server) -> None:
    base, _ = live_server
    query = urllib.parse.urlencode({"style": "walk", "seed": 1, "move": "0,1", "facing": "0,1",
                                    "reset": "1", "targets": "0,50,0", "target_space": "offset"})
    status, body = get(base + "/api/plan?" + query)
    assert status == 200, "an over-far target is a *result*, not an error"
    assert body["target_hit"]["within_tolerance"] is False
    assert "root_planar_m" in body["target_hit"]["exceeded"]
    assert all(value == value for value in body["world_positions_xyz"]), "no NaN may escape"


@NEEDS_MODEL
@pytest.mark.parametrize("targets", ["north", "1", "1,2,3,4", [[0.0, 1.0, 0.0]] * 5,
                                     [{"x": "not-a-number", "z": 0}],
                                     [{"x": 1.0, "z": 0.0, "pose": [1.0, 2.0]}],
                                     [{"x": 0.0, "z": 0.0, "heading": "sideways"}]])
def test_live_bad_targets_are_a_400_not_a_crash(live_server, targets) -> None:
    """Validation lives in the planner, and it must fail closed with a reason."""
    base, _ = live_server
    status, body = post(base + "/api/plan", {"style": "walk", "targets": targets,
                                             "reset": True})
    assert status == 400, body
    assert body["ok"] is False
    assert body["error"]


@NEEDS_MODEL
def test_live_target_request_does_not_disturb_the_next_default_plan(live_server) -> None:
    """Placing a target is additive: the default path after it is unchanged."""
    base, _ = live_server
    plain = get(base + "/api/plan?style=walk&seed=1&move=0,1&facing=0,1&reset=1")[1]
    get(base + "/api/plan?" + urllib.parse.urlencode(
        {"style": "walk", "seed": 1, "move": "0,1", "facing": "0,1", "reset": "1",
         "targets": "0,3,0", "target_space": "offset"}))
    again = get(base + "/api/plan?style=walk&seed=1&move=0,1&facing=0,1&reset=1")[1]
    assert again["world_positions_xyz"] == plain["world_positions_xyz"]
    assert TARGET_PLAN_KEYS.isdisjoint(again)
