"""The viewer's switch path, over real HTTP.

Two layers, the same shape as ``tests/test_viewer_targets.py``:

* a **stub planner** layer that pins the protocol contract: the new keywords reach
  the planner only when a switch/sequence was actually asked for, an ordinary
  request keeps the exact call shape and the exact response key set it had before,
  and bad input is a 400 rather than a crash;
* a **real planner** layer that binds the actual :class:`mlx_viewer.Planner` and
  drives it over HTTP, checking the measured seam block and the page's controls.

The second layer is skipped when the bundle or the reference styles are absent.
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

#: The keys a plain ``/api/plan`` response had before this round added any switch
#: support.  Switching must add to this set, never replace it.
BASE_PLAN_KEYS = {
    "ok", "style", "seed", "move", "facing", "advance", "reset", "fps", "frames",
    "joints", "world_positions_xyz", "targets", "metrics", "timing", "sampling",
}
SWITCH_PLAN_KEYS = {"switch"}


class StubPlanner:
    """A planner that records exactly which keywords the handler passed."""

    def __init__(self) -> None:
        self.device = "gpu"
        self.bundle_path = ROOT / "checkpoints/g1-f32-mlx"
        self.styles_path = ROOT / "viewer"
        self.parameters = 183148382
        self.styles = ["idle", "walk", "walk_zombie", "slow_walk"]
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
        if extra.get("switch") or extra.get("sequence"):
            payload["switch"] = {"mode": "sequence" if extra.get("sequence") else "live",
                                 "conditioned": True, "chain": [style],
                                 "seams": [], "all_seams_within_p95": True}
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


@pytest.fixture(scope="module")
def live_server():
    if not HAVE_MODEL:
        pytest.skip("bundle and reference styles required")
    # ``cpu`` on purpose, exactly like tests/test_viewer_targets.py: MLX's GPU
    # stream is thread-local and the server plans from a worker thread, so a
    # gpu-backed planner built on the test thread aborts the process.  The switch
    # logic is backend-independent; the protocol is what this layer pins.
    planner = mlx_viewer.Planner(MLX_BUNDLE, STYLES, "cpu")

    class Bound(Handler):
        planner = None
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


def get(url: str, raw: bool = False):
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
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
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


# ---------------------------------------------------------------------------
# protocol contract (stub planner)
# ---------------------------------------------------------------------------
def test_an_ordinary_request_does_not_grow_the_call_or_the_response(stub_server) -> None:
    base, planner = stub_server
    status, body = get(base + "/api/plan?style=walk&seed=3&move=0,1&facing=0,1&advance=12")
    assert status == 200
    # No new keyword reaches the planner, so an alternative ``plan`` implementation
    # keeps serving this page unchanged.
    assert planner.calls[-1] == {"style": "walk", "seed": 3, "move": [0.0, 1.0],
                                 "facing": [0.0, 1.0], "reset": False, "advance": 12,
                                 "argmax": False}
    assert set(body) == BASE_PLAN_KEYS
    assert SWITCH_PLAN_KEYS.isdisjoint(body)


def test_the_switch_flag_is_forwarded_as_a_keyword(stub_server) -> None:
    base, planner = stub_server
    status, body = get(base + "/api/plan?style=walk_zombie&advance=20&switch=1")
    assert status == 200
    call = planner.calls[-1]
    assert call["switch"] is True and call["style"] == "walk_zombie"
    assert call["advance"] == 20 and call["reset"] is False
    assert "sequence" not in call
    assert set(body) == BASE_PLAN_KEYS | SWITCH_PLAN_KEYS
    assert body["switch"]["mode"] == "live"


def test_switch_off_is_the_untouched_call(stub_server) -> None:
    base, planner = stub_server
    get(base + "/api/plan?style=walk_zombie&advance=20&switch=0")
    assert "switch" not in planner.calls[-1]


def test_a_sequence_is_forwarded_with_its_seam_frames(stub_server) -> None:
    base, planner = stub_server
    status, body = post(base + "/api/plan",
                        {"style": "walk", "sequence": ["walk_zombie", "walk"],
                         "seams": [20, 20]})
    assert status == 200
    call = planner.calls[-1]
    assert call["sequence"] == ["walk_zombie", "walk"]
    assert call["seams"] == [20, 20]
    assert "switch" not in call          # the sequence supersedes the single switch
    assert body["switch"]["mode"] == "sequence"


def test_a_query_string_sequence_is_parsed_the_same_way(stub_server) -> None:
    base, planner = stub_server
    query = urllib.parse.urlencode({"style": "walk", "sequence": "walk_zombie,walk",
                                    "seams": "12,18"})
    status, _ = get(base + "/api/plan?" + query)
    assert status == 200
    assert planner.calls[-1]["sequence"] == ["walk_zombie", "walk"]
    assert planner.calls[-1]["seams"] == [12, 18]


def test_an_empty_sequence_is_treated_as_no_sequence(stub_server) -> None:
    base, planner = stub_server
    get(base + "/api/plan?style=walk&sequence=&seams=")
    assert "sequence" not in planner.calls[-1]
    assert "switch" not in planner.calls[-1]


def test_a_switch_with_an_unknown_style_is_a_404_not_a_crash(stub_server) -> None:
    base, _ = stub_server
    status, body = get(base + "/api/plan?style=nope&switch=1")
    assert status == 404
    assert body["ok"] is False


def test_the_page_ships_the_switch_controls_and_the_hud_readout() -> None:
    html = (ROOT / "viewer/index.html").read_text(encoding="utf-8")
    for needle in ("switch-select", "switch-go", "switch-play", "switch-clear",
                   "switch-status", "hud-seamgap", "hud-seamjump", "stat-seam",
                   "seam-swatch", "mlx-target-keep"):
        assert needle in html, f"index.html is missing {needle}"
    script = (ROOT / "viewer/app.js").read_text(encoding="utf-8")
    for needle in ("switchTo", "switchMode", "applySwitch", "rebuildSeamMarkers",
                   "state.seams", "requestPlan(0, {sequence:", "targetsActive"):
        assert needle in script, f"app.js is missing {needle}"
    # The switch must be sticky: the flag is read from long-lived state at request
    # time, not passed once at the call site.
    assert "switch: sequence ? undefined : state.switchMode" in script


# ---------------------------------------------------------------------------
# the real planner, over HTTP
# ---------------------------------------------------------------------------
@NEEDS_MODEL
def test_live_default_response_is_unchanged(live_server) -> None:
    base, _ = live_server
    status, body = get(base + "/api/plan?style=walk&seed=1&move=0,1&facing=0,1&reset=1")
    assert status == 200
    assert set(body) == BASE_PLAN_KEYS


@NEEDS_MODEL
def test_live_switch_measures_the_seam_and_keeps_conditioning(live_server) -> None:
    base, _ = live_server
    get(base + "/api/plan?style=walk&seed=1&move=0,1&facing=0,1&reset=1")
    status, body = get(base + "/api/plan?style=walk_zombie&seed=1&move=0,1&facing=0,1"
                              "&advance=20&switch=1")
    assert status == 200
    block = body["switch"]
    assert block["mode"] == "live" and block["conditioned"] is True
    assert block["chain"] == ["walk", "walk_zombie"]
    seam = block["seams"][0]
    assert seam["joined_frame"] == 0
    assert seam["metrics"]["index_in"] == 20
    # The whole point: the handoff is on the pose, and the outgoing velocity is
    # inside the segment's own p95.
    assert seam["metrics"]["root_position_m"] < 1.5e-3
    assert seam["metrics"]["ratio"]["root_velocity"] <= 1.0
    assert seam["metrics"]["criterion"].startswith("seam |dq| <= ")
    # ... and it does not teleport: the new clip starts where the old one was.
    assert body["metrics"]["root_first_frame"][2] > 0.05

    # A follow-up replan in the same style (no flag needed: the agent's context is
    # already B's) still starts from B and does not reseed back to the origin.
    status, again = get(base + "/api/plan?style=walk_zombie&seed=1&move=0,1&facing=0,1"
                                "&advance=20&switch=1")
    assert again["metrics"]["root_first_frame"][2] > 0.05
    assert again["switch"]["from_style"] == "walk_zombie"


@NEEDS_MODEL
def test_live_switch_off_is_the_old_reseed_and_teleports(live_server) -> None:
    """The control the report is measured against, on the same server."""
    base, _ = live_server
    get(base + "/api/plan?style=walk&seed=1&move=0,1&facing=0,1&reset=1")
    _, walked = get(base + "/api/plan?style=walk&seed=1&move=0,1&facing=0,1&advance=20")
    far = walked["metrics"]["root_first_frame"][2]
    assert far > 0.05
    status, body = get(base + "/api/plan?style=walk_zombie&seed=1&move=0,1&facing=0,1"
                              "&advance=20")          # no switch flag
    assert status == 200
    assert "switch" not in body
    # The reseed put B's clip back at its own origin: this is the teleport the
    # switch removes, and it is still what a plain style change does.
    assert abs(body["metrics"]["root_first_frame"][2]) < 1.0e-6


@NEEDS_MODEL
def test_live_sequence_is_one_clip_with_every_seam(live_server) -> None:
    base, _ = live_server
    status, body = post(base + "/api/plan",
                        {"style": "walk", "seed": 1, "move": [0, 1], "facing": [0, 1],
                         "advance": 20, "sequence": ["walk_zombie", "walk"],
                         "seams": [20, 20]})
    assert status == 200
    block = body["switch"]
    assert block["chain"] == ["walk", "walk_zombie", "walk"]
    assert block["mode"] == "sequence"
    assert body["frames"] == 20 + 20 + 44
    assert [leg["style"] for leg in block["legs"]] == ["walk", "walk_zombie", "walk"]
    assert [leg["start"] for leg in block["legs"]] == [0, 20, 40]
    assert [seam["joined_frame"] for seam in block["seams"]] == [20, 40]
    for seam in block["seams"]:
        assert seam["metrics"]["root_position_m"] < 1.5e-3
        assert seam["metrics"]["ratio"]["root_velocity"] <= 1.0
    # The clip really is the joined motion, not three separate ones: it is longer
    # than any single leg and the position array has one entry per frame.
    joints = body["joints"]
    assert len(body["world_positions_xyz"]) == body["frames"] * joints * 3


@NEEDS_MODEL
def test_live_sequence_leaves_the_agent_in_a_usable_state(live_server) -> None:
    base, planner = live_server
    post(base + "/api/plan", {"style": "walk", "seed": 1, "move": [0, 1], "facing": [0, 1],
                              "advance": 20, "sequence": ["walk_zombie", "walk"],
                              "seams": [20, 20]})
    assert planner._agent_style == "walk"
    status, body = get(base + "/api/plan?style=walk&seed=1&move=0,1&facing=0,1&advance=20")
    assert status == 200
    assert set(body) == BASE_PLAN_KEYS
    assert body["frames"] == 44
