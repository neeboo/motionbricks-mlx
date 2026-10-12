"""The viewer server's routing, protocol and path safety.

These tests do not load the model.  They bind the real handler to a stub planner
on an ephemeral 127.0.0.1 port, so what is exercised is exactly the code path a
browser or a ``curl`` call takes: request parsing, the JSON envelope, status
codes, and the static-file guard.  The one thing they cannot cover is MLX
planning itself; that is what the ``mlx_viewer`` end-to-end probe in
``docs/VIEWER.md`` does against a live server.
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mlx_viewer  # noqa: E402
from mlx_viewer import Handler, PlannerStats, Server  # noqa: E402


class StubPlanner:
    """The attribute surface ``Handler`` reads, with no MLX behind it."""

    def __init__(self) -> None:
        self.device = "gpu"
        self.bundle_path = ROOT / "checkpoints/g1-f32-mlx"
        self.styles_path = ROOT / "viewer"
        self.parameters = 183148382
        self.styles = ["idle", "walk", "walk_zombie", "stealth_walk"]
        self.default_style = "walk"
        self.warm_frames = 44
        self.cold_start_ms = 317.0
        self.skeleton = [
            {"name": "pelvis_skel", "parent": -1, "position": [0.0, 0.0, 0.0]},
            {"name": "left_hip_pitch_skel", "parent": 0, "position": [0.06, -0.1, 0.0]},
        ]
        self.stats = PlannerStats()
        self.client: dict | None = None
        self.calls: list[dict] = []

    def plan(self, style, seed, move, facing, reset, advance, argmax=False) -> dict:
        if style not in self.styles:
            raise KeyError(f"unknown style {style!r}")
        if advance < 0:
            raise ValueError("advance must be >= 0")
        self.calls.append({"style": style, "seed": seed, "move": list(move),
                           "facing": list(facing), "reset": reset, "advance": advance,
                           "argmax": argmax})
        self.stats.record(26.6)
        return {
            "ok": True, "style": style, "seed": seed, "move": list(move),
            "facing": list(facing), "advance": advance, "reset": reset,
            "fps": 30, "frames": 2, "joints": 2,
            "world_positions_xyz": [0.0, 0.0, 0.0, 0.1, 0.2, 0.3,
                                    0.0, 0.0, 0.0, 0.2, 0.3, 0.4],
            "targets": {"frames": 2, "joints": 2,
                        "world_positions_xyz": [0.0, 0.0, 0.0, 0.5, 0.5, 0.5,
                                                0.0, 0.0, 0.0, 0.6, 0.6, 0.6]},
            "metrics": {"root_travel_metres": 0.123, "duration_seconds": 2 / 30,
                        "root_first_frame": [0.0, 0.0, 0.0],
                        "root_last_frame": [0.0, 0.0, 0.2]},
            "timing": {"plan_ms": 26.6, "fk_ms": 1.3, "server_total_ms": 27.9},
            "sampling": "argmax" if argmax else "gumbel",
        }


@pytest.fixture(scope="module")
def server():
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
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_index_is_served(server) -> None:
    base, _ = server
    status, body = get(base + "/", raw=True)
    assert status == 200
    assert b"<title>MotionBricks MLX" in body
    assert b"app.js" in body


@pytest.mark.parametrize("asset,needle", [
    ("style.css", b"#mlx-status"),
    ("app.js", b"/api/plan"),
    ("rig.js", b"SkeletonRig"),
    ("vendor/three.module.min.js", b"THREE"),
    ("vendor/THREE-LICENSE.txt", b"MIT License"),
])
def test_static_assets_are_served(server, asset: str, needle: bytes) -> None:
    base, _ = server
    status, body = get(f"{base}/{asset}", raw=True)
    assert status == 200
    assert needle in body


def test_health_reports_the_planner(server) -> None:
    base, _ = server
    status, body = get(base + "/api/health")
    assert status == 200
    assert body["device"] == "mlx-gpu"
    assert body["learned_parameters"] == 183148382
    assert body["style_count"] == 4


def test_styles_lists_the_real_inventory_and_skeleton(server) -> None:
    base, _ = server
    status, body = get(base + "/api/styles")
    assert status == 200
    assert body["styles"][:2] == ["idle", "walk"]
    assert body["default"] == "walk"
    assert body["skeleton"][2 - 1]["name"] == "left_hip_pitch_skel"


def test_get_plan_drives_a_live_plan(server) -> None:
    base, planner = server
    status, body = get(base + "/api/plan?style=walk_zombie&seed=9&move=1,0&facing=1,0&advance=3")
    assert status == 200
    assert body["style"] == "walk_zombie"
    assert body["frames"] == 2
    assert body["targets"]["frames"] == 2
    assert len(body["world_positions_xyz"]) == body["frames"] * body["joints"] * 3
    assert planner.calls[-1] == {"style": "walk_zombie", "seed": 9, "move": [1.0, 0.0],
                                 "facing": [1.0, 0.0], "reset": False, "advance": 3,
                                 "argmax": False}


def test_post_plan_is_what_the_page_uses(server) -> None:
    base, planner = server
    status, body = post(base + "/api/plan", {"style": "idle", "seed": 2, "move": [0, 0],
                                             "facing": [0, 1], "reset": True, "advance": 0})
    assert status == 200
    assert body["reset"] is True
    assert planner.calls[-1]["reset"] is True


def test_unknown_style_is_a_404_not_a_crash(server) -> None:
    base, _ = server
    status, body = get(base + "/api/plan?style=does_not_exist")
    assert status == 404
    assert body["ok"] is False
    assert "unknown style" in body["error"]


def test_bad_direction_is_a_400(server) -> None:
    base, _ = server
    status, body = get(base + "/api/plan?style=walk&move=north")
    assert status == 400
    assert body["ok"] is False


def test_negative_advance_is_rejected(server) -> None:
    base, _ = server
    status, body = get(base + "/api/plan?style=walk&advance=-4")
    assert status == 400
    assert body["ok"] is False


def test_stats_endpoint_counts_plans(server) -> None:
    base, _ = server
    get(base + "/api/plan?style=walk")
    get(base + "/api/plan?style=idle")
    status, body = get(base + "/api/stats")
    assert status == 200
    assert body["plans"] >= 2
    assert body["plan_ms_p50"] > 0
    assert "client" in body


def test_client_metrics_are_stored_and_echoed(server) -> None:
    base, planner = server
    status, body = post(base + "/api/client-metrics", {
        "first_pose_ms": 123.4, "ready_ms": 88.0, "clips": 1, "style": "walk",
        "frames": 44, "fps": 30, "playing": True, "device": "mlx-gpu"})
    assert status == 200 and body["stored"] is True
    assert planner.client is not None
    status, body = get(base + "/api/stats")
    assert status == 200
    assert body["client"]["first_pose_ms"] == 123.4
    assert body["client"]["device"] == "mlx-gpu"


def test_bad_json_body_is_a_400(server) -> None:
    base, _ = server
    request = urllib.request.Request(base + "/api/plan", data=b"{not json", method="POST",
                                     headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            status, body = response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        status, body = error.code, json.loads(error.read())
    assert status == 400
    assert "bad json" in body["error"]


@pytest.mark.parametrize("route", [
    "/../scripts/mlx_viewer.py",
    "/..%2fscripts%2fmlx_viewer.py",
    "/%2e%2e/%2e%2e/etc/hosts",
])
def test_traversal_outside_the_viewer_root_is_refused(server, route: str) -> None:
    base, _ = server
    status, body = get(base + route)
    assert status in (403, 404), (route, status)
    if isinstance(body, dict):
        assert body.get("ok") is False


def test_missing_asset_is_a_404(server) -> None:
    base, _ = server
    status, body = get(base + "/nope.css")
    assert status == 404
    assert body["ok"] is False


def test_static_path_guard_is_the_mechanism() -> None:
    """Direct unit check of the guard, independent of URL normalisation."""
    class Bound(Handler):
        planner = StubPlanner()

    handler = Bound.__new__(Bound)
    assert handler.static_path("/") == mlx_viewer.VIEWER_ROOT / "index.html"
    assert handler.static_path("/app.js") == mlx_viewer.VIEWER_ROOT / "app.js"
    assert handler.static_path("/../../etc/hosts") is None
    assert handler.static_path("/../NOTICE") is None
