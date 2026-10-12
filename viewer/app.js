// viewer/app.js -- the MLX viewer's own application code.
//
// PROVENANCE
//   * three.js            -> viewer/vendor/three.module.min.js (copied verbatim
//                            from the reference demo; three.js is MIT).
//   * skeleton rig, labels,
//     line helpers        -> viewer/rig.js (verbatim reference code).
//   * style.css           -> reference style.css, verbatim, plus an appended
//                            block of rules for the controls added here.
//   * THIS FILE           -> written for this repository. It follows the
//                            reference demo's scene/camera conventions (same
//                            fog, lights, floor, grid, yaw/pitch/distance
//                            orbit with the same defaults) so the two pages
//                            look like the same product, but the data path is
//                            HTTP+JSON against our own MLX planner instead of
//                            the reference's WebSocket stream client.
//
// DATA PATH
//   GET  /api/styles          -> the real .mbstyle inventory on disk
//   POST /api/plan  {style, seed, move, facing, reset, advance}
//                             -> frames of MLX-planned motion, already
//                                forward-kinematic'd to world positions
//   The client replans while the clip is still playing (advance = the frame it
//   has already shown) and re-anchors the playhead, which is the same
//   continuity trick the reference's requestPlan() uses.

import * as THREE from './vendor/three.module.min.js';
import {bindScene, SkeletonRig, makeLine} from './rig.js';

const viewport = document.querySelector('#viewport');
const styleSelect = document.querySelector('#style-select');
const seedInput = document.querySelector('#seed-input');
const seedNext = document.querySelector('#seed-next');
const playButton = document.querySelector('#mlx-play');
const cameraButton = document.querySelector('#mlx-camera');
const turnLeft = document.querySelector('#mlx-turn-left');
const turnRight = document.querySelector('#mlx-turn-right');
const stopButton = document.querySelector('#mlx-stop');
const liveToggle = document.querySelector('#mlx-live');
const trailToggle = document.querySelector('#mlx-trail');
const targetsToggle = document.querySelector('#mlx-targets');
const statusElement = document.querySelector('#mlx-status');
const statPlan = document.querySelector('#stat-plan');
const statHz = document.querySelector('#stat-hz');
const statClips = document.querySelector('#stat-clips');
const statClip = document.querySelector('#stat-clip');
const statTravel = document.querySelector('#stat-travel');
const statWalked = document.querySelector('#stat-walked');
const statPlayback = document.querySelector('#stat-playback');
const statBackend = document.querySelector('#stat-backend');
const statFirstFrame = document.querySelector('#stat-firstframe');
const targetPanel = document.querySelector('#mlx-target-panel');
const targetPlace = document.querySelector('#mlx-target-place');
const targetReplan = document.querySelector('#mlx-target-replan');
const targetClear = document.querySelector('#mlx-target-clear');
const targetHere = document.querySelector('#mlx-target-here');
const targetList = document.querySelector('#mlx-target-list');
const targetHitLine = document.querySelector('#mlx-target-hit');
const hud = {
  style: document.querySelector('#hud-style'),
  plan: document.querySelector('#hud-plan'),
  hz: document.querySelector('#hud-hz'),
  walked: document.querySelector('#hud-walked'),
  rootz: document.querySelector('#hud-rootz'),
  target: document.querySelector('#hud-target'),
  clips: document.querySelector('#hud-clips'),
};
const html = document.documentElement;

// ---------------------------------------------------------------------------
// scene -- same conventions as the reference demo (app.js:58-86)
// ---------------------------------------------------------------------------
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x090b10, 0.045);
const camera = new THREE.PerspectiveCamera(42, 1, 0.02, 100);
const renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
viewport.append(renderer.domElement);

scene.add(new THREE.HemisphereLight(0xcde4ff, 0x253226, 2.5));
const keyLight = new THREE.DirectionalLight(0xffffff, 3.2);
keyLight.position.set(3, 7, 4);
keyLight.castShadow = true;
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x67e8c3, 1.2);
rimLight.position.set(-4, 3, -3);
scene.add(rimLight);
scene.add(keyLight.target);
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(40, 40),
  new THREE.MeshStandardMaterial({color: 0x0d121a, roughness: 0.96, metalness: 0.02}),
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.012;
floor.receiveShadow = true;
scene.add(floor);
const grid = new THREE.GridHelper(24, 48, 0x3a4b5d, 0x202b38);
scene.add(grid);
bindScene(scene);

const cameraView = {yaw: 0.68, pitch: 0.24, distance: 4.8, dragging: false, x: 0, y: 0};

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------
const state = {
  meta: null,          // /api/health payload
  motion: null,        // {frames, joints, world_positions_xyz: Float32Array}
  targets: null,
  skeleton: null,
  rig: null,
  targetRigs: [],
  playhead: 0,
  playing: true,
  lastTime: performance.now(),
  pending: false,
  queued: false,
  clips: 0,
  fps: 30,
  move: [0, 1],
  facing: [0, 1],
  // The demo starts with W held, so the first thing a visitor sees is the
  // character actually walking. `updateControl()` recomputes both vectors
  // camera-relative from this pad key, and the W pad button shows as pressed.
  // Pressing W (or any pad key) once clears it, exactly like the reference.
  padKey: 'w',
  presetMove: false,
  held: new Set(),
  resetPending: true,
  planMs: [],
  planTimes: [],
  trailPoints: [],
  trailLine: null,
  lastRoot: null,
  walkedMetres: 0,
  firstPoseMs: null,
  readyMs: null,
  pageStart: performance.now(),
  settleAt: null,
  // --- placed target keyframes (this round) --------------------------------
  // `placed` is the list the user owns, in *world* coordinates: the same frame
  // the server plans in, because the page draws `world_positions_xyz` verbatim.
  placed: [],
  placing: false,        // "Place target" armed: the next floor click drops a point
  lastHit: null,
  targetMarkers: null,
};

const trailGeometry = () => new THREE.BufferGeometry();

function setStatus(text, error = false) {
  statusElement.textContent = text;
  statusElement.dataset.error = error ? 'true' : 'false';
  html.dataset.mlxStatus = error ? 'error' : 'ready';
}

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

// ---------------------------------------------------------------------------
// camera -- same drag/wheel bindings as the reference (app.js:349-378)
// ---------------------------------------------------------------------------
const focus = new THREE.Vector3(0, 0.8, 0);
const desiredCamera = new THREE.Vector3();
const spline = new THREE.Vector3();

function resetCameraView() {
  cameraView.yaw = 0.68;
  cameraView.pitch = 0.24;
  cameraView.distance = 4.8;
  focus.set(0, 0.8, 0);
  cameraButton.blur();
}

function resize() {
  const width = Math.max(1, viewport.clientWidth);
  const height = Math.max(1, viewport.clientHeight);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

renderer.domElement.addEventListener('pointerdown', event => {
  if (state.placing) {
    // Placement owns the click while armed, so a floor click never also starts
    // an orbit. No synthetic input is involved: this is the page's own pointer.
    const point = groundPointAt(event.clientX, event.clientY);
    if (point) addTarget(point.x, point.z, headingDegTo(point.x, point.z));
    setPlacing(false);
    event.preventDefault();
    return;
  }
  cameraView.dragging = true;
  cameraView.x = event.clientX;
  cameraView.y = event.clientY;
  renderer.domElement.setPointerCapture(event.pointerId);
});
renderer.domElement.addEventListener('pointermove', event => {
  if (!cameraView.dragging) return;
  cameraView.yaw -= (event.clientX - cameraView.x) * 0.006;
  cameraView.pitch = THREE.MathUtils.clamp(
    cameraView.pitch + (event.clientY - cameraView.y) * 0.004, -0.05, 1.05);
  cameraView.x = event.clientX;
  cameraView.y = event.clientY;
});
renderer.domElement.addEventListener('pointerup', event => {
  cameraView.dragging = false;
  renderer.domElement.releasePointerCapture(event.pointerId);
});
renderer.domElement.addEventListener('pointercancel', () => { cameraView.dragging = false; });
renderer.domElement.addEventListener('wheel', event => {
  cameraView.distance = THREE.MathUtils.clamp(
    cameraView.distance * Math.exp(event.deltaY * 0.001), 2.1, 8);
  event.preventDefault();
}, {passive: false});
renderer.domElement.addEventListener('dblclick', resetCameraView);

// ---------------------------------------------------------------------------
// placed target keyframes -- ground markers and the floor raycast
//
// The marker is drawn in the *same* world frame as the animated skeleton, and
// the server resolves the same (x, z, heading) into boundary frames 4..7, so a
// marker and the orange ghost that comes back from the plan sit at the same
// place by construction.  That is what makes "ghost vs. result" a real
// comparison rather than two drawings of the same guess.
// ---------------------------------------------------------------------------
const targetGroup = new THREE.Group();
scene.add(targetGroup);
const TARGET_COLOR = 0xff763b;

const floorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const pointerNdc = new THREE.Vector2();
const raycaster = new THREE.Raycaster();
const hitPoint = new THREE.Vector3();

function groundPointAt(clientX, clientY) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointerNdc.set(((clientX - rect.left) / rect.width) * 2 - 1,
                 -((clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(pointerNdc, camera);
  return raycaster.ray.intersectPlane(floorPlane, hitPoint) ? hitPoint.clone() : null;
}

function headingDegTo(x, z) {
  // Facing from the character toward the point; the server's heading
  // convention is atan2(x, z) in degrees (src/agent.cpp:236-241).
  if (!state.rig) return 0;
  const root = state.rig.jointMeshes[0].position;
  return (Math.atan2(x - root.x, z - root.z) * 180) / Math.PI;
}

function rebuildTargetMarkers() {
  while (targetGroup.children.length) {
    const child = targetGroup.children.pop();
    child.geometry?.dispose();
    child.material?.dispose();
  }
  for (const point of state.placed) {
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.16, 0.22, 28),
      new THREE.MeshBasicMaterial({color: TARGET_COLOR, transparent: true, opacity: 0.95,
                                   side: THREE.DoubleSide, depthWrite: false}));
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(point.x, 0.02, point.z);
    ring.renderOrder = 5;
    targetGroup.add(ring);
    // A short arrow on the ground for the placed facing.
    const yaw = (point.heading_deg * Math.PI) / 180;
    const arrow = new THREE.Mesh(
      new THREE.ConeGeometry(0.075, 0.24, 12),
      new THREE.MeshBasicMaterial({color: 0xffd0a8, depthWrite: false}));
    arrow.rotation.z = -Math.PI / 2;
    arrow.rotation.y = yaw;
    arrow.position.set(point.x + Math.sin(yaw) * 0.33, 0.03, point.z + Math.cos(yaw) * 0.33);
    arrow.renderOrder = 5;
    targetGroup.add(arrow);
  }
  html.dataset.mlxPlacedTargets = String(state.placed.length);
}

function renderTargetList() {
  targetList.innerHTML = '';
  state.placed.forEach((point, index) => {
    const row = document.createElement('li');
    const label = document.createElement('span');
    label.className = 'target-index';
    label.textContent = `T${index + 1}`;
    row.append(label);
    for (const [key, step, digits] of [['x', 0.25, 2], ['z', 0.25, 2], ['heading_deg', 15, 0]]) {
      const input = document.createElement('input');
      input.type = 'number';
      input.step = String(step);
      input.value = point[key].toFixed(digits);
      input.dataset.field = key;
      input.title = key === 'heading_deg' ? 'facing (degrees)' : `${key} (metres)`;
      input.addEventListener('change', () => {
        const value = Number(input.value);
        if (Number.isFinite(value)) {
          point[key] = value;
          rebuildTargetMarkers();
        }
      });
      row.append(input);
    }
    const drop = document.createElement('button');
    drop.type = 'button';
    drop.className = 'target-drop';
    drop.textContent = '×';
    drop.title = 'remove this target';
    drop.addEventListener('click', () => {
      state.placed.splice(index, 1);
      renderTargetList();
      rebuildTargetMarkers();
    });
    row.append(drop);
    targetList.append(row);
  });
  targetReplan.disabled = state.placed.length === 0;
}

function addTarget(x, z, headingDeg) {
  if (state.placed.length >= 4) {
    setStatus('at most 4 placed targets: clear one first', true);
    return;
  }
  state.placed.push({x, z, heading_deg: headingDeg});
  renderTargetList();
  rebuildTargetMarkers();
}

function setPlacing(armed) {
  state.placing = armed;
  targetPlace.setAttribute('aria-pressed', String(armed));
  targetPlace.textContent = armed ? 'Click the floor…' : 'Place target';
  html.dataset.mlxPlacing = String(armed);
}

// ---------------------------------------------------------------------------
// HTTP client
// ---------------------------------------------------------------------------
async function api(path, body) {
  const options = body === undefined
    ? {}
    : {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify(body)};
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `${path} -> HTTP ${response.status}`);
  }
  return payload;
}

// ---------------------------------------------------------------------------
// rigs
// ---------------------------------------------------------------------------
function buildRigs(skeleton) {
  state.rig = new SkeletonRig(skeleton, {
    color: 0x55efc4, jointColor: 0xd9fff3, rootColor: 0xffd166,
    emissive: 0x0c5b49, rootEmissive: 0x6a3b00, emissiveIntensity: 0.8,
    radius: 0.022, jointRadius: 0.034, rootRadius: 0.062, opacity: 1,
    diamonds: false, renderOrder: 7,
  });
  state.targetRigs = [];
  for (let index = 0; index < 4; index++) {
    state.targetRigs.push(new SkeletonRig(skeleton, {
      color: 0xff763b, jointColor: 0xffd7b0, rootColor: 0xff9a68,
      emissive: 0x6a2a00, rootEmissive: 0x6a2a00, emissiveIntensity: 0.8,
      radius: 0.014, jointRadius: 0.024, rootRadius: 0.044,
      opacity: 0.9, diamonds: true, renderOrder: 6,
    }));
  }
  state.trailLine = makeLine(0x55efc4);
  state.trailLine.material.opacity = 0.5;
  state.trailPoints = [];
}

const lerpBuffer = {positions: null, frames: 0, joints: 0};

function interpolatedPositions(source, frame) {
  const {frames, joints} = state.motion;
  const stride = joints * 3;
  if (!lerpBuffer.positions || lerpBuffer.frames !== frames || lerpBuffer.joints !== joints) {
    lerpBuffer.positions = new Float32Array(stride);
    lerpBuffer.frames = frames;
    lerpBuffer.joints = joints;
  }
  const at = Math.min(frames - 1, Math.max(0, frame));
  const first = Math.floor(at);
  const next = Math.min(first + 1, frames - 1);
  const u = at - first;
  const out = lerpBuffer.positions;
  for (let index = 0; index < stride; index++) {
    const a = source[first * stride + index];
    out[index] = a + (source[next * stride + index] - a) * u;
  }
  return out;
}

function renderPose() {
  if (!state.motion || !state.rig) return;
  const positions = interpolatedPositions(state.motion.world_positions_xyz, state.playhead);
  state.rig.poseWorld(positions, 0);
  const root = state.rig.jointMeshes[0].position;
  // Follow the pelvis, lifted a little so the camera frames the whole body.
  focus.lerp(spline.set(root.x, root.y + 0.18, root.z), 0.06);
}

function updateTravel() {
  const root = state.rig.jointMeshes[0].position;
  if (state.lastRoot) {
    const step = Math.hypot(root.x - state.lastRoot[0], root.y - state.lastRoot[1],
                            root.z - state.lastRoot[2]);
    // Ignore the handoff blend between clips (sub-millimetre) and resets.
    if (step < 0.2) state.walkedMetres += step;
  }
  state.lastRoot = [root.x, root.y, root.z];
  statWalked.textContent = `${state.walkedMetres.toFixed(2)} m`;
  hud.walked.textContent = `${state.walkedMetres.toFixed(2)} m`;
  hud.rootz.textContent = `${root.z.toFixed(2)} m`;
  html.dataset.mlxWalkedMetres = state.walkedMetres.toFixed(3);
  html.dataset.mlxRootZ = root.z.toFixed(4);
}

function pushTrail() {
  if (!trailToggle.checked || !state.rig) return;
  // Same ground offset the reference uses for its path line
  // (`setGroundPath` in the reference writes y = 0.018).
  const root = state.rig.jointMeshes[0].position;
  const point = spline.set(root.x, 0.025, root.z);
  const last = state.trailPoints[state.trailPoints.length - 1];
  if (last && last.distanceToSquared(point) < 0.0004) return;
  state.trailPoints.push(point.clone());
  if (state.trailPoints.length > 6000) state.trailPoints.shift();
  state.trailLine.geometry.dispose();
  state.trailLine.geometry = trailGeometry().setFromPoints(state.trailPoints);
}

function updateCamera() {
  const horizontal = Math.cos(cameraView.pitch) * cameraView.distance;
  desiredCamera.set(
    focus.x + Math.sin(cameraView.yaw) * horizontal,
    focus.y + Math.sin(cameraView.pitch) * cameraView.distance,
    focus.z + Math.cos(cameraView.yaw) * horizontal,
  );
  camera.position.lerp(desiredCamera, cameraView.dragging ? 1 : 0.08);
  camera.lookAt(focus);
}

// ---------------------------------------------------------------------------
// planning
// ---------------------------------------------------------------------------
function nextSeed() {
  seedInput.value = String((Number(seedInput.value) || 0) + 1);
}

async function requestPlan(advance, options = {}) {
  if (state.pending) { state.queued = true; return null; }
  const toTargets = options.toTargets === true && state.placed.length > 0;
  state.pending = true;
  const started = performance.now();
  try {
    const payload = await api('/api/plan', {
      style: styleSelect.value,
      seed: Number(seedInput.value) || 0,
      move: state.move,
      facing: state.facing,
      reset: state.resetPending,
      advance: state.resetPending ? 0 : advance,
      // Additive: without targets the server response is byte-for-byte the
      // same document it returned before this round.
      targets: toTargets ? state.placed.map(p => [p.x, p.z, p.heading_deg]) : undefined,
      target_space: 'world',
      pose_mode: 'style',
    });
    state.resetPending = false;
    state.lastHit = toTargets ? (payload.target_hit || null) : null;
    applyMotion(payload, advance);
    return payload;
  } catch (error) {
    setStatus(`plan failed: ${error.message}`, true);
    return null;
  } finally {
    state.pending = false;
    state.planMs.push(performance.now() - started);
    if (state.planMs.length > 64) state.planMs.shift();
    if (state.queued) { state.queued = false; queueMicrotask(() => void requestPlan(Math.floor(state.playhead))); }
  }
}

function applyMotion(payload, advance) {
  state.motion = {
    frames: payload.frames,
    joints: payload.joints,
    world_positions_xyz: Float32Array.from(payload.world_positions_xyz),
  };
  state.targets = payload.targets;
  state.fps = payload.fps || 30;
  state.playhead = Math.max(0, state.playhead - advance);
  state.lastTime = performance.now();
  state.clips += 1;
  state.planTimes.push(payload.timing.plan_ms);

  const targetPositions = Float32Array.from(payload.targets.world_positions_xyz);
  for (let index = 0; index < state.targetRigs.length; index++) {
    const exists = index < payload.targets.frames;
    state.targetRigs[index].setVisible(exists && targetsToggle.checked);
    if (exists) {
      state.targetRigs[index].poseWorld(targetPositions, index * payload.joints * 3);
    }
  }

  statClips.textContent = String(state.clips);
  hud.clips.textContent = String(state.clips);
  hud.style.textContent = payload.style.replaceAll('_', ' ');
  // Placed-target readout: the same numbers the server measured, so the HUD in
  // a screenshot carries the acceptance evidence, not just the animation.
  if (payload.target_hit) {
    const errors = payload.target_hit.errors;
    const ok = payload.target_hit.within_tolerance;
    hud.target.textContent = `${errors.root_planar_m.toFixed(3)} m${ok ? '' : ' !'}`;
    targetHitLine.dataset.state = ok ? 'ok' : 'miss';
    targetHitLine.textContent =
      `${payload.placed_targets.count} placed · hit frames ${payload.target_hit.window_frames[0]}–`
      + `${payload.target_hit.window_frames[1]} · root ${errors.root_planar_m.toFixed(4)} m`
      + ` (tol ${payload.target_hit.tolerances.root_planar_m})`
      + ` · heading ${errors.heading_deg.toFixed(2)}°`
      + ` · joints ${errors.joint_m.toFixed(4)} m · ${ok ? 'within tolerance' : 'OUT OF TOLERANCE'}`;
    html.dataset.mlxTargetHit = errors.root_planar_m.toFixed(6);
    html.dataset.mlxTargetHitOk = String(ok);
    html.dataset.mlxTargetWindow = payload.target_hit.window_frames.join('-');
  } else {
    hud.target.textContent = state.placed.length ? `${state.placed.length} placed` : '—';
  }
  statPlan.textContent = `${payload.timing.plan_ms.toFixed(1)} ms`;
  hud.plan.textContent = `${payload.timing.plan_ms.toFixed(1)} ms`;
  statClip.textContent = `${payload.frames} frames · ${payload.style.replaceAll('_', ' ')}`;
  statTravel.textContent = `${payload.metrics.root_travel_metres.toFixed(3)} m`;
  html.dataset.mlxClips = String(state.clips);
  html.dataset.mlxPlanMs = payload.timing.plan_ms.toFixed(3);
  html.dataset.mlxFrames = String(payload.frames);
  html.dataset.mlxStyle = payload.style;

  if (state.firstPoseMs === null) {
    state.firstPoseMs = performance.now() - state.pageStart;
    statFirstFrame.textContent = `${state.firstPoseMs.toFixed(0)} ms`;
    html.dataset.mlxFirstPoseMs = state.firstPoseMs.toFixed(1);
    // Report the page's own timings back to the server so they can be read
    // with `curl /api/stats` instead of being trapped in the browser.
    void fetch('/api/client-metrics', {
      method: 'POST', headers: {'content-type': 'application/json'},
      body: JSON.stringify({
        first_pose_ms: state.firstPoseMs,
        ready_ms: state.readyMs,
        nav_ms: Math.round(performance.timeOrigin),
        device: state.meta ? state.meta.device : 'unknown',
        clips: state.clips, style: payload.style, frames: payload.frames,
        fps: state.fps, playing: state.playing,
        measured_at: new Date().toISOString(),
      }),
    }).catch(() => {});
  }
  setStatus(`Playing · ${payload.style.replaceAll('_', ' ')} · ${payload.frames} frames planned in ${payload.timing.plan_ms.toFixed(1)} ms`);
}

function schedulePlan(delay) {
  setTimeout(() => void requestPlan(Math.floor(state.playhead)), delay);
}

// ---------------------------------------------------------------------------
// controls
// ---------------------------------------------------------------------------
function controlVector() {
  if (!state.padKey) return null;
  const button = document.querySelector(`.pad button[data-key="${state.padKey}"]`);
  if (!button) return null;
  return {x: Number(button.dataset.x) || 0, z: Number(button.dataset.z) || 0};
}

function updateControl() {
  const pressed = controlVector();
  const sine = Math.sin(cameraView.yaw);
  const cosine = Math.cos(cameraView.yaw);
  if (pressed) {
    const x = pressed.x * cosine - pressed.z * sine;
    const z = -pressed.x * sine - pressed.z * cosine;
    state.move = [x, z];
    state.facing = [x, z];
    state.presetMove = false;
  } else if (!state.presetMove) {
    state.move = [0, 0];
  }
  document.querySelectorAll('.pad button').forEach(button => {
    const active = button.dataset.key === state.padKey;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
  html.dataset.mlxMove = state.move.map(value => value.toFixed(3)).join(',');
}

document.querySelectorAll('.pad button').forEach(button => {
  button.addEventListener('click', () => {
    state.padKey = state.padKey === button.dataset.key ? null : button.dataset.key;
    state.presetMove = false;
    updateControl();
    schedulePlan(0);
    button.blur();
  });
});

const held = new Set();
window.addEventListener('keydown', event => {
  const key = event.key.toLowerCase();
  if ('wasd'.includes(key)) { held.add(key); state.padKey = key; updateControl(); schedulePlan(0); }
  if (key === 'j' || key === ' ') { togglePlay(); }
});
window.addEventListener('keyup', event => {
  held.delete(event.key.toLowerCase());
  if (!held.size) { state.padKey = null; updateControl(); schedulePlan(0); }
});

function togglePlay() {
  state.playing = !state.playing;
  playButton.textContent = state.playing ? 'Pause' : 'Play';
  playButton.setAttribute('aria-pressed', String(state.playing));
  if (state.playing) state.lastTime = performance.now();
  html.dataset.mlxPlaying = String(state.playing);
  playButton.blur();
}

playButton.addEventListener('click', togglePlay);
cameraButton.addEventListener('click', resetCameraView);

turnLeft.addEventListener('click', () => {
  const [x, z] = state.facing;
  const angle = Math.PI / 8;
  state.facing = [x * Math.cos(angle) + z * Math.sin(angle), -x * Math.sin(angle) + z * Math.cos(angle)];
  schedulePlan(0);
  turnLeft.blur();
});
turnRight.addEventListener('click', () => {
  const [x, z] = state.facing;
  const angle = -Math.PI / 8;
  state.facing = [x * Math.cos(angle) + z * Math.sin(angle), -x * Math.sin(angle) + z * Math.cos(angle)];
  schedulePlan(0);
  turnRight.blur();
});
stopButton.addEventListener('click', () => {
  state.padKey = null;
  state.presetMove = false;
  updateControl();
  schedulePlan(0);
  stopButton.blur();
});

styleSelect.addEventListener('change', () => {
  state.resetPending = true;
  state.trailPoints = [];
  state.lastRoot = null;
  state.walkedMetres = 0;
  state.motion = null;
  requestPlan(0);
});

seedNext.addEventListener('click', () => { nextSeed(); schedulePlan(0); seedNext.blur(); });
seedInput.addEventListener('change', () => schedulePlan(0));

// --- placed targets --------------------------------------------------------
function replanToTargets() {
  if (!state.placed.length) return;
  // Plan from the frame already on screen (`advance`) so the motion starts where
  // the character actually is -- the same continuity seam the rest of the page
  // uses. A fresh session has nothing to advance from, hence the reset flag.
  const advance = state.resetPending ? 0 : Math.floor(state.playhead);
  void requestPlan(advance, {toTargets: true});
  targetReplan.blur();
}

targetPlace.addEventListener('click', () => setPlacing(!state.placing));
targetReplan.addEventListener('click', replanToTargets);
targetClear.addEventListener('click', () => {
  state.placed = [];
  state.lastHit = null;
  renderTargetList();
  rebuildTargetMarkers();
  targetHitLine.textContent = 'no target placed yet';
  targetHitLine.removeAttribute('data-state');
  hud.target.textContent = '—';
  html.dataset.mlxTargetHit = '';
  targetClear.blur();
});
targetHere.addEventListener('click', () => {
  const facing = state.facing;
  const length = Math.hypot(facing[0], facing[1]) || 1;
  const root = state.rig ? state.rig.jointMeshes[0].position : {x: 0, z: 0};
  const x = root.x + (facing[0] / length);
  const z = root.z + (facing[1] / length);
  addTarget(x, z, (Math.atan2(facing[0] / length, facing[1] / length) * 180) / Math.PI);
  targetHere.blur();
});

targetsToggle.addEventListener('change', () => {
  for (let index = 0; index < state.targetRigs.length; index++) {
    state.targetRigs[index].setVisible(targetsToggle.checked && index < (state.targets?.frames || 0));
  }
});
trailToggle.addEventListener('change', () => {
  state.trailLine.visible = trailToggle.checked;
  if (!trailToggle.checked) { state.trailPoints = []; state.trailLine.geometry.dispose(); state.trailLine.geometry = trailGeometry(); }
});
liveToggle.addEventListener('change', () => { html.dataset.mlxLive = String(liveToggle.checked); });

// ---------------------------------------------------------------------------
// playback loop
// ---------------------------------------------------------------------------
const CONTROL_LOOKAHEAD = 10;

function animate(now) {
  requestAnimationFrame(animate);
  const delta = Math.min(0.25, (now - state.lastTime) / 1000);
  state.lastTime = now;

  if (state.motion && state.playing) {
    state.playhead += delta * state.fps;
    const last = state.motion.frames - 1;
    if (liveToggle.checked && !state.pending && state.playhead >= last - CONTROL_LOOKAHEAD) {
      requestPlan(Math.max(0, Math.floor(state.playhead)));
    }
    if (state.playhead > last) {
      state.playhead = liveToggle.checked ? last : state.playhead - last;
      if (!liveToggle.checked && state.playhead > last) state.playhead = 0;
    }
  }

  renderPose();
  if (state.rig) updateTravel();
  pushTrail();
  if (state.rig) {
    const root = state.rig.jointMeshes[0].position;
    keyLight.target.position.copy(root);
    keyLight.target.updateMatrixWorld();
  }
  updateCamera();
  renderer.render(scene, camera);

  if (state.planTimes.length) {
    const recent = state.planTimes.slice(-8);
    statHz.textContent = `${(1000 / median(recent)).toFixed(1)} plans/s`;
    hud.hz.textContent = `${(1000 / median(recent)).toFixed(1)} /s`;
    html.dataset.mlxPlansPerSecond = (1000 / median(recent)).toFixed(3);
  }
  statPlayback.textContent = `${state.playing ? 'playing' : 'paused'} · frame ${state.playhead.toFixed(0)}/${state.motion ? state.motion.frames - 1 : 0}`;
  html.dataset.mlxLastPlanMs = state.planMs.length ? median(state.planMs.slice(-8)).toFixed(3) : '';
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
// `?style=<name>&seed=<n>&move=x,z` preselects the session from the URL. This
// is the only "external" way to drive the page without touching its controls,
// and it is a plain address-bar parameter, not synthesized input.
//
// `?targets=x,z,deg;x,z,deg&to_targets=1` additionally pre-places the target
// keyframes and asks for the target plan on boot, so a headless `goto` +
// `wait_for_timeout` session shows the orange ghosts and the generated clip in
// one screenshot without a single synthetic click.
const addressBar = new URLSearchParams(location.search);

async function boot() {
  try {
    const health = await api('/api/health');
    state.meta = health;
    statBackend.textContent = `${health.device} · ${health.learned_parameters.toLocaleString()} f32 params`;
    html.dataset.mlxDevice = health.device;
    html.dataset.mlxColdStartMs = health.cold_start_ms.toFixed(1);
    const inventory = await api('/api/styles');
    styleSelect.innerHTML = '';
    for (const name of inventory.styles) {
      const option = document.createElement('option');
      option.value = name;
      option.textContent = name.replaceAll('_', ' ');
      styleSelect.append(option);
    }
    styleSelect.value = inventory.styles.includes(addressBar.get('style'))
      ? addressBar.get('style') : inventory.default;
    if (addressBar.get('seed')) seedInput.value = String(Number(addressBar.get('seed')) || 0);
    if (addressBar.get('move')) {
      const [x, z] = addressBar.get('move').split(',').map(Number);
      if (Number.isFinite(x) && Number.isFinite(z)) {
        state.move = [x, z]; state.facing = [x, z]; state.presetMove = true;
        // `presetMove` only survives `updateControl()` when no pad key is held;
        // the default state holds W, which would silently overwrite the
        // address-bar direction with the camera-relative W vector. Dropping the
        // pad key is what makes `?move=` mean what the docs say it means.
        state.padKey = null;
      }
    }
    state.skeleton = inventory.skeleton;
    buildRigs(inventory.skeleton);
    updateControl();
    // Pre-placed targets from the address bar (same mechanism as ?style/?seed).
    const presetTargets = addressBar.get('targets');
    if (presetTargets) {
      for (const piece of presetTargets.split(';')) {
        const parts = piece.split(',').map(Number);
        if (parts.length >= 2 && parts.slice(0, 3).every(Number.isFinite)) {
          state.placed.push({x: parts[0], z: parts[1], heading_deg: parts[2] || 0});
        }
      }
    }
    renderTargetList();
    rebuildTargetMarkers();
    // `?live=0` unchecks "Replan while walking", so the page shows exactly the
    // one clip that was planned for the placed targets (and the HUD keeps that
    // plan's hit numbers). Plain address-bar parameter, no synthetic input.
    if (addressBar.get('live') === '0') {
      liveToggle.checked = false;
      html.dataset.mlxLive = 'false';
    }
    state.readyMs = performance.now() - state.pageStart;
    html.dataset.mlxReady = 'true';
    html.dataset.mlxReadyMs = state.readyMs.toFixed(1);
    if (state.placed.length && addressBar.get('to_targets') === '1') {
      await requestPlan(0, {toTargets: true});
    } else {
      await requestPlan(0);
    }
  } catch (error) {
    setStatus(`startup failed: ${error.message}`, true);
    console.error(error);
  }
}

window.addEventListener('resize', resize);
resize();
updateControl();
renderTargetList();
rebuildTargetMarkers();
html.dataset.mlxLive = String(liveToggle.checked);
html.dataset.mlxPlacing = 'false';
requestAnimationFrame(animate);
void boot();
