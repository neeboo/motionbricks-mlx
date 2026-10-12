// viewer/rig.js -- skeleton rig + ground-path renderer.
//
// PROVENANCE: the code below is copied VERBATIM from the Apache-2.0 reference
// demo `demo/web/app.js` in the read-only checkout ~/dev/motion-bricks-cpp/repo:
//   * lines  88-95  -- the shared cylinder/sphere/octahedron geometries and the
//                      module-level vector scratch variables
//   * lines  97-113 -- `labelSprite`
//   * lines 115-252 -- `class SkeletonRig`
//   * lines 254-272 -- `makeLine` / `setGroundPath`
// Copyright the motion-bricks-cpp authors. Licensed under the Apache License,
// Version 2.0 (this repository carries the licence text in ../LICENSE and the
// attribution record in ../NOTICE).
//
// The seam we exploit is the one the reference itself defines:
// `SkeletonRig.poseWorld(positions, offset)` consumes exactly the flattened
// `world_positions_xyz` array (frames * joints * 3) that our MLX exporter
// `motionbricks_mlx.export.plan_json` already produces. No renderer adaptation
// was needed -- only a different producer of that array.
//
// The one mechanical edit: the reference reads a module-level `scene` singleton
// from app.js, so here it is injected through `bindScene()` instead. Nothing
// else was changed.

import * as THREE from './vendor/three.module.min.js';

let scene = null;
export function bindScene(target) { scene = target; }

const cylinderGeometry = new THREE.CylinderGeometry(1, 1, 1, 8, 1, false);
const sphereGeometry = new THREE.SphereGeometry(1, 14, 10);
const diamondGeometry = new THREE.OctahedronGeometry(1, 0);
const up = new THREE.Vector3(0, 1, 0);
const startPoint = new THREE.Vector3();
const endPoint = new THREE.Vector3();
const direction = new THREE.Vector3();
const midpoint = new THREE.Vector3();

function labelSprite(text, color, opacity) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const context = canvas.getContext('2d');
  context.font = '700 30px system-ui';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillStyle = color;
  context.fillText(text, 128, 32);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({map: texture, transparent: true, opacity, depthTest: false}));
  sprite.scale.set(0.5, 0.125, 1);
  sprite.renderOrder = 20;
  return sprite;
}

class SkeletonRig {
  constructor(joints, options) {
    this.bones = [];
    this.jointMeshes = [];
    this.segments = [];
    this.group = new THREE.Group();
    this.radius = options.radius;
    this.jointRadius = options.jointRadius;
    this.rootRadius = options.rootRadius;
    this.labelHeight = options.labelHeight ?? 0.34;
    this.labelOffset = options.labelOffset ?? 0;
    this.boneMaterial = new THREE.MeshStandardMaterial({
      color: options.color, emissive: options.emissive, emissiveIntensity: options.emissiveIntensity,
      roughness: 0.42, transparent: options.opacity < 1, opacity: options.opacity,
      depthWrite: options.opacity >= 0.95, depthTest: options.depthTest ?? true,
    });
    this.jointMaterial = new THREE.MeshStandardMaterial({
      color: options.jointColor, emissive: options.emissive, emissiveIntensity: options.emissiveIntensity,
      roughness: 0.35, transparent: options.opacity < 1, opacity: options.opacity,
      depthWrite: options.opacity >= 0.95, depthTest: options.depthTest ?? true,
    });
    this.rootMaterial = new THREE.MeshStandardMaterial({
      color: options.rootColor, emissive: options.rootEmissive, emissiveIntensity: 1.4,
      transparent: options.opacity < 1, opacity: options.opacity, depthWrite: options.opacity >= 0.95,
      depthTest: options.depthTest ?? true,
    });
    for (let index = 0; index < joints.length; index++) {
      const joint = joints[index];
      const bone = new THREE.Bone();
      bone.name = joint.name;
      if (joint.parent < 0) {
        bone.position.set(0, 0, 0);
      } else {
        const parent = joints[joint.parent].position;
        bone.position.set(joint.position[0] - parent[0], joint.position[1] - parent[1], joint.position[2] - parent[2]);
        this.bones[joint.parent].add(bone);
      }
      this.bones.push(bone);
      const geometry = options.diamonds ? diamondGeometry : sphereGeometry;
      const marker = new THREE.Mesh(geometry, index === 0 ? this.rootMaterial : this.jointMaterial);
      marker.castShadow = !options.diamonds;
      marker.renderOrder = options.renderOrder;
      this.jointMeshes.push(marker);
      this.group.add(marker);
      if (joint.parent >= 0) {
        const segment = new THREE.Mesh(cylinderGeometry, this.boneMaterial);
        segment.castShadow = !options.diamonds;
        segment.renderOrder = options.renderOrder;
        this.segments.push({mesh: segment, child: index, parent: joint.parent});
        this.group.add(segment);
      }
    }
    scene.add(this.bones[0]);
    scene.add(this.group);
    this.label = options.label ? labelSprite(options.label, options.labelColor, options.opacity) : null;
    if (this.label) scene.add(this.label);
  }

  pose(roots, rotations, frame, frameCount) {
    const index = Math.min(frameCount - 1, Math.max(0, frame));
    this.bones[0].position.fromArray(roots, index * 3);
    for (let joint = 0; joint < this.bones.length; joint++) {
      this.bones[joint].quaternion.fromArray(rotations, (index * this.bones.length + joint) * 4);
    }
    this.updateGeometry();
  }

  poseInterpolated(roots, rotations, frame, frameCount) {
    const at=THREE.MathUtils.clamp(frame,0,frameCount-1), first=Math.floor(at),next=Math.min(first+1,frameCount-1),u=at-first;
    this.bones[0].position.fromArray(roots,first*3).lerp(new THREE.Vector3().fromArray(roots,next*3),u);
    const q=new THREE.Quaternion();
    for(let j=0;j<this.bones.length;j++)this.bones[j].quaternion.fromArray(rotations,(first*this.bones.length+j)*4)
      .slerp(q.fromArray(rotations,(next*this.bones.length+j)*4),u);
    this.updateGeometry();
  }

  poseWorld(positions, offset = 0) {
    for (let joint = 0; joint < this.jointMeshes.length; joint++) {
      const marker = this.jointMeshes[joint];
      marker.position.fromArray(positions, offset + joint * 3);
      marker.scale.setScalar(joint === 0 ? this.rootRadius : this.jointRadius);
    }
    for (const segment of this.segments) {
      startPoint.copy(this.jointMeshes[segment.parent].position);
      endPoint.copy(this.jointMeshes[segment.child].position);
      direction.subVectors(endPoint, startPoint);
      const length = direction.length();
      midpoint.addVectors(startPoint, endPoint).multiplyScalar(0.5);
      segment.mesh.position.copy(midpoint);
      segment.mesh.quaternion.setFromUnitVectors(up, direction.normalize());
      segment.mesh.scale.set(this.radius, length, this.radius);
    }
    if (this.label) {
      endPoint.copy(this.jointMeshes[0].position);
      this.label.position.set(endPoint.x + this.labelOffset, endPoint.y + this.labelHeight, endPoint.z);
    }
  }

  updateGeometry() {
    this.bones[0].updateMatrixWorld(true);
    for (let index = 0; index < this.bones.length; index++) {
      this.bones[index].getWorldPosition(endPoint);
      const marker = this.jointMeshes[index];
      marker.position.copy(endPoint);
      const radius = index === 0 ? this.rootRadius : this.jointRadius;
      marker.scale.setScalar(radius);
    }
    for (const segment of this.segments) {
      this.bones[segment.parent].getWorldPosition(startPoint);
      this.bones[segment.child].getWorldPosition(endPoint);
      direction.subVectors(endPoint, startPoint);
      const length = direction.length();
      midpoint.addVectors(startPoint, endPoint).multiplyScalar(0.5);
      segment.mesh.position.copy(midpoint);
      segment.mesh.quaternion.setFromUnitVectors(up, direction.normalize());
      segment.mesh.scale.set(this.radius, length, this.radius);
    }
    if (this.label) {
      this.bones[0].getWorldPosition(endPoint);
      this.label.position.set(endPoint.x + this.labelOffset, endPoint.y + this.labelHeight, endPoint.z);
    }
  }

  setVisible(visible) {
    this.group.visible = visible;
    this.bones[0].visible = visible;
    if (this.label) this.label.visible = visible;
  }

  setOpacity(opacity) {
    for (const material of [this.boneMaterial, this.jointMaterial, this.rootMaterial]) {
      material.opacity = opacity;
      material.transparent = opacity < 1;
      material.depthWrite = opacity >= 0.95;
    }
    if (this.label) this.label.material.opacity = opacity;
  }
}

function makeLine(color, dashed = false) {
  const material = dashed
    ? new THREE.LineDashedMaterial({color, dashSize: 0.09, gapSize: 0.055, transparent: true, opacity: 0.9})
    : new THREE.LineBasicMaterial({color, transparent: true, opacity: 0.72});
  const line = new THREE.Line(new THREE.BufferGeometry(), material);
  line.renderOrder = 1;
  scene.add(line);
  return line;
}

function setGroundPath(line, roots, frames) {
  const points = [];
  for (let frame = 0; frame < frames; frame++) {
    points.push(new THREE.Vector3(roots[frame * 3], 0.018, roots[frame * 3 + 2]));
  }
  line.geometry.dispose();
  line.geometry = new THREE.BufferGeometry().setFromPoints(points);
  if (line.material.isLineDashedMaterial) line.computeLineDistances();
}
export {SkeletonRig, labelSprite, makeLine, setGroundPath};
