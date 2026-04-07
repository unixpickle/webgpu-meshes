export type Vec3 = [number, number, number];

export function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function scale(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}

export function addScalar(a: Vec3, s: number): Vec3 {
  return [a[0] + s, a[1] + s, a[2] + s];
}

export function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

export function length(a: Vec3): number {
  return Math.hypot(a[0], a[1], a[2]);
}

export function normalizeSafe(a: Vec3, fallback: Vec3): Vec3 {
  const len = length(a);
  if (!(len > 1e-20) || !Number.isFinite(len)) return fallback;
  return scale(a, 1 / len);
}

export function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

export function clampVec3(v: Vec3, lo: Vec3, hi: Vec3): Vec3 {
  return [clamp(v[0], lo[0], hi[0]), clamp(v[1], lo[1], hi[1]), clamp(v[2], lo[2], hi[2])];
}

export function orthoBasis(axis: Vec3): [Vec3, Vec3] {
  const helper = Math.abs(axis[0]) < 0.5 ? [1, 0, 0] as Vec3 : [0, 1, 0] as Vec3;
  const b1 = normalizeSafe(cross(axis, helper), [0, 0, 1]);
  const b2 = normalizeSafe(cross(axis, b1), [0, 1, 0]);
  return [b1, b2];
}
