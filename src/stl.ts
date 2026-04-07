import type { IndexedMesh } from './mesh';

export function meshToBinarySTL(mesh: IndexedMesh, solidName = 'dual-contour-mesh'): Blob {
  if (mesh.indices.length % 3 !== 0) {
    throw new Error('meshToBinarySTL(): mesh indices length must be a multiple of 3.');
  }

  const triangleCount = mesh.indices.length / 3;
  const buffer = new ArrayBuffer(84 + triangleCount * 50);
  const bytes = new Uint8Array(buffer, 0, 80);
  const header = `${solidName}`.slice(0, 80);
  for (let i = 0; i < header.length; i++) {
    bytes[i] = header.charCodeAt(i) & 0x7f;
  }

  const view = new DataView(buffer);
  view.setUint32(80, triangleCount, true);

  for (let triIndex = 0; triIndex < triangleCount; triIndex++) {
    const indexBase = triIndex * 3;
    const a = mesh.indices[indexBase + 0] * 3;
    const b = mesh.indices[indexBase + 1] * 3;
    const c = mesh.indices[indexBase + 2] * 3;
    const ax = mesh.positions[a + 0];
    const ay = mesh.positions[a + 1];
    const az = mesh.positions[a + 2];
    const bx = mesh.positions[b + 0];
    const by = mesh.positions[b + 1];
    const bz = mesh.positions[b + 2];
    const cx = mesh.positions[c + 0];
    const cy = mesh.positions[c + 1];
    const cz = mesh.positions[c + 2];
    const [nx, ny, nz] = triangleNormal(
      [ax, ay, az],
      [bx, by, bz],
      [cx, cy, cz],
    );

    let offset = 84 + triIndex * 50;
    for (const value of [nx, ny, nz, ax, ay, az, bx, by, bz, cx, cy, cz]) {
      view.setFloat32(offset, value, true);
      offset += 4;
    }
    view.setUint16(offset, 0, true);
  }

  return new Blob([buffer], { type: 'model/stl' });
}

function triangleNormal(a: [number, number, number], b: [number, number, number], c: [number, number, number]): [number, number, number] {
  const abx = b[0] - a[0];
  const aby = b[1] - a[1];
  const abz = b[2] - a[2];
  const acx = c[0] - a[0];
  const acy = c[1] - a[1];
  const acz = c[2] - a[2];
  const nx = aby * acz - abz * acy;
  const ny = abz * acx - abx * acz;
  const nz = abx * acy - aby * acx;
  const length = Math.hypot(nx, ny, nz);
  if (!(length > 1e-20) || !Number.isFinite(length)) {
    return [0, 0, 0];
  }
  return [nx / length, ny / length, nz / length];
}
