import type { Vec3 } from './vec3';

export interface IndexedMesh {
  positions: Float32Array;
  indices: Uint32Array;
}

export interface CPUVertex {
  position: Vec3;
  cubeIndex: number | null;
  original: boolean;
}

export interface CPUTriangle {
  a: number;
  b: number;
  c: number;
}

const DELETED_TRIANGLE = -1;

export class CPUMesh {
  private positions: Float32Array;
  private cubeIndices: Int32Array;
  private originalFlags: Uint8Array;
  private triangleIndices: Int32Array;
  private vertexCountValue: number;
  private triangleCountValue: number;

  constructor(vertices: CPUVertex[], triangles: CPUTriangle[]) {
    const vertexCapacity = Math.max(1, vertices.length);
    const triangleCapacity = Math.max(1, triangles.length);
    this.positions = new Float32Array(vertexCapacity * 3);
    this.cubeIndices = new Int32Array(vertexCapacity);
    this.originalFlags = new Uint8Array(vertexCapacity);
    this.triangleIndices = new Int32Array(triangleCapacity * 3);
    this.triangleIndices.fill(DELETED_TRIANGLE);
    this.vertexCountValue = vertices.length;
    this.triangleCountValue = triangles.length;

    for (let i = 0; i < vertices.length; i++) {
      this.writeVertex(i, vertices[i]);
    }
    for (let i = 0; i < triangles.length; i++) {
      this.writeTriangle(i, triangles[i]);
    }
  }

  get vertexCount(): number {
    return this.vertexCountValue;
  }

  get triangleCount(): number {
    return this.triangleCountValue;
  }

  vertex(index: number): CPUVertex {
    this.assertVertexIndex(index);
    const base = index * 3;
    const cubeIndex = this.cubeIndices[index];
    return {
      position: [this.positions[base], this.positions[base + 1], this.positions[base + 2]],
      cubeIndex: cubeIndex < 0 ? null : cubeIndex,
      original: this.originalFlags[index] !== 0,
    };
  }

  vertexPosition(index: number): Vec3 {
    this.assertVertexIndex(index);
    const base = index * 3;
    return [this.positions[base], this.positions[base + 1], this.positions[base + 2]];
  }

  setVertexPosition(index: number, position: Vec3): void {
    this.assertVertexIndex(index);
    const base = index * 3;
    this.positions[base] = position[0];
    this.positions[base + 1] = position[1];
    this.positions[base + 2] = position[2];
  }

  vertexCubeIndex(index: number): number | null {
    this.assertVertexIndex(index);
    const cubeIndex = this.cubeIndices[index];
    return cubeIndex < 0 ? null : cubeIndex;
  }

  vertexOriginal(index: number): boolean {
    this.assertVertexIndex(index);
    return this.originalFlags[index] !== 0;
  }

  addVertex(vertex: CPUVertex): number {
    this.ensureVertexCapacity(this.vertexCountValue + 1);
    const index = this.vertexCountValue++;
    this.writeVertex(index, vertex);
    return index;
  }

  addTriangle(t: CPUTriangle): number {
    this.ensureTriangleCapacity(this.triangleCountValue + 1);
    const index = this.triangleCountValue++;
    this.writeTriangle(index, t);
    return index;
  }

  setTriangle(triIndex: number, triangle: CPUTriangle): void {
    this.assertTriangleIndex(triIndex);
    this.writeTriangle(triIndex, triangle);
  }

  removeTriangle(triIndex: number): void {
    this.assertTriangleIndex(triIndex);
    const base = triIndex * 3;
    this.triangleIndices[base] = DELETED_TRIANGLE;
    this.triangleIndices[base + 1] = DELETED_TRIANGLE;
    this.triangleIndices[base + 2] = DELETED_TRIANGLE;
  }

  triangle(triIndex: number): CPUTriangle | null {
    this.assertTriangleIndex(triIndex);
    const base = triIndex * 3;
    const a = this.triangleIndices[base];
    if (a === DELETED_TRIANGLE) return null;
    return {
      a,
      b: this.triangleIndices[base + 1],
      c: this.triangleIndices[base + 2],
    };
  }

  activeTriangleIndices(): number[] {
    const out: number[] = [];
    for (let i = 0; i < this.triangleCountValue; i++) {
      if (this.triangleIndices[i * 3] !== DELETED_TRIANGLE) out.push(i);
    }
    return out;
  }

  findTrianglesWithEdge(u: number, v: number): number[] {
    const tris: number[] = [];
    for (let i = 0; i < this.triangleCountValue; i++) {
      const base = i * 3;
      const a = this.triangleIndices[base];
      if (a === DELETED_TRIANGLE) continue;
      const b = this.triangleIndices[base + 1];
      const c = this.triangleIndices[base + 2];
      const hasU = a === u || b === u || c === u;
      const hasV = a === v || b === v || c === v;
      if (hasU && hasV) tris.push(i);
    }
    return tris;
  }

  compact(): IndexedMesh {
    const used = new Uint8Array(this.vertexCountValue);
    let usedCount = 0;
    let validTriangleCount = 0;

    for (let i = 0; i < this.triangleCountValue; i++) {
      const base = i * 3;
      const a = this.triangleIndices[base];
      if (a === DELETED_TRIANGLE) continue;
      const b = this.triangleIndices[base + 1];
      const c = this.triangleIndices[base + 2];
      if (a === b || b === c || c === a) continue;
      if (!this.triangleHasFiniteArea(a, b, c)) continue;
      if (used[a] === 0) { used[a] = 1; usedCount++; }
      if (used[b] === 0) { used[b] = 1; usedCount++; }
      if (used[c] === 0) { used[c] = 1; usedCount++; }
      validTriangleCount++;
    }

    const mapping = new Int32Array(this.vertexCountValue);
    mapping.fill(-1);
    const positions = new Float32Array(usedCount * 3);
    let nextVertex = 0;
    for (let oldId = 0; oldId < this.vertexCountValue; oldId++) {
      if (used[oldId] === 0) continue;
      mapping[oldId] = nextVertex++;
      const src = oldId * 3;
      const dst = mapping[oldId] * 3;
      positions[dst] = this.positions[src];
      positions[dst + 1] = this.positions[src + 1];
      positions[dst + 2] = this.positions[src + 2];
    }

    const indices = new Uint32Array(validTriangleCount * 3);
    let nextIndex = 0;
    for (let i = 0; i < this.triangleCountValue; i++) {
      const base = i * 3;
      const a = this.triangleIndices[base];
      if (a === DELETED_TRIANGLE) continue;
      const b = this.triangleIndices[base + 1];
      const c = this.triangleIndices[base + 2];
      if (a === b || b === c || c === a) continue;
      if (!this.triangleHasFiniteArea(a, b, c)) continue;
      indices[nextIndex++] = mapping[a];
      indices[nextIndex++] = mapping[b];
      indices[nextIndex++] = mapping[c];
    }

    return { positions, indices };
  }

  private writeVertex(index: number, vertex: CPUVertex): void {
    const base = index * 3;
    this.positions[base] = vertex.position[0];
    this.positions[base + 1] = vertex.position[1];
    this.positions[base + 2] = vertex.position[2];
    this.cubeIndices[index] = vertex.cubeIndex ?? -1;
    this.originalFlags[index] = vertex.original ? 1 : 0;
  }

  private writeTriangle(index: number, triangle: CPUTriangle): void {
    const base = index * 3;
    this.triangleIndices[base] = triangle.a;
    this.triangleIndices[base + 1] = triangle.b;
    this.triangleIndices[base + 2] = triangle.c;
  }

  private ensureVertexCapacity(requiredCount: number): void {
    if (requiredCount <= this.cubeIndices.length) return;
    const newCapacity = Math.max(requiredCount, this.cubeIndices.length * 2);
    const positions = new Float32Array(newCapacity * 3);
    positions.set(this.positions.subarray(0, this.vertexCountValue * 3));
    this.positions = positions;

    const cubeIndices = new Int32Array(newCapacity);
    cubeIndices.set(this.cubeIndices.subarray(0, this.vertexCountValue));
    cubeIndices.fill(-1, this.vertexCountValue);
    this.cubeIndices = cubeIndices;

    const originalFlags = new Uint8Array(newCapacity);
    originalFlags.set(this.originalFlags.subarray(0, this.vertexCountValue));
    this.originalFlags = originalFlags;
  }

  private ensureTriangleCapacity(requiredCount: number): void {
    if (requiredCount <= this.triangleIndices.length / 3) return;
    const oldCapacity = this.triangleIndices.length / 3;
    const newCapacity = Math.max(requiredCount, oldCapacity * 2);
    const triangleIndices = new Int32Array(newCapacity * 3);
    triangleIndices.set(this.triangleIndices.subarray(0, this.triangleCountValue * 3));
    triangleIndices.fill(DELETED_TRIANGLE, this.triangleCountValue * 3);
    this.triangleIndices = triangleIndices;
  }

  private triangleHasFiniteArea(a: number, b: number, c: number): boolean {
    const aBase = a * 3;
    const bBase = b * 3;
    const cBase = c * 3;
    const abx = this.positions[bBase] - this.positions[aBase];
    const aby = this.positions[bBase + 1] - this.positions[aBase + 1];
    const abz = this.positions[bBase + 2] - this.positions[aBase + 2];
    const acx = this.positions[cBase] - this.positions[aBase];
    const acy = this.positions[cBase + 1] - this.positions[aBase + 1];
    const acz = this.positions[cBase + 2] - this.positions[aBase + 2];
    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;
    const area2 = Math.hypot(nx, ny, nz);
    return area2 > 1e-12 && Number.isFinite(area2);
  }

  private assertVertexIndex(index: number): void {
    if (index < 0 || index >= this.vertexCountValue) {
      throw new Error(`vertex index out of range: ${index}`);
    }
  }

  private assertTriangleIndex(index: number): void {
    if (index < 0 || index >= this.triangleCountValue) {
      throw new Error(`triangle index out of range: ${index}`);
    }
  }
}
