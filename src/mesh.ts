import { cross, length, sub, type Vec3 } from './vec3';

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

export class CPUMesh {
  readonly vertices: CPUVertex[];
  readonly triangles: Array<CPUTriangle | null>;

  constructor(vertices: CPUVertex[], triangles: CPUTriangle[]) {
    this.vertices = vertices;
    this.triangles = triangles.slice();
  }

  addTriangle(t: CPUTriangle): number {
    this.triangles.push(t);
    return this.triangles.length - 1;
  }

  removeTriangle(triIndex: number): void {
    this.triangles[triIndex] = null;
  }

  triangle(triIndex: number): CPUTriangle | null {
    return this.triangles[triIndex];
  }

  activeTriangleIndices(): number[] {
    const out: number[] = [];
    for (let i = 0; i < this.triangles.length; i++) {
      if (this.triangles[i] !== null) out.push(i);
    }
    return out;
  }

  findTrianglesWithEdge(u: number, v: number): number[] {
    const tris: number[] = [];
    for (let i = 0; i < this.triangles.length; i++) {
      const tri = this.triangles[i];
      if (!tri) continue;
      const hasU = tri.a === u || tri.b === u || tri.c === u;
      const hasV = tri.a === v || tri.b === v || tri.c === v;
      if (hasU && hasV) tris.push(i);
    }
    return tris;
  }

  compact(): IndexedMesh {
    const used = new Set<number>();
    for (const tri of this.triangles) {
      if (!tri) continue;
      if (tri.a === tri.b || tri.b === tri.c || tri.c === tri.a) continue;
      const pa = this.vertices[tri.a].position;
      const pb = this.vertices[tri.b].position;
      const pc = this.vertices[tri.c].position;
      const area2 = length(cross(sub(pb, pa), sub(pc, pa)));
      if (!(area2 > 1e-12) || !Number.isFinite(area2)) continue;
      used.add(tri.a);
      used.add(tri.b);
      used.add(tri.c);
    }

    const mapping = new Map<number, number>();
    const positions: number[] = [];
    for (const oldId of used) {
      mapping.set(oldId, positions.length / 3);
      const p = this.vertices[oldId].position;
      positions.push(p[0], p[1], p[2]);
    }

    const indices: number[] = [];
    for (const tri of this.triangles) {
      if (!tri) continue;
      if (tri.a === tri.b || tri.b === tri.c || tri.c === tri.a) continue;
      const pa = this.vertices[tri.a].position;
      const pb = this.vertices[tri.b].position;
      const pc = this.vertices[tri.c].position;
      const area2 = length(cross(sub(pb, pa), sub(pc, pa)));
      if (!(area2 > 1e-12) || !Number.isFinite(area2)) continue;
      indices.push(mapping.get(tri.a)!, mapping.get(tri.b)!, mapping.get(tri.c)!);
    }

    return {
      positions: new Float32Array(positions),
      indices: new Uint32Array(indices),
    };
  }
}
