import { add, cross, dot, normalizeSafe, scale, type Vec3 } from './vec3';

type Vec2 = [number, number];

class Complex {
  readonly re: number;
  readonly im: number;

  constructor(re: number, im = 0) {
    this.re = re;
    this.im = im;
  }

  add(other: Complex): Complex {
    return new Complex(this.re + other.re, this.im + other.im);
  }

  sub(other: Complex): Complex {
    return new Complex(this.re - other.re, this.im - other.im);
  }

  mul(other: Complex): Complex {
    return new Complex(
      this.re * other.re - this.im * other.im,
      this.re * other.im + this.im * other.re,
    );
  }

  div(other: Complex): Complex {
    const denom = other.re * other.re + other.im * other.im;
    return new Complex(
      (this.re * other.re + this.im * other.im) / denom,
      (this.im * other.re - this.re * other.im) / denom,
    );
  }

  abs(): number {
    return Math.hypot(this.re, this.im);
  }

  sqrt(): Complex {
    const r = this.abs();
    const real = Math.sqrt(Math.max(0, (r + this.re) / 2));
    const imag = Math.sign(this.im || 1) * Math.sqrt(Math.max(0, (r - this.re) / 2));
    return new Complex(real, imag);
  }

  pow(exp: number): Complex {
    const r = this.abs();
    const theta = Math.atan2(this.im, this.re);
    const rp = Math.pow(r, exp);
    const angle = theta * exp;
    return new Complex(rp * Math.cos(angle), rp * Math.sin(angle));
  }
}

class Matrix2 {
  readonly data: [number, number, number, number];

  constructor(data: [number, number, number, number]) {
    this.data = data;
  }

  det(): number {
    return this.data[0] * this.data[3] - this.data[1] * this.data[2];
  }

  mulColumn(c: Vec2): Vec2 {
    return [
      this.data[0] * c[0] + this.data[1] * c[1],
      this.data[2] * c[0] + this.data[3] * c[1],
    ];
  }

  transpose(): Matrix2 {
    return new Matrix2([this.data[0], this.data[2], this.data[1], this.data[3]]);
  }

  eigenvalues(): [Complex, Complex] {
    const a = new Complex(1);
    const b = new Complex(-(this.data[0] + this.data[3]));
    const c = new Complex(this.det());
    const disc = b.mul(b).sub(a.mul(c).mul(new Complex(4))).sqrt();
    return [
      b.mul(new Complex(-1)).sub(disc).div(a.mul(new Complex(2))),
      b.mul(new Complex(-1)).add(disc).div(a.mul(new Complex(2))),
    ];
  }

  symEigs(vals: [Complex, Complex]): [Vec2, Vec2] {
    const r1: Vec2 = [this.data[0] - vals[0].re, this.data[1]];
    const r2: Vec2 = [this.data[2], this.data[3] - vals[0].re];
    const n1 = vec2Norm(r1);
    const n2 = vec2Norm(r2);
    if (n1 === 0 && n2 === 0) {
      return [[1, 0], [0, 1]];
    }

    let secondEig: Vec2 = scale2(r1, 1 / n1);
    if (n2 > n1) {
      secondEig = scale2(r2, 1 / n2);
    }
    const firstEig: Vec2 = [-secondEig[1], secondEig[0]];
    return [firstEig, secondEig];
  }

  symEigDecomp(): { s: Matrix2; v: Matrix2 } {
    const eigVals = this.eigenvalues();
    if (eigVals[0].re < eigVals[1].re) {
      [eigVals[0], eigVals[1]] = [eigVals[1], eigVals[0]];
    }
    const [v1, v2] = this.symEigs(eigVals);
    return {
      v: new Matrix2([v1[0], v2[0], v1[1], v2[1]]),
      s: new Matrix2([eigVals[0].re, 0, 0, eigVals[1].re]),
    };
  }
}

class Matrix3 {
  readonly data: [number, number, number, number, number, number, number, number, number];

  constructor(data: [number, number, number, number, number, number, number, number, number]) {
    this.data = data;
  }

  det(): number {
    return this.data[0] * (this.data[4] * this.data[8] - this.data[5] * this.data[7]) -
      this.data[1] * (this.data[3] * this.data[8] - this.data[5] * this.data[6]) +
      this.data[2] * (this.data[3] * this.data[7] - this.data[4] * this.data[6]);
  }

  mul(m1: Matrix3): Matrix3 {
    const a = this.data;
    const b = m1.data;
    return new Matrix3([
      a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
      a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
      a[0] * b[2] + a[1] * b[5] + a[2] * b[8],

      a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
      a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
      a[3] * b[2] + a[4] * b[5] + a[5] * b[8],

      a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
      a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
      a[6] * b[2] + a[7] * b[5] + a[8] * b[8],
    ]);
  }

  mulColumn(c: Vec3): Vec3 {
    return [
      this.data[0] * c[0] + this.data[1] * c[1] + this.data[2] * c[2],
      this.data[3] * c[0] + this.data[4] * c[1] + this.data[5] * c[2],
      this.data[6] * c[0] + this.data[7] * c[1] + this.data[8] * c[2],
    ];
  }

  transpose(): Matrix3 {
    return new Matrix3([
      this.data[0], this.data[3], this.data[6],
      this.data[1], this.data[4], this.data[7],
      this.data[2], this.data[5], this.data[8],
    ]);
  }

  eigenvalues(): [Complex, Complex, Complex] {
    const m = this.data;
    const a00 = m[0];
    const a01 = 0.5 * (m[1] + m[3]);
    const a02 = 0.5 * (m[2] + m[6]);
    const a11 = m[4];
    const a12 = 0.5 * (m[5] + m[7]);
    const a22 = m[8];

    const p1 = a01 * a01 + a02 * a02 + a12 * a12;
    if (p1 <= 1e-20) {
      const values = [a00, a11, a22].sort((x, y) => y - x);
      return [new Complex(values[0]), new Complex(values[1]), new Complex(values[2])];
    }

    const q = (a00 + a11 + a22) / 3;
    const b00 = a00 - q;
    const b11 = a11 - q;
    const b22 = a22 - q;
    const p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2 * p1;
    const p = Math.sqrt(Math.max(0, p2 / 6));
    if (p <= 1e-20) {
      return [new Complex(q), new Complex(q), new Complex(q)];
    }

    const invP = 1 / p;
    const r = 0.5 * (
      (b00 * invP) * ((b11 * invP) * (b22 * invP) - (a12 * invP) * (a12 * invP)) -
      (a01 * invP) * ((a01 * invP) * (b22 * invP) - (a12 * invP) * (a02 * invP)) +
      (a02 * invP) * ((a01 * invP) * (a12 * invP) - (b11 * invP) * (a02 * invP))
    );
    const phi = Math.acos(clampScalar(r, -1, 1)) / 3;
    const twoPiOverThree = 2.0943951023931953;
    const eig0 = q + 2 * p * Math.cos(phi);
    const eig2 = q + 2 * p * Math.cos(phi + twoPiOverThree);
    const eig1 = 3 * q - eig0 - eig2;
    const values = [eig0, eig1, eig2].sort((x, y) => y - x);
    return [new Complex(values[0]), new Complex(values[1]), new Complex(values[2])];
  }

  symEigVector(val: number): Vec3 {
    const m = this.data;
    const row1: Vec3 = [m[0] - val, m[1], m[2]];
    const row2: Vec3 = [m[3], m[4] - val, m[5]];
    const row3: Vec3 = [m[6], m[7], m[8] - val];

    let bestVector: Vec3 = [1, 0, 0];
    let bestResult = 0;
    let triedAny = false;
    const tryVector = (c: Vec3) => {
      const norm = vec3Norm(c);
      if (norm === 0) {
        return;
      }
      const v = scale(c, 1 / norm);
      const out = Math.max(Math.max(Math.abs(dot(row1, v)), Math.abs(dot(row2, v))), Math.abs(dot(row3, v)));
      if (!triedAny || out < bestResult) {
        bestVector = v;
        bestResult = out;
        triedAny = true;
      }
    };
    const tryOrtho = (c: Vec3) => {
      if (c[0] === 0 && c[1] === 0 && c[2] === 0) {
        return;
      }
      const [v1] = orthoBasis3(c);
      tryVector(v1);
    };

    tryOrtho(row1);
    tryOrtho(row2);
    tryOrtho(row3);
    tryVector(cross(row1, row2));
    tryVector(cross(row1, row3));
    tryVector(cross(row2, row3));

    if (!triedAny) {
      return [1, 0, 0];
    }
    return bestVector;
  }

  symEigDecomp(): { s: Matrix3; v: Matrix3 } {
    const eigVals = this.eigenvalues().map((x) => x.re).sort((a, b) => b - a) as Vec3;
    const v0 = this.symEigVector(eigVals[0]);
    const [basis1, basis2] = orthoBasis3(v0);

    const out1 = this.mulColumn(basis1);
    const out2 = this.mulColumn(basis2);
    const mat2x2 = new Matrix2([
      dot(out1, basis1),
      0.5 * (dot(out1, basis2) + dot(out2, basis1)),
      0.5 * (dot(out1, basis2) + dot(out2, basis1)),
      dot(out2, basis2),
    ]);

    const { v: subV, s: subS } = mat2x2.symEigDecomp();
    const sv = subV.data;
    const subV1 = add(scale(basis1, sv[0]), scale(basis2, sv[2]));
    const subV2 = add(scale(basis1, sv[1]), scale(basis2, sv[3]));

    return {
      v: newMatrix3Columns(v0, subV1, subV2),
      s: new Matrix3([
        eigVals[0], 0, 0,
        0, subS.data[0], 0,
        0, 0, subS.data[3],
      ]),
    };
  }
}

export function leastSquaresReg3(a: Vec3[], b: number[], lambda: number, epsilon: number): Vec3 {
  const leftSide = new Array<number>(9).fill(0) as Matrix3['data'];
  let rightSide: Vec3 = [0, 0, 0];
  for (let row = 0; row < a.length; row++) {
    const v = a[row];
    rightSide = add(rightSide, scale(v, b[row]));
    let outIdx = 0;
    for (let j = 0; j < 3; j++) {
      for (let i = 0; i < 3; i++) {
        leftSide[outIdx] += v[i] * v[j];
        outIdx++;
      }
    }
  }

  leftSide[0] += lambda;
  leftSide[4] += lambda;
  leftSide[8] += lambda;

  const { s, v } = new Matrix3(leftSide).symEigDecomp();
  const sData = [...s.data] as Matrix3['data'];
  for (let i = 0; i < 3; i++) {
    const idx = i * 4;
    if (sData[idx] > epsilon) {
      sData[idx] = 1 / sData[idx];
    } else {
      sData[idx] = 0;
    }
  }
  return v.mul(new Matrix3(sData)).mul(v.transpose()).mulColumn(rightSide);
}

function newMatrix3Columns(c1: Vec3, c2: Vec3, c3: Vec3): Matrix3 {
  return new Matrix3([
    c1[0], c2[0], c3[0],
    c1[1], c2[1], c3[1],
    c1[2], c2[2], c3[2],
  ]);
}

function vec2Norm(v: Vec2): number {
  return Math.hypot(v[0], v[1]);
}

function vec3Norm(v: Vec3): number {
  return Math.hypot(v[0], v[1], v[2]);
}

function scale2(v: Vec2, s: number): Vec2 {
  return [v[0] * s, v[1] * s];
}

function clampScalar(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

function orthoBasis3(axis: Vec3): [Vec3, Vec3] {
  const helper: Vec3 = Math.abs(axis[0]) < 0.5 ? [1, 0, 0] : [0, 1, 0];
  const b1 = normalizeSafe(cross(axis, helper), [0, 0, 1]);
  const b2 = normalizeSafe(cross(axis, b1), [0, 1, 0]);
  return [b1, b2];
}
