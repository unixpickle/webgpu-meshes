export const qefWGSL = /* wgsl */`
struct Basis3 {
  b1: vec3<f32>,
  b2: vec3<f32>,
};

struct SymEig2 {
  values: vec2<f32>,
  v0: vec2<f32>,
  v1: vec2<f32>,
};

struct SymEig3 {
  values: vec3<f32>,
  v0: vec3<f32>,
  v1: vec3<f32>,
  v2: vec3<f32>,
};

fn qefNormalizeSafe(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
  let norm2 = dot(v, v);
  if (norm2 <= 1e-20) {
    return fallback;
  }
  return v * inverseSqrt(norm2);
}

fn qefOrthoBasis(axis: vec3<f32>) -> Basis3 {
  let helper = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(axis.x) < 0.5);
  let b1 = qefNormalizeSafe(cross(axis, helper), vec3<f32>(0.0, 0.0, 1.0));
  let b2 = qefNormalizeSafe(cross(axis, b1), vec3<f32>(0.0, 1.0, 0.0));
  return Basis3(b1, b2);
}

fn symDet3(a00: f32, a01: f32, a02: f32, a11: f32, a12: f32, a22: f32) -> f32 {
  return
    a00 * (a11 * a22 - a12 * a12) -
    a01 * (a01 * a22 - a12 * a02) +
    a02 * (a01 * a12 - a11 * a02);
}

fn sort3Desc(values: vec3<f32>) -> vec3<f32> {
  var x = values.x;
  var y = values.y;
  var z = values.z;
  if (x < y) { let t = x; x = y; y = t; }
  if (y < z) { let t = y; y = z; z = t; }
  if (x < y) { let t = x; x = y; y = t; }
  return vec3<f32>(x, y, z);
}

fn symEigenvalues3(a00: f32, a01: f32, a02: f32, a11: f32, a12: f32, a22: f32) -> vec3<f32> {
  let p1 = a01 * a01 + a02 * a02 + a12 * a12;
  if (p1 <= 1e-20) {
    return sort3Desc(vec3<f32>(a00, a11, a22));
  }

  let q = (a00 + a11 + a22) / 3.0;
  let b00 = a00 - q;
  let b11 = a11 - q;
  let b22 = a22 - q;
  let p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * p1;
  let p = sqrt(max(0.0, p2 / 6.0));
  if (p <= 1e-20) {
    return vec3<f32>(q, q, q);
  }

  let invP = 1.0 / p;
  let r = 0.5 * symDet3(
    b00 * invP,
    a01 * invP,
    a02 * invP,
    b11 * invP,
    a12 * invP,
    b22 * invP,
  );
  let phi = acos(clamp(r, -1.0, 1.0)) / 3.0;
  let twoPiOverThree = 2.0943951023931953;
  let eig0 = q + 2.0 * p * cos(phi);
  let eig2 = q + 2.0 * p * cos(phi + twoPiOverThree);
  let eig1 = 3.0 * q - eig0 - eig2;
  return sort3Desc(vec3<f32>(eig0, eig1, eig2));
}

fn symMulColumn3(a00: f32, a01: f32, a02: f32, a11: f32, a12: f32, a22: f32, v: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    a00 * v.x + a01 * v.y + a02 * v.z,
    a01 * v.x + a11 * v.y + a12 * v.z,
    a02 * v.x + a12 * v.y + a22 * v.z,
  );
}

fn updateBestNullCandidate(
  row1: vec3<f32>,
  row2: vec3<f32>,
  row3: vec3<f32>,
  candidate: vec3<f32>,
  bestVec_: ptr<function, vec3<f32>>,
  bestScore_: ptr<function, f32>,
  tried_: ptr<function, u32>,
) {
  let norm2 = dot(candidate, candidate);
  if (norm2 <= 1e-20) {
    return;
  }
  let v = candidate * inverseSqrt(norm2);
  let score = max(max(abs(dot(row1, v)), abs(dot(row2, v))), abs(dot(row3, v)));
  if ((*tried_ == 0u) || (score < *bestScore_)) {
    *bestVec_ = v;
    *bestScore_ = score;
    *tried_ = 1u;
  }
}

fn symEigVector3(a00: f32, a01: f32, a02: f32, a11: f32, a12: f32, a22: f32, eigenvalue: f32) -> vec3<f32> {
  let row1 = vec3<f32>(a00 - eigenvalue, a01, a02);
  let row2 = vec3<f32>(a01, a11 - eigenvalue, a12);
  let row3 = vec3<f32>(a02, a12, a22 - eigenvalue);

  var bestVec = vec3<f32>(1.0, 0.0, 0.0);
  var bestScore = 0.0;
  var tried = 0u;

  if (row1.x != 0.0 || row1.y != 0.0 || row1.z != 0.0) {
    let basis = qefOrthoBasis(row1);
    updateBestNullCandidate(row1, row2, row3, basis.b1, &bestVec, &bestScore, &tried);
  }
  if (row2.x != 0.0 || row2.y != 0.0 || row2.z != 0.0) {
    let basis = qefOrthoBasis(row2);
    updateBestNullCandidate(row1, row2, row3, basis.b1, &bestVec, &bestScore, &tried);
  }
  if (row3.x != 0.0 || row3.y != 0.0 || row3.z != 0.0) {
    let basis = qefOrthoBasis(row3);
    updateBestNullCandidate(row1, row2, row3, basis.b1, &bestVec, &bestScore, &tried);
  }

  updateBestNullCandidate(row1, row2, row3, cross(row1, row2), &bestVec, &bestScore, &tried);
  updateBestNullCandidate(row1, row2, row3, cross(row1, row3), &bestVec, &bestScore, &tried);
  updateBestNullCandidate(row1, row2, row3, cross(row2, row3), &bestVec, &bestScore, &tried);

  if (tried == 0u) {
    return vec3<f32>(1.0, 0.0, 0.0);
  }
  return bestVec;
}

fn symEigDecomp2(a: f32, b: f32, d: f32) -> SymEig2 {
  let trace = 0.5 * (a + d);
  let diff = 0.5 * (a - d);
  let root = sqrt(max(0.0, diff * diff + b * b));
  var lambda0 = trace + root;
  var lambda1 = trace - root;
  if (lambda0 < lambda1) {
    let tmp = lambda0;
    lambda0 = lambda1;
    lambda1 = tmp;
  }

  let r1 = vec2<f32>(a - lambda0, b);
  let r2 = vec2<f32>(b, d - lambda0);
  let n1 = length(r1);
  let n2 = length(r2);

  if (n1 == 0.0 && n2 == 0.0) {
    return SymEig2(vec2<f32>(lambda0, lambda1), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
  }

  var secondEig = r1;
  var secondNorm = n1;
  if (n2 > n1) {
    secondEig = r2;
    secondNorm = n2;
  }
  secondEig /= secondNorm;
  let firstEig = vec2<f32>(-secondEig.y, secondEig.x);

  return SymEig2(vec2<f32>(lambda0, lambda1), firstEig, secondEig);
}

fn symEigDecomp3(a00: f32, a01: f32, a02: f32, a11: f32, a12: f32, a22: f32) -> SymEig3 {
  let values = symEigenvalues3(a00, a01, a02, a11, a12, a22);
  let v0 = symEigVector3(a00, a01, a02, a11, a12, a22, values.x);
  let basis = qefOrthoBasis(v0);

  let out1 = symMulColumn3(a00, a01, a02, a11, a12, a22, basis.b1);
  let out2 = symMulColumn3(a00, a01, a02, a11, a12, a22, basis.b2);
  let sub00 = dot(out1, basis.b1);
  let sub01 = 0.5 * (dot(out1, basis.b2) + dot(out2, basis.b1));
  let sub11 = dot(out2, basis.b2);
  let sub = symEigDecomp2(sub00, sub01, sub11);

  let v1 = basis.b1 * sub.v0.x + basis.b2 * sub.v0.y;
  let v2 = basis.b1 * sub.v1.x + basis.b2 * sub.v1.y;
  return SymEig3(vec3<f32>(values.x, sub.values.x, sub.values.y), v0, v1, v2);
}

fn solveSymmetric3(a00: f32, a01: f32, a02: f32, a11: f32, a12: f32, a22: f32, b: vec3<f32>) -> vec3<f32> {
  let eig = symEigDecomp3(a00, a01, a02, a11, a12, a22);
  let proj0 = dot(eig.v0, b);
  let proj1 = dot(eig.v1, b);
  let proj2 = dot(eig.v2, b);

  var result = vec3<f32>(0.0);
  if (eig.values.x > params.singularValueEpsilon) {
    result += eig.v0 * (proj0 / eig.values.x);
  }
  if (eig.values.y > params.singularValueEpsilon) {
    result += eig.v1 * (proj1 / eig.values.y);
  }
  if (eig.values.z > params.singularValueEpsilon) {
    result += eig.v2 * (proj2 / eig.values.z);
  }
  return result;
}
`;
