import { CPUMesh, type CPUTriangle, type CPUVertex, type IndexedMesh } from './mesh';
import { qefWGSL } from './qef_wgsl';
import {
  add,
  addScalar,
  clamp,
  clampVec3,
  cross,
  dot,
  normalizeSafe,
  orthoBasis,
  scale,
  sub,
  type Vec3,
} from './vec3';

const GPUBufferUsageAny: any = (globalThis as any).GPUBufferUsage;
const GPUMapModeAny: any = (globalThis as any).GPUMapMode;

export interface DualContourResult {
  /** Mesh emitted directly from the GPU passes, before CPU repair. */
  initial: IndexedMesh;
  /** Mesh after duplicate resolution and singular edge / singular vertex repair on the CPU. */
  repaired: IndexedMesh;
}

export type TriangleMode = 'max-min-area' | 'sharpest' | 'flattest';

export interface DualContouringWebGPUOptions {
  device: any;
  /**
   * WGSL source that defines:
   *   fn solidOccupancy(p: vec3<f32>) -> bool
   * The function is inlined into every compute shader.
   */
  solidWGSL: string;
  min: Vec3;
  max: Vec3;
  delta: number;
  noJitter?: boolean;
  clip?: boolean;
  repair?: boolean;
  cubeMargin?: number;
  repairEpsilon?: number;
  singularValueEpsilon?: number;
  l2Penalty?: number;
  triangleMode?: TriangleMode;
  bisectionSteps?: number;
  normalStep?: number;
  workgroupSize?: number;
  label?: string;
}

interface GridInfo {
  minCorner: Vec3;
  maxCorner: Vec3;
  delta: number;
  nx: number;
  ny: number;
  nz: number;
  cubeMarginWorld: number;
  repairEpsilonWorld: number;
}

interface GPUCubeVertex {
  active: boolean;
  position: Vec3;
}

interface GPUTriangle {
  a: number;
  b: number;
  c: number;
}

interface SingularEdgeGroup {
  u: number;
  v: number;
  pairs: Array<[number, number]>;
  inwardDirs: Vec3[];
}

interface SingularVertexGroup {
  vertex: number;
  components: number[][];
}

interface StageTiming {
  stage: string;
  ms: number;
}

export async function dualContourWebGPU(options: DualContouringWebGPUOptions): Promise<DualContourResult> {
  return dualContourWebGPUWithGPUHermite(options);
}

async function dualContourWebGPUWithGPUHermite(options: DualContouringWebGPUOptions): Promise<DualContourResult> {
  const timings: StageTiming[] = [];
  const startTime = performance.now();
  let stageStartTime = startTime;
  const markStage = (stage: string) => {
    const now = performance.now();
    timings.push({ stage, ms: now - stageStartTime });
    stageStartTime = now;
  };

  const config = normalizeOptions(options);
  const grid = createGridInfo(config.min, config.max, config.delta, config.noJitter, config.cubeMargin, config.repairEpsilon);
  markStage('normalize options + grid');

  const device = config.device;
  const workgroupSize = config.workgroupSize;

  const cornerCount = (grid.nx + 1) * (grid.ny + 1) * (grid.nz + 1);
  const xEdgeCount = grid.nx * (grid.ny + 1) * (grid.nz + 1);
  const yEdgeCount = (grid.nx + 1) * grid.ny * (grid.nz + 1);
  const zEdgeCount = (grid.nx + 1) * (grid.ny + 1) * grid.nz;
  const cubeCount = grid.nx * grid.ny * grid.nz;
  const triangleStride = 16;
  const deviceLimits = device.limits as Record<string, number | undefined> | undefined;
  const maxStorageBindingSize = deviceLimits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const maxBufferSize = deviceLimits?.maxBufferSize ?? Number.POSITIVE_INFINITY;

  const cornerBuffer = device.createBuffer({
    label: `${config.label}-corners`,
    size: cornerCount * 4,
    usage: GPUBufferUsageAny.STORAGE | GPUBufferUsageAny.COPY_SRC,
  });

  const hermiteStride = 48;

  const xEdgeBuffer = device.createBuffer({
    label: `${config.label}-x-edges`,
    size: xEdgeCount * hermiteStride,
    usage: GPUBufferUsageAny.STORAGE | GPUBufferUsageAny.COPY_SRC,
  });
  const yEdgeBuffer = device.createBuffer({
    label: `${config.label}-y-edges`,
    size: yEdgeCount * hermiteStride,
    usage: GPUBufferUsageAny.STORAGE | GPUBufferUsageAny.COPY_SRC,
  });
  const zEdgeBuffer = device.createBuffer({
    label: `${config.label}-z-edges`,
    size: zEdgeCount * hermiteStride,
    usage: GPUBufferUsageAny.STORAGE | GPUBufferUsageAny.COPY_SRC,
  });

  const cubeStride = 32;
  const cubeBuffer = device.createBuffer({
    label: `${config.label}-cubes`,
    size: cubeCount * cubeStride,
    usage: GPUBufferUsageAny.STORAGE | GPUBufferUsageAny.COPY_SRC,
  });
  const xTriangleCountBuffer = device.createBuffer({
    label: `${config.label}-x-triangle-counts`,
    size: xEdgeCount * 4,
    usage: GPUBufferUsageAny.STORAGE | GPUBufferUsageAny.COPY_SRC,
  });
  const yTriangleCountBuffer = device.createBuffer({
    label: `${config.label}-y-triangle-counts`,
    size: yEdgeCount * 4,
    usage: GPUBufferUsageAny.STORAGE | GPUBufferUsageAny.COPY_SRC,
  });
  const zTriangleCountBuffer = device.createBuffer({
    label: `${config.label}-z-triangle-counts`,
    size: zEdgeCount * 4,
    usage: GPUBufferUsageAny.STORAGE | GPUBufferUsageAny.COPY_SRC,
  });
  markStage('create GPU buffers');

  const uniformBuffer = device.createBuffer({
    label: `${config.label}-params`,
    size: 64,
    usage: GPUBufferUsageAny.UNIFORM | GPUBufferUsageAny.COPY_DST,
  });
  device.queue.writeBuffer(uniformBuffer, 0, packUniforms(grid, config));

  const shaders = buildShaderBundle(config.solidWGSL, workgroupSize);
  const cornerPipeline = device.createComputePipeline({
    label: `${config.label}-corner-pipeline`,
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: shaders.corner }), entryPoint: 'main' },
  });
  const xEdgePipeline = device.createComputePipeline({
    label: `${config.label}-x-edge-pipeline`,
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: shaders.edgeX }), entryPoint: 'main' },
  });
  const yEdgePipeline = device.createComputePipeline({
    label: `${config.label}-y-edge-pipeline`,
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: shaders.edgeY }), entryPoint: 'main' },
  });
  const zEdgePipeline = device.createComputePipeline({
    label: `${config.label}-z-edge-pipeline`,
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: shaders.edgeZ }), entryPoint: 'main' },
  });
  const cubePipeline = device.createComputePipeline({
    label: `${config.label}-cube-pipeline`,
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: shaders.cube }), entryPoint: 'main' },
  });
  const countXPipeline = device.createComputePipeline({
    label: `${config.label}-count-x-pipeline`,
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: shaders.countX }), entryPoint: 'main' },
  });
  const countYPipeline = device.createComputePipeline({
    label: `${config.label}-count-y-pipeline`,
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: shaders.countY }), entryPoint: 'main' },
  });
  const countZPipeline = device.createComputePipeline({
    label: `${config.label}-count-z-pipeline`,
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: shaders.countZ }), entryPoint: 'main' },
  });
  const emitXPipeline = device.createComputePipeline({
    label: `${config.label}-emit-x-pipeline`,
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: shaders.emitX }), entryPoint: 'main' },
  });
  const emitYPipeline = device.createComputePipeline({
    label: `${config.label}-emit-y-pipeline`,
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: shaders.emitY }), entryPoint: 'main' },
  });
  const emitZPipeline = device.createComputePipeline({
    label: `${config.label}-emit-z-pipeline`,
    layout: 'auto',
    compute: { module: device.createShaderModule({ code: shaders.emitZ }), entryPoint: 'main' },
  });
  markStage('build shaders + pipelines');

  const cornerBindGroup = device.createBindGroup({
    layout: cornerPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: cornerBuffer } },
    ],
  });
  const xEdgeBindGroup = device.createBindGroup({
    layout: xEdgePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: cornerBuffer } },
      { binding: 2, resource: { buffer: xEdgeBuffer } },
    ],
  });
  const yEdgeBindGroup = device.createBindGroup({
    layout: yEdgePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: cornerBuffer } },
      { binding: 2, resource: { buffer: yEdgeBuffer } },
    ],
  });
  const zEdgeBindGroup = device.createBindGroup({
    layout: zEdgePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: cornerBuffer } },
      { binding: 2, resource: { buffer: zEdgeBuffer } },
    ],
  });
  const cubeBindGroup = device.createBindGroup({
    layout: cubePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: cornerBuffer } },
      { binding: 2, resource: { buffer: xEdgeBuffer } },
      { binding: 3, resource: { buffer: yEdgeBuffer } },
      { binding: 4, resource: { buffer: zEdgeBuffer } },
      { binding: 5, resource: { buffer: cubeBuffer } },
    ],
  });
  const countXBindGroup = device.createBindGroup({
    layout: countXPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 2, resource: { buffer: xEdgeBuffer } },
      { binding: 3, resource: { buffer: cubeBuffer } },
      { binding: 4, resource: { buffer: xTriangleCountBuffer } },
    ],
  });
  const countYBindGroup = device.createBindGroup({
    layout: countYPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 2, resource: { buffer: yEdgeBuffer } },
      { binding: 3, resource: { buffer: cubeBuffer } },
      { binding: 4, resource: { buffer: yTriangleCountBuffer } },
    ],
  });
  const countZBindGroup = device.createBindGroup({
    layout: countZPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 2, resource: { buffer: zEdgeBuffer } },
      { binding: 3, resource: { buffer: cubeBuffer } },
      { binding: 4, resource: { buffer: zTriangleCountBuffer } },
    ],
  });
  markStage('create bind groups');

  const cubeReadback = device.createBuffer({
    label: `${config.label}-cube-readback`,
    size: cubeCount * cubeStride,
    usage: GPUBufferUsageAny.COPY_DST | GPUBufferUsageAny.MAP_READ,
  });
  const xTriangleCountReadback = device.createBuffer({
    label: `${config.label}-x-triangle-counts-readback`,
    size: xEdgeCount * 4,
    usage: GPUBufferUsageAny.COPY_DST | GPUBufferUsageAny.MAP_READ,
  });
  const yTriangleCountReadback = device.createBuffer({
    label: `${config.label}-y-triangle-counts-readback`,
    size: yEdgeCount * 4,
    usage: GPUBufferUsageAny.COPY_DST | GPUBufferUsageAny.MAP_READ,
  });
  const zTriangleCountReadback = device.createBuffer({
    label: `${config.label}-z-triangle-counts-readback`,
    size: zEdgeCount * 4,
    usage: GPUBufferUsageAny.COPY_DST | GPUBufferUsageAny.MAP_READ,
  });
  markStage('create readback buffers');

  const encoder = device.createCommandEncoder({ label: `${config.label}-count-encoder` });
  {
    const pass = encoder.beginComputePass({ label: `${config.label}-count-pass` });

    dispatch1D(pass, cornerPipeline, cornerBindGroup, cornerCount, workgroupSize);
    dispatch1D(pass, xEdgePipeline, xEdgeBindGroup, xEdgeCount, workgroupSize);
    dispatch1D(pass, yEdgePipeline, yEdgeBindGroup, yEdgeCount, workgroupSize);
    dispatch1D(pass, zEdgePipeline, zEdgeBindGroup, zEdgeCount, workgroupSize);
    dispatch1D(pass, cubePipeline, cubeBindGroup, cubeCount, workgroupSize);
    dispatch1D(pass, countXPipeline, countXBindGroup, xEdgeCount, workgroupSize);
    dispatch1D(pass, countYPipeline, countYBindGroup, yEdgeCount, workgroupSize);
    dispatch1D(pass, countZPipeline, countZBindGroup, zEdgeCount, workgroupSize);

    pass.end();
  }
  markStage('encode count pass');

  encoder.copyBufferToBuffer(xTriangleCountBuffer, 0, xTriangleCountReadback, 0, xEdgeCount * 4);
  encoder.copyBufferToBuffer(yTriangleCountBuffer, 0, yTriangleCountReadback, 0, yEdgeCount * 4);
  encoder.copyBufferToBuffer(zTriangleCountBuffer, 0, zTriangleCountReadback, 0, zEdgeCount * 4);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  markStage('submit count pass + GPU execution');

  const [xCountBytes, yCountBytes, zCountBytes] = await Promise.all([
    readBuffer(xTriangleCountReadback),
    readBuffer(yTriangleCountReadback),
    readBuffer(zTriangleCountReadback),
  ]);
  markStage('GPU count readback + map');

  const xTriangleCounts = new Uint32Array(xCountBytes);
  const yTriangleCounts = new Uint32Array(yCountBytes);
  const zTriangleCounts = new Uint32Array(zCountBytes);
  const xTriangleOffsets = exclusiveScanCounts(xTriangleCounts);
  const yTriangleOffsets = exclusiveScanCounts(yTriangleCounts, xTriangleOffsets.totalCount);
  const zTriangleOffsets = exclusiveScanCounts(zTriangleCounts, yTriangleOffsets.totalCount);
  const triangleCount = zTriangleOffsets.totalCount;
  const triangleBufferSize = Math.max(16, triangleCount * triangleStride);
  if (triangleBufferSize > maxStorageBindingSize) {
    throw new Error(
      `Triangle output buffer requires ${triangleBufferSize} bytes, which exceeds the device storage-buffer binding limit of ${maxStorageBindingSize} bytes. ` +
      `Use a larger delta / smaller bounds, or request a device with a higher maxStorageBufferBindingSize.`
    );
  }
  if (triangleBufferSize > maxBufferSize) {
    throw new Error(
      `Triangle output buffer requires ${triangleBufferSize} bytes, which exceeds the device maxBufferSize of ${maxBufferSize} bytes. ` +
      `Use a larger delta or smaller bounds.`
    );
  }
  markStage('decode counts + prefix sums');

  const xTriangleOffsetBuffer = device.createBuffer({
    label: `${config.label}-x-triangle-offsets`,
    size: xEdgeCount * 4,
    usage: GPUBufferUsageAny.STORAGE | GPUBufferUsageAny.COPY_DST,
  });
  const yTriangleOffsetBuffer = device.createBuffer({
    label: `${config.label}-y-triangle-offsets`,
    size: yEdgeCount * 4,
    usage: GPUBufferUsageAny.STORAGE | GPUBufferUsageAny.COPY_DST,
  });
  const zTriangleOffsetBuffer = device.createBuffer({
    label: `${config.label}-z-triangle-offsets`,
    size: zEdgeCount * 4,
    usage: GPUBufferUsageAny.STORAGE | GPUBufferUsageAny.COPY_DST,
  });
  const triangleBuffer = device.createBuffer({
    label: `${config.label}-triangles`,
    size: triangleBufferSize,
    usage: GPUBufferUsageAny.STORAGE | GPUBufferUsageAny.COPY_SRC,
  });
  const triangleReadback = device.createBuffer({
    label: `${config.label}-triangle-readback`,
    size: triangleBufferSize,
    usage: GPUBufferUsageAny.COPY_DST | GPUBufferUsageAny.MAP_READ,
  });
  markStage('create exact emit buffers');

  device.queue.writeBuffer(xTriangleOffsetBuffer, 0, xTriangleOffsets.offsets);
  device.queue.writeBuffer(yTriangleOffsetBuffer, 0, yTriangleOffsets.offsets);
  device.queue.writeBuffer(zTriangleOffsetBuffer, 0, zTriangleOffsets.offsets);
  markStage('upload triangle offsets');

  const emitXBindGroup = device.createBindGroup({
    layout: emitXPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: cornerBuffer } },
      { binding: 2, resource: { buffer: xEdgeBuffer } },
      { binding: 3, resource: { buffer: cubeBuffer } },
      { binding: 4, resource: { buffer: xTriangleOffsetBuffer } },
      { binding: 5, resource: { buffer: triangleBuffer } },
    ],
  });
  const emitYBindGroup = device.createBindGroup({
    layout: emitYPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: cornerBuffer } },
      { binding: 2, resource: { buffer: yEdgeBuffer } },
      { binding: 3, resource: { buffer: cubeBuffer } },
      { binding: 4, resource: { buffer: yTriangleOffsetBuffer } },
      { binding: 5, resource: { buffer: triangleBuffer } },
    ],
  });
  const emitZBindGroup = device.createBindGroup({
    layout: emitZPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: cornerBuffer } },
      { binding: 2, resource: { buffer: zEdgeBuffer } },
      { binding: 3, resource: { buffer: cubeBuffer } },
      { binding: 4, resource: { buffer: zTriangleOffsetBuffer } },
      { binding: 5, resource: { buffer: triangleBuffer } },
    ],
  });
  markStage('create emit bind groups');

  const emitEncoder = device.createCommandEncoder({ label: `${config.label}-emit-encoder` });
  {
    const pass = emitEncoder.beginComputePass({ label: `${config.label}-emit-pass` });
    dispatch1D(pass, emitXPipeline, emitXBindGroup, xEdgeCount, workgroupSize);
    dispatch1D(pass, emitYPipeline, emitYBindGroup, yEdgeCount, workgroupSize);
    dispatch1D(pass, emitZPipeline, emitZBindGroup, zEdgeCount, workgroupSize);
    pass.end();
  }
  markStage('encode emit pass');

  emitEncoder.copyBufferToBuffer(cubeBuffer, 0, cubeReadback, 0, cubeCount * cubeStride);
  emitEncoder.copyBufferToBuffer(triangleBuffer, 0, triangleReadback, 0, triangleBufferSize);
  device.queue.submit([emitEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  markStage('submit emit pass + GPU execution');

  const [cubeBytes, triangleBytes] = await Promise.all([
    readBuffer(cubeReadback),
    readBuffer(triangleReadback),
  ]);
  markStage('GPU emit readback + map');

  const cubes = decodeCubeVertices(cubeBytes);
  const triangles = decodeTriangles(triangleBytes, triangleCount);
  markStage('decode emitted mesh');

  const cpuMesh = buildMeshFromGPUOutput(cubes, triangles);
  markStage('CPU mesh assembly');
  const initial = cpuMesh.compact();
  markStage('compact initial mesh');

  if (config.repair) {
    resolveDuplicateOriginalVertices(cpuMesh, grid);
    const edgeRepairEpsilon = grid.repairEpsilonWorld * 0.49;
    repairSingularEdges(cpuMesh, grid, edgeRepairEpsilon, config.clip);
    repairSingularVertices(cpuMesh, grid, edgeRepairEpsilon, config.clip);
    markStage('CPU repair');
  } else {
    markStage('CPU repair (skipped)');
  }

  const repaired = cpuMesh.compact();
  markStage('compact repaired mesh');
  logDualContourTimings(config.label, timings, performance.now() - startTime);
  return { initial, repaired };
}

function normalizeOptions(options: DualContouringWebGPUOptions): Required<Omit<DualContouringWebGPUOptions, 'device' | 'solidWGSL' | 'min' | 'max' | 'delta'>> & Pick<DualContouringWebGPUOptions, 'device' | 'solidWGSL' | 'min' | 'max' | 'delta'> {
  return {
    device: options.device,
    solidWGSL: options.solidWGSL,
    min: options.min,
    max: options.max,
    delta: options.delta,
    noJitter: options.noJitter ?? false,
    clip: options.clip ?? true,
    repair: options.repair ?? true,
    cubeMargin: options.cubeMargin ?? 0.001,
    repairEpsilon: options.repairEpsilon ?? 0.01,
    singularValueEpsilon: options.singularValueEpsilon ?? 0.1,
    l2Penalty: options.l2Penalty ?? 0,
    triangleMode: options.triangleMode ?? 'max-min-area',
    bisectionSteps: options.bisectionSteps ?? 32,
    normalStep: options.normalStep ?? 1e-4,
    workgroupSize: options.workgroupSize ?? 256,
    label: options.label ?? 'dual-contour-webgpu',
  };
}

function createGridInfo(min: Vec3, max: Vec3, delta: number, noJitter: boolean, cubeMargin: number, repairEpsilon: number): GridInfo {
  const jitter = noJitter ? 0 : delta * 0.012923982;
  const minExpanded: Vec3 = [min[0] - delta + jitter, min[1] - delta + jitter, min[2] - delta + jitter];
  const maxExpanded: Vec3 = [max[0] + delta + jitter, max[1] + delta + jitter, max[2] + delta + jitter];
  const countX = Math.round((maxExpanded[0] - minExpanded[0]) / delta) + 1;
  const countY = Math.round((maxExpanded[1] - minExpanded[1]) / delta) + 1;
  const countZ = Math.round((maxExpanded[2] - minExpanded[2]) / delta) + 1;
  if (countX < 2 || countY < 2 || countZ < 2) {
    throw new Error('dualContourWebGPU(): invalid bounds; expanded grid must contain at least one cube along each axis.');
  }
  return {
    minCorner: minExpanded,
    maxCorner: [minExpanded[0] + (countX - 1) * delta, minExpanded[1] + (countY - 1) * delta, minExpanded[2] + (countZ - 1) * delta],
    delta,
    nx: countX - 1,
    ny: countY - 1,
    nz: countZ - 1,
    cubeMarginWorld: cubeMargin * delta,
    repairEpsilonWorld: repairEpsilon * delta,
  };
}

function packUniforms(grid: GridInfo, options: ReturnType<typeof normalizeOptions>): ArrayBuffer {
  const buffer = new ArrayBuffer(64);
  const f32 = new Float32Array(buffer);
  const u32 = new Uint32Array(buffer);

  f32[0] = grid.minCorner[0];
  f32[1] = grid.minCorner[1];
  f32[2] = grid.minCorner[2];
  f32[3] = grid.delta;
  f32[4] = grid.cubeMarginWorld;
  f32[5] = options.l2Penalty;
  f32[6] = options.normalStep;
  f32[7] = options.singularValueEpsilon;
  u32[8] = grid.nx;
  u32[9] = grid.ny;
  u32[10] = grid.nz;
  u32[11] = options.bisectionSteps;
  u32[12] = options.clip ? 1 : 0;
  u32[13] = triangleModeToInt(options.triangleMode);
  u32[14] = 0;
  u32[15] = 0;
  return buffer;
}

function triangleModeToInt(mode: TriangleMode): number {
  switch (mode) {
    case 'max-min-area': return 0;
    case 'sharpest': return 1;
    case 'flattest': return 2;
  }
}

function dispatch1D(pass: any, pipeline: any, bindGroup: any, count: number, workgroupSize: number): void {
  if (count <= 0) return;
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(count / workgroupSize));
}

async function readBuffer(buffer: any): Promise<ArrayBuffer> {
  await buffer.mapAsync(GPUMapModeAny.READ);
  const src = buffer.getMappedRange();
  const out = src.slice(0);
  buffer.unmap();
  return out;
}

function buildMeshFromGPUOutput(cubes: GPUCubeVertex[], triangles: GPUTriangle[]): CPUMesh {
  const vertices: CPUVertex[] = cubes.map((cube, cubeIndex) => (
    cube.active
      ? { position: cube.position, cubeIndex, original: true }
      : { position: [0, 0, 0], cubeIndex: null, original: false }
  ));

  return new CPUMesh(vertices, triangles as CPUTriangle[]);
}

function decodeCubeVertices(bytes: ArrayBuffer): GPUCubeVertex[] {
  const view = new DataView(bytes);
  const result: GPUCubeVertex[] = new Array(bytes.byteLength / 32);
  for (let i = 0; i < result.length; i++) {
    const base = i * 32;
    result[i] = {
      active: view.getUint32(base + 0, true) !== 0,
      position: [
        view.getFloat32(base + 16, true),
        view.getFloat32(base + 20, true),
        view.getFloat32(base + 24, true),
      ],
    };
  }
  return result;
}

function decodeTriangles(bytes: ArrayBuffer, count: number): GPUTriangle[] {
  const view = new DataView(bytes);
  const safeCount = Math.min(count, Math.floor(bytes.byteLength / 16));
  const result: GPUTriangle[] = new Array(safeCount);
  for (let i = 0; i < safeCount; i++) {
    const base = i * 16;
    result[i] = {
      a: view.getUint32(base + 0, true),
      b: view.getUint32(base + 4, true),
      c: view.getUint32(base + 8, true),
    };
  }
  return result;
}

function exclusiveScanCounts(counts: Uint32Array, baseOffset = 0): { offsets: Uint32Array; totalCount: number } {
  const offsets = new Uint32Array(counts.length);
  let running = baseOffset;
  for (let i = 0; i < counts.length; i++) {
    offsets[i] = running;
    running += counts[i];
  }
  return { offsets, totalCount: running };
}

function resolveDuplicateOriginalVertices(mesh: CPUMesh, grid: GridInfo): void {
  const groups = new Map<string, number[]>();
  for (let i = 0; i < mesh.vertexCount; i++) {
    if (!mesh.vertexOriginal(i)) continue;
    const cubeIndex = mesh.vertexCubeIndex(i);
    if (cubeIndex === null) continue;
    const key = vec3ExactKey(mesh.vertexPosition(i));
    let ids = groups.get(key);
    if (!ids) {
      ids = [];
      groups.set(key, ids);
    }
    ids.push(i);
  }

  for (const ids of groups.values()) {
    if (ids.length <= 1) continue;
    for (let k = 0; k < ids.length; k++) {
      const vid = ids[k];
      const cubeIndex = mesh.vertexCubeIndex(vid);
      if (cubeIndex === null) continue;
      const [min, max] = cubeBoundsFromIndex(cubeIndex, grid);
      mesh.setVertexPosition(vid, clampVec3(mesh.vertexPosition(vid), addScalar(min, grid.cubeMarginWorld), addScalar(max, -grid.cubeMarginWorld)));
    }

    const remaining = new Map<string, number[]>();
    for (const vid of ids) {
      const key = vec3ExactKey(mesh.vertexPosition(vid));
      let arr = remaining.get(key);
      if (!arr) {
        arr = [];
        remaining.set(key, arr);
      }
      arr.push(vid);
    }

    for (const same of remaining.values()) {
      if (same.length <= 1) continue;
      for (let i = 1; i < same.length; i++) {
        const vid = same[i];
        const cubeIndex = mesh.vertexCubeIndex(vid);
        if (cubeIndex === null) continue;
        const [min, max] = cubeBoundsFromIndex(cubeIndex, grid);
        const center = scale(add(min, max), 0.5);
        const position = mesh.vertexPosition(vid);
        const dir = normalizeSafe(sub(center, position), hashDirection(cubeIndex));
        const maxStep = Math.max(grid.cubeMarginWorld * 0.5, grid.repairEpsilonWorld * 0.25);
        mesh.setVertexPosition(vid, clampVec3(add(position, scale(dir, maxStep)), addScalar(min, grid.cubeMarginWorld), addScalar(max, -grid.cubeMarginWorld)));
      }
    }
  }
}

function repairSingularEdges(mesh: CPUMesh, grid: GridInfo, epsilon: number, clip: boolean): void {
  const groups = singularEdgeGroups(mesh);
  if (groups.length === 0) return;

  if (clip) {
    const toClamp = new Set<number>();
    for (const group of groups) {
      for (const [ta, tb] of group.pairs) {
        const triA = mesh.triangle(ta);
        const triB = mesh.triangle(tb);
        if (triA) {
          toClamp.add(triA.a);
          toClamp.add(triA.b);
          toClamp.add(triA.c);
        }
        if (triB) {
          toClamp.add(triB.a);
          toClamp.add(triB.b);
          toClamp.add(triB.c);
        }
      }
    }
    for (const vid of toClamp) clampOriginalVertex(mesh, vid, grid, epsilon);
  }

  for (const group of groups) {
    const refreshed = recomputeSingularEdgeGroup(mesh, group.u, group.v);
    if (!refreshed || refreshed.pairs.length === 0) continue;

    const edgeMid = scale(add(mesh.vertexPosition(group.u), mesh.vertexPosition(group.v)), 0.5);
    for (let i = 0; i < refreshed.pairs.length; i++) {
      const inwardDir = refreshed.inwardDirs[i];
      const newVertexId = mesh.addVertex({
        position: add(edgeMid, scale(inwardDir, epsilon)),
        cubeIndex: null,
        original: false,
      });

      const pair = refreshed.pairs[i];
      for (const triId of pair) {
        const tri = mesh.triangle(triId);
        if (!tri) continue;
        const other = thirdVertexOfTriangle(tri, refreshed.u, refreshed.v);
        if (other < 0) continue;
        let t1: CPUTriangle = { a: other, b: refreshed.u, c: newVertexId };
        let t2: CPUTriangle = { a: other, b: newVertexId, c: refreshed.v };
        const originalOrientation = segmentOrientation(tri, other, refreshed.u);
        if (segmentOrientation(t1, other, refreshed.u) !== originalOrientation) {
          t1 = { a: refreshed.u, b: other, c: newVertexId };
          t2 = { a: newVertexId, b: other, c: refreshed.v };
        }
        mesh.removeTriangle(triId);
        mesh.addTriangle(t1);
        mesh.addTriangle(t2);
      }
    }
  }
}

function repairSingularVertices(mesh: CPUMesh, grid: GridInfo, epsilon: number, clip: boolean): void {
  const groups = singularVertexGroups(mesh);
  if (groups.length === 0) return;

  if (clip) {
    const toClamp = new Set<number>();
    for (const group of groups) {
      for (const component of group.components) {
        for (const triId of component) {
          const tri = mesh.triangle(triId);
          if (!tri) continue;
          toClamp.add(tri.a);
          toClamp.add(tri.b);
          toClamp.add(tri.c);
        }
      }
    }
    for (const vid of toClamp) clampOriginalVertex(mesh, vid, grid, epsilon);
  }

  for (const group of groups) {
    const v = group.vertex;
    const center = mesh.vertexPosition(v);
    for (const component of group.components) {
      let dir: Vec3 = [0, 0, 0];
      for (const triId of component) {
        const tri = mesh.triangle(triId);
        if (!tri) continue;
        const [o1, o2] = otherSegmentOfTriangle(tri, v);
        const p1 = mesh.vertexPosition(o1);
        const p2 = mesh.vertexPosition(o2);
        const dotv = clamp(dot(normalizeSafe(sub(p1, center), [1, 0, 0]), normalizeSafe(sub(p2, center), [0, 1, 0])), -1.0, 1.0);
        const theta = Math.acos(dotv);
        dir = add(dir, scale(triangleNormal(mesh, tri), -theta));
      }
      const newVertexId = mesh.addVertex({
        position: add(center, scale(normalizeSafe(dir, hashDirection(v)), epsilon)),
        cubeIndex: null,
        original: false,
      });
      for (const triId of component) {
        const tri = mesh.triangle(triId);
        if (!tri) continue;
        if (tri.a === v) tri.a = newVertexId;
        else if (tri.b === v) tri.b = newVertexId;
        else if (tri.c === v) tri.c = newVertexId;
        mesh.setTriangle(triId, tri);
      }
    }
  }
}

function singularEdgeGroups(mesh: CPUMesh): SingularEdgeGroup[] {
  const edgeMap = new Map<string, { u: number; v: number; triangles: number[] }>();
  for (let i = 0; i < mesh.triangleCount; i++) {
    const tri = mesh.triangle(i);
    if (!tri) continue;
    appendEdgeRecord(edgeMap, tri.a, tri.b, i);
    appendEdgeRecord(edgeMap, tri.b, tri.c, i);
    appendEdgeRecord(edgeMap, tri.c, tri.a, i);
  }

  const groups: SingularEdgeGroup[] = [];
  for (const entry of edgeMap.values()) {
    if (entry.triangles.length > 2) {
      const group = buildSingularEdgeGroup(mesh, entry.u, entry.v, entry.triangles);
      if (group) groups.push(group);
    }
  }
  return groups;
}

function recomputeSingularEdgeGroup(mesh: CPUMesh, u: number, v: number): SingularEdgeGroup | null {
  const triangles = mesh.findTrianglesWithEdge(u, v);
  if (triangles.length <= 2 || triangles.length % 2 !== 0) return null;
  return buildSingularEdgeGroup(mesh, u, v, triangles);
}

function buildSingularEdgeGroup(mesh: CPUMesh, u: number, v: number, triIndices: number[]): SingularEdgeGroup | null {
  if (triIndices.length <= 2 || triIndices.length % 2 !== 0) return null;

  const p0 = mesh.vertexPosition(u);
  const p1 = mesh.vertexPosition(v);
  const axis = normalizeSafe(sub(p0, p1), [1, 0, 0]);
  const [b1, b2] = orthoBasis(axis);
  const midpoint = scale(add(p0, p1), 0.5);

  const sortable = triIndices.map((triId) => {
    const tri = mesh.triangle(triId)!;
    const other = thirdVertexOfTriangle(tri, u, v);
    const triVec = normalizeSafe(sub(mesh.vertexPosition(other), midpoint), b1);
    const x = dot(b1, triVec);
    const y = dot(b2, triVec);
    const theta = Math.atan2(y, x);
    const normal = triangleNormal(mesh, tri);
    const nx = dot(b1, normal);
    const ny = dot(b2, normal);
    const normalDir = (nx * y - ny * x) > 0;
    return { triId, theta, normalDir };
  }).sort((a, b) => a.theta - b.theta);

  if (sortable.length > 2 && sortable[0].normalDir) {
    const first = sortable.shift()!;
    first.theta += Math.PI * 2;
    sortable.push(first);
  }

  for (let i = 0; i < sortable.length; i += 2) {
    const triA = mesh.triangle(sortable[i].triId)!;
    for (let j = i + 1; j < sortable.length; j++) {
      const triB = mesh.triangle(sortable[j].triId)!;
      if (segmentOrientation(triA, u, v) !== segmentOrientation(triB, u, v)) {
        if (j !== i + 1) {
          const tmp = sortable[i + 1];
          sortable[i + 1] = sortable[j];
          sortable[j] = tmp;
        }
        break;
      }
    }
  }

  const pairs: Array<[number, number]> = [];
  const inwardDirs: Vec3[] = [];
  for (let i = 0; i < sortable.length; i += 2) {
    pairs.push([sortable[i].triId, sortable[i + 1].triId]);
    const theta = 0.5 * (sortable[i].theta + sortable[i + 1].theta);
    inwardDirs.push(add(scale(b1, Math.cos(theta)), scale(b2, Math.sin(theta))));
  }

  return { u, v, pairs, inwardDirs };
}

function singularVertexGroups(mesh: CPUMesh): SingularVertexGroup[] {
  const vertexToTriangles = new Map<number, number[]>();
  for (let i = 0; i < mesh.triangleCount; i++) {
    const tri = mesh.triangle(i);
    if (!tri) continue;
    pushMapArray(vertexToTriangles, tri.a, i);
    pushMapArray(vertexToTriangles, tri.b, i);
    pushMapArray(vertexToTriangles, tri.c, i);
  }

  const result: SingularVertexGroup[] = [];
  for (const [vertexId, triIndices] of vertexToTriangles.entries()) {
    if (triIndices.length < 2) continue;
    const dsu = new DisjointSet(triIndices.length);
    const edgeAroundVertex = new Map<number, number[]>();

    for (let localIndex = 0; localIndex < triIndices.length; localIndex++) {
      const tri = mesh.triangle(triIndices[localIndex]);
      if (!tri) continue;
      const [o1, o2] = otherSegmentOfTriangle(tri, vertexId);
      pushMapArray(edgeAroundVertex, o1, localIndex);
      pushMapArray(edgeAroundVertex, o2, localIndex);
    }

    for (const members of edgeAroundVertex.values()) {
      for (let i = 1; i < members.length; i++) dsu.union(members[0], members[i]);
    }

    const componentsByRoot = new Map<number, number[]>();
    for (let localIndex = 0; localIndex < triIndices.length; localIndex++) {
      pushMapArray(componentsByRoot, dsu.find(localIndex), triIndices[localIndex]);
    }

    const components = [...componentsByRoot.values()];
    if (components.length > 1) result.push({ vertex: vertexId, components });
  }

  return result;
}

function appendEdgeRecord(map: Map<string, { u: number; v: number; triangles: number[] }>, a: number, b: number, triId: number): void {
  const [u, v] = a < b ? [a, b] : [b, a];
  const key = `${u}:${v}`;
  let record = map.get(key);
  if (!record) {
    record = { u, v, triangles: [] };
    map.set(key, record);
  }
  record.triangles.push(triId);
}

function thirdVertexOfTriangle(tri: CPUTriangle, u: number, v: number): number {
  if (tri.a !== u && tri.a !== v) return tri.a;
  if (tri.b !== u && tri.b !== v) return tri.b;
  if (tri.c !== u && tri.c !== v) return tri.c;
  return -1;
}

function otherSegmentOfTriangle(tri: CPUTriangle, v: number): [number, number] {
  if (tri.a === v) return [tri.b, tri.c];
  if (tri.b === v) return [tri.c, tri.a];
  if (tri.c === v) return [tri.a, tri.b];
  throw new Error('vertex is not part of triangle');
}

function segmentOrientation(tri: CPUTriangle, u: number, v: number): boolean {
  const verts = [tri.a, tri.b, tri.c];
  for (let i = 0; i < 3; i++) {
    if (verts[i] === u) return verts[(i + 2) % 3] === v;
  }
  throw new Error('segmentOrientation(): first edge endpoint not present in triangle');
}

function clampOriginalVertex(mesh: CPUMesh, vertexId: number, grid: GridInfo, epsilon: number): void {
  if (!mesh.vertexOriginal(vertexId)) return;
  const cubeIndex = mesh.vertexCubeIndex(vertexId);
  if (cubeIndex === null) return;
  const [min, max] = cubeBoundsFromIndex(cubeIndex, grid);
  mesh.setVertexPosition(vertexId, clampVec3(mesh.vertexPosition(vertexId), addScalar(min, epsilon), addScalar(max, -epsilon)));
}

function cubeBoundsFromIndex(cubeIndex: number, grid: GridInfo): [Vec3, Vec3] {
  const ix = cubeIndex % grid.nx;
  const t = Math.floor(cubeIndex / grid.nx);
  const iy = t % grid.ny;
  const iz = Math.floor(t / grid.ny);
  const min: Vec3 = [
    grid.minCorner[0] + ix * grid.delta,
    grid.minCorner[1] + iy * grid.delta,
    grid.minCorner[2] + iz * grid.delta,
  ];
  const max: Vec3 = [min[0] + grid.delta, min[1] + grid.delta, min[2] + grid.delta];
  return [min, max];
}

class DisjointSet {
  readonly parent: Int32Array;
  readonly rank: Int32Array;

  constructor(size: number) {
    this.parent = new Int32Array(size);
    this.rank = new Int32Array(size);
    for (let i = 0; i < size; i++) this.parent[i] = i;
  }

  find(x: number): number {
    let p = this.parent[x];
    while (p !== this.parent[p]) p = this.parent[p];
    let cur = x;
    while (cur !== p) {
      const next = this.parent[cur];
      this.parent[cur] = p;
      cur = next;
    }
    return p;
  }

  union(a: number, b: number): void {
    let ra = this.find(a);
    let rb = this.find(b);
    if (ra === rb) return;
    if (this.rank[ra] < this.rank[rb]) [ra, rb] = [rb, ra];
    this.parent[rb] = ra;
    if (this.rank[ra] === this.rank[rb]) this.rank[ra]++;
  }
}

function pushMapArray<K>(map: Map<K, number[]>, key: K, value: number): void {
  let arr = map.get(key);
  if (!arr) {
    arr = [];
    map.set(key, arr);
  }
  arr.push(value);
}

function triangleNormal(mesh: CPUMesh, tri: CPUTriangle): Vec3 {
  const a = mesh.vertexPosition(tri.a);
  const b = mesh.vertexPosition(tri.b);
  const c = mesh.vertexPosition(tri.c);
  return normalizeSafe(cross(sub(b, a), sub(c, a)), [1, 0, 0]);
}

function vec3ExactKey(v: Vec3): string {
  return `${float32Bits(v[0])}:${float32Bits(v[1])}:${float32Bits(v[2])}`;
}

const f32Scratch = new Float32Array(1);
const u32Scratch = new Uint32Array(f32Scratch.buffer);

function float32Bits(x: number): number {
  f32Scratch[0] = x;
  return u32Scratch[0];
}

function hashDirection(seed: number): Vec3 {
  let x = (seed * 1664525 + 1013904223) >>> 0;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  const fx = (((x >>> 0) / 0xffffffff) * 2 - 1) || 0.37;
  x = (x * 1664525 + 1013904223) >>> 0;
  const fy = (((x >>> 0) / 0xffffffff) * 2 - 1) || -0.58;
  x = (x * 1664525 + 1013904223) >>> 0;
  const fz = (((x >>> 0) / 0xffffffff) * 2 - 1) || 0.71;
  return normalizeSafe([fx, fy, fz], [1, 0, 0]);
}


interface ShaderBundle {
  corner: string;
  edgeX: string;
  edgeY: string;
  edgeZ: string;
  cube: string;
  countX: string;
  countY: string;
  countZ: string;
  emitX: string;
  emitY: string;
  emitZ: string;
}

function buildShaderBundle(solidWGSL: string, workgroupSize: number): ShaderBundle {
  return {
    corner: buildCornerShader(solidWGSL, workgroupSize),
    edgeX: buildEdgeShader(solidWGSL, workgroupSize, 'x'),
    edgeY: buildEdgeShader(solidWGSL, workgroupSize, 'y'),
    edgeZ: buildEdgeShader(solidWGSL, workgroupSize, 'z'),
    cube: buildCubeShader(solidWGSL, workgroupSize),
    countX: buildCountShader(solidWGSL, workgroupSize, 'x'),
    countY: buildCountShader(solidWGSL, workgroupSize, 'y'),
    countZ: buildCountShader(solidWGSL, workgroupSize, 'z'),
    emitX: buildEmitShader(solidWGSL, workgroupSize, 'x'),
    emitY: buildEmitShader(solidWGSL, workgroupSize, 'y'),
    emitZ: buildEmitShader(solidWGSL, workgroupSize, 'z'),
  };
}

function surfaceHelpersWGSL(_solidWGSL: string): string {
  return /* wgsl */`
fn estimateNormal(p: vec3<f32>) -> vec3<f32> {
  let eps = max(params.normalStep, 1e-5);
  var axes = array<vec3<f32>, 3>(
    vec3<f32>(-0.7107294727984605, -0.12934902142019175, 0.6914712193238857) * eps,
    vec3<f32>(0.09870891687574183, -0.9915624053549226, -0.08402705526185106) * eps,
    vec3<f32>(0.696505682837434, 0.008533870423146774, 0.7175005274080017) * eps,
  );
  var contains = array<u32, 3>(
    select(0u, 1u, solidOccupancy(p + axes[0])),
    select(0u, 1u, solidOccupancy(p + axes[1])),
    select(0u, 1u, solidOccupancy(p + axes[2])),
  );
  var planeAxes = array<vec3<f32>, 2>(vec3<f32>(0.0), vec3<f32>(0.0));

  for (var i = 0u; i < 2u; i++) {
    var v1 = axes[i];
    let c1 = contains[i] != 0u;
    var v2 = axes[i + 1u];
    let c2 = contains[i + 1u] != 0u;
    if (!c1) {
      v1 = -v1;
    }
    if (c2) {
      v2 = -v2;
    }
    for (var j = 0u; j < 18u; j++) {
      var mp = v1 + v2;
      let mpNorm = length(mp);
      if (mpNorm > 1e-20) {
        mp *= eps / mpNorm;
      }
      if (solidOccupancy(p + mp)) {
        v1 = mp;
      } else {
        v2 = mp;
      }
    }
    planeAxes[i] = v1 + v2;
    if (i == 0u && abs(dot(planeAxes[0], axes[1])) > abs(dot(planeAxes[0], axes[0]))) {
      let tmpAxis = axes[0];
      axes[0] = axes[1];
      axes[1] = tmpAxis;
      let tmpContains = contains[0];
      contains[0] = contains[1];
      contains[1] = tmpContains;
    }
  }

  var res = normalize(cross(planeAxes[0], planeAxes[1]));
  if (solidOccupancy(p + res * eps)) {
    res = -res;
  }
  return res;
}

fn bisectOccupancyEdge(p0: vec3<f32>, p1: vec3<f32>, occ0: bool) -> vec3<f32> {
  var lo = p0;
  var hi = p1;
  var loOcc = occ0;
  var i = 0u;
  loop {
    if (i >= params.bisectionSteps) { break; }
    let mid = 0.5 * (lo + hi);
    let midOcc = solidOccupancy(mid);
    if (midOcc == loOcc) {
      lo = mid;
    } else {
      hi = mid;
    }
    i = i + 1u;
  }
  return 0.5 * (lo + hi);
}
`;
}

function wgslHeader(solidWGSL: string): string {
  const surfaceHelpers = surfaceHelpersWGSL(solidWGSL);
  return /* wgsl */`
struct DCParams {
  minCorner: vec3<f32>,
  delta: f32,
  cubeMargin: f32,
  l2Penalty: f32,
  normalStep: f32,
  singularValueEpsilon: f32,
  nx: u32,
  ny: u32,
  nz: u32,
  bisectionSteps: u32,
  clip: u32,
  triangleMode: u32,
  _pad0: u32,
  _pad1: u32,
};

struct HermiteEdge {
  isActive: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  pos: vec4<f32>,
  normal: vec4<f32>,
};

struct CubeVertex {
  isActive: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  pos: vec4<f32>,
};

struct TriangleIndex {
  a: u32,
  b: u32,
  c: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> params: DCParams;

${solidWGSL}

fn cornerIndex(ix: u32, iy: u32, iz: u32) -> u32 {
  let sx = params.nx + 1u;
  let sy = params.ny + 1u;
  return ix + sx * (iy + sy * iz);
}

fn cubeIndex(ix: u32, iy: u32, iz: u32) -> u32 {
  return ix + params.nx * (iy + params.ny * iz);
}

fn xEdgeIndex(ix: u32, iy: u32, iz: u32) -> u32 {
  return ix + params.nx * (iy + (params.ny + 1u) * iz);
}

fn yEdgeIndex(ix: u32, iy: u32, iz: u32) -> u32 {
  return ix + (params.nx + 1u) * (iy + params.ny * iz);
}

fn zEdgeIndex(ix: u32, iy: u32, iz: u32) -> u32 {
  return ix + (params.nx + 1u) * (iy + (params.ny + 1u) * iz);
}

fn cornerPosition(ix: u32, iy: u32, iz: u32) -> vec3<f32> {
  return params.minCorner + vec3<f32>(f32(ix), f32(iy), f32(iz)) * params.delta;
}

fn cubeMin(ix: u32, iy: u32, iz: u32) -> vec3<f32> {
  return cornerPosition(ix, iy, iz);
}

fn cubeMax(ix: u32, iy: u32, iz: u32) -> vec3<f32> {
  return cornerPosition(ix + 1u, iy + 1u, iz + 1u);
}

${surfaceHelpers}

fn triNormal(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
  let n = cross(b - a, c - a);
  let n2 = dot(n, n);
  if (n2 < 1e-20) {
    return vec3<f32>(0.0, 0.0, 0.0);
  }
  return normalize(n);
}

fn triArea2(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
  return length(cross(b - a, c - a));
}

${qefWGSL}

fn chooseFirstDiagonal(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, v3: vec3<f32>) -> bool {
  let areaA = min(triArea2(v0, v1, v2), triArea2(v0, v2, v3));
  let areaB = min(triArea2(v1, v2, v3), triArea2(v1, v3, v0));
  let dotA = dot(triNormal(v0, v1, v2), triNormal(v0, v2, v3));
  let dotB = dot(triNormal(v1, v2, v3), triNormal(v1, v3, v0));
  if (params.triangleMode == 0u) {
    return areaA > areaB;
  }
  if (params.triangleMode == 1u) {
    return dotA < dotB;
  }
  return dotA > dotB;
}
`;
}

function buildCornerShader(solidWGSL: string, workgroupSize: number): string {
  return /* wgsl */`
${wgslHeader(solidWGSL)}
@group(0) @binding(1) var<storage, read_write> corners: array<u32>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let sx = params.nx + 1u;
  let sy = params.ny + 1u;
  let sz = params.nz + 1u;
  let total = sx * sy * sz;
  if (i >= total) { return; }
  let ix = i % sx;
  let t = i / sx;
  let iy = t % sy;
  let iz = t / sy;
  corners[i] = select(0u, 1u, solidOccupancy(cornerPosition(ix, iy, iz)));
}
`;
}

function buildEdgeShader(solidWGSL: string, workgroupSize: number, axis: 'x' | 'y' | 'z'): string {
  const decode = axis === 'x'
    ? `
  let total = params.nx * (params.ny + 1u) * (params.nz + 1u);
  if (i >= total) { return; }
  let ix = i % params.nx;
  let t = i / params.nx;
  let iy = t % (params.ny + 1u);
  let iz = t / (params.ny + 1u);
  let c0 = cornerIndex(ix, iy, iz);
  let c1 = cornerIndex(ix + 1u, iy, iz);
  let p0 = cornerPosition(ix, iy, iz);
  let p1 = cornerPosition(ix + 1u, iy, iz);
`
    : axis === 'y'
      ? `
  let total = (params.nx + 1u) * params.ny * (params.nz + 1u);
  if (i >= total) { return; }
  let ix = i % (params.nx + 1u);
  let t = i / (params.nx + 1u);
  let iy = t % params.ny;
  let iz = t / params.ny;
  let c0 = cornerIndex(ix, iy, iz);
  let c1 = cornerIndex(ix, iy + 1u, iz);
  let p0 = cornerPosition(ix, iy, iz);
  let p1 = cornerPosition(ix, iy + 1u, iz);
`
      : `
  let total = (params.nx + 1u) * (params.ny + 1u) * params.nz;
  if (i >= total) { return; }
  let ix = i % (params.nx + 1u);
  let t = i / (params.nx + 1u);
  let iy = t % (params.ny + 1u);
  let iz = t / (params.ny + 1u);
  let c0 = cornerIndex(ix, iy, iz);
  let c1 = cornerIndex(ix, iy, iz + 1u);
  let p0 = cornerPosition(ix, iy, iz);
  let p1 = cornerPosition(ix, iy, iz + 1u);
`;

  return /* wgsl */`
${wgslHeader(solidWGSL)}
@group(0) @binding(1) var<storage, read> corners: array<u32>;
@group(0) @binding(2) var<storage, read_write> edges: array<HermiteEdge>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
${decode}
  let occ0 = corners[c0] != 0u;
  let occ1 = corners[c1] != 0u;
  if (occ0 == occ1) {
    edges[i].isActive = 0u;
    edges[i].pos = vec4<f32>(0.0);
    edges[i].normal = vec4<f32>(0.0);
    return;
  }
  let hit = bisectOccupancyEdge(p0, p1, occ0);
  edges[i].isActive = 1u;
  edges[i].pos = vec4<f32>(hit, 1.0);
  edges[i].normal = vec4<f32>(estimateNormal(hit), 0.0);
}
`;
}

function buildCubeShader(solidWGSL: string, workgroupSize: number): string {
  return /* wgsl */`
${wgslHeader(solidWGSL)}
@group(0) @binding(1) var<storage, read> corners: array<u32>;
@group(0) @binding(2) var<storage, read> xEdges: array<HermiteEdge>;
@group(0) @binding(3) var<storage, read> yEdges: array<HermiteEdge>;
@group(0) @binding(4) var<storage, read> zEdges: array<HermiteEdge>;
@group(0) @binding(5) var<storage, read_write> cubes: array<CubeVertex>;

fn cubeCornerValue(ix: u32, iy: u32, iz: u32, dx: u32, dy: u32, dz: u32) -> bool {
  return corners[cornerIndex(ix + dx, iy + dy, iz + dz)] != 0u;
}

fn clipToCube(p: vec3<f32>, ix: u32, iy: u32, iz: u32) -> vec3<f32> {
  let margin = params.cubeMargin;
  let lo = cubeMin(ix, iy, iz) + vec3<f32>(margin);
  let hi = cubeMax(ix, iy, iz) - vec3<f32>(margin);
  return clamp(p, lo, hi);
}

fn accumulateHermite(edge: HermiteEdge, mp: vec3<f32>, a00_: ptr<function, f32>, a01_: ptr<function, f32>, a02_: ptr<function, f32>, a11_: ptr<function, f32>, a12_: ptr<function, f32>, a22_: ptr<function, f32>, rhs_: ptr<function, vec3<f32>>) {
  if (edge.isActive == 0u) { return; }
  let n = edge.normal.xyz;
  let v = edge.pos.xyz - mp;
  *a00_ += n.x * n.x;
  *a01_ += n.x * n.y;
  *a02_ += n.x * n.z;
  *a11_ += n.y * n.y;
  *a12_ += n.y * n.z;
  *a22_ += n.z * n.z;
  *rhs_ += n * dot(v, n);
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let total = params.nx * params.ny * params.nz;
  if (i >= total) { return; }

  let ix = i % params.nx;
  let t = i / params.nx;
  let iy = t % params.ny;
  let iz = t / params.ny;

  let c000 = cubeCornerValue(ix, iy, iz, 0u, 0u, 0u);
  let c100 = cubeCornerValue(ix, iy, iz, 1u, 0u, 0u);
  let c010 = cubeCornerValue(ix, iy, iz, 0u, 1u, 0u);
  let c110 = cubeCornerValue(ix, iy, iz, 1u, 1u, 0u);
  let c001 = cubeCornerValue(ix, iy, iz, 0u, 0u, 1u);
  let c101 = cubeCornerValue(ix, iy, iz, 1u, 0u, 1u);
  let c011 = cubeCornerValue(ix, iy, iz, 0u, 1u, 1u);
  let c111 = cubeCornerValue(ix, iy, iz, 1u, 1u, 1u);

  let firstValue = c000;
  let activeCube =
    (c100 != firstValue) || (c010 != firstValue) || (c110 != firstValue) ||
    (c001 != firstValue) || (c101 != firstValue) || (c011 != firstValue) || (c111 != firstValue);
  if (!activeCube) {
    cubes[i].isActive = 0u;
    cubes[i].pos = vec4<f32>(0.0);
    return;
  }

  let e0 = xEdges[xEdgeIndex(ix, iy, iz)];
  let e1 = yEdges[yEdgeIndex(ix, iy, iz)];
  let e2 = yEdges[yEdgeIndex(ix + 1u, iy, iz)];
  let e3 = xEdges[xEdgeIndex(ix, iy + 1u, iz)];
  let e4 = zEdges[zEdgeIndex(ix, iy, iz)];
  let e5 = zEdges[zEdgeIndex(ix + 1u, iy, iz)];
  let e6 = zEdges[zEdgeIndex(ix, iy + 1u, iz)];
  let e7 = zEdges[zEdgeIndex(ix + 1u, iy + 1u, iz)];
  let e8 = xEdges[xEdgeIndex(ix, iy, iz + 1u)];
  let e9 = yEdges[yEdgeIndex(ix, iy, iz + 1u)];
  let e10 = yEdges[yEdgeIndex(ix + 1u, iy, iz + 1u)];
  let e11 = xEdges[xEdgeIndex(ix, iy + 1u, iz + 1u)];

  var massPoint = vec3<f32>(0.0);
  var count = 0.0;
  if (e0.isActive != 0u) { massPoint += e0.pos.xyz; count += 1.0; }
  if (e1.isActive != 0u) { massPoint += e1.pos.xyz; count += 1.0; }
  if (e2.isActive != 0u) { massPoint += e2.pos.xyz; count += 1.0; }
  if (e3.isActive != 0u) { massPoint += e3.pos.xyz; count += 1.0; }
  if (e4.isActive != 0u) { massPoint += e4.pos.xyz; count += 1.0; }
  if (e5.isActive != 0u) { massPoint += e5.pos.xyz; count += 1.0; }
  if (e6.isActive != 0u) { massPoint += e6.pos.xyz; count += 1.0; }
  if (e7.isActive != 0u) { massPoint += e7.pos.xyz; count += 1.0; }
  if (e8.isActive != 0u) { massPoint += e8.pos.xyz; count += 1.0; }
  if (e9.isActive != 0u) { massPoint += e9.pos.xyz; count += 1.0; }
  if (e10.isActive != 0u) { massPoint += e10.pos.xyz; count += 1.0; }
  if (e11.isActive != 0u) { massPoint += e11.pos.xyz; count += 1.0; }

  if (count <= 0.0) {
    cubes[i].isActive = 0u;
    cubes[i].pos = vec4<f32>(0.0);
    return;
  }
  massPoint /= count;

  var a00 = params.l2Penalty;
  var a01 = 0.0;
  var a02 = 0.0;
  var a11 = params.l2Penalty;
  var a12 = 0.0;
  var a22 = params.l2Penalty;
  var rhs = vec3<f32>(0.0);

  accumulateHermite(e0, massPoint, &a00, &a01, &a02, &a11, &a12, &a22, &rhs);
  accumulateHermite(e1, massPoint, &a00, &a01, &a02, &a11, &a12, &a22, &rhs);
  accumulateHermite(e2, massPoint, &a00, &a01, &a02, &a11, &a12, &a22, &rhs);
  accumulateHermite(e3, massPoint, &a00, &a01, &a02, &a11, &a12, &a22, &rhs);
  accumulateHermite(e4, massPoint, &a00, &a01, &a02, &a11, &a12, &a22, &rhs);
  accumulateHermite(e5, massPoint, &a00, &a01, &a02, &a11, &a12, &a22, &rhs);
  accumulateHermite(e6, massPoint, &a00, &a01, &a02, &a11, &a12, &a22, &rhs);
  accumulateHermite(e7, massPoint, &a00, &a01, &a02, &a11, &a12, &a22, &rhs);
  accumulateHermite(e8, massPoint, &a00, &a01, &a02, &a11, &a12, &a22, &rhs);
  accumulateHermite(e9, massPoint, &a00, &a01, &a02, &a11, &a12, &a22, &rhs);
  accumulateHermite(e10, massPoint, &a00, &a01, &a02, &a11, &a12, &a22, &rhs);
  accumulateHermite(e11, massPoint, &a00, &a01, &a02, &a11, &a12, &a22, &rhs);

  var p = massPoint + solveSymmetric3(a00, a01, a02, a11, a12, a22, rhs);
  if (params.clip != 0u) {
    p = clipToCube(p, ix, iy, iz);
  }

  cubes[i].isActive = 1u;
  cubes[i].pos = vec4<f32>(p, 1.0);
}
`;
}

function buildEmitShader(solidWGSL: string, workgroupSize: number, axis: 'x' | 'y' | 'z'): string {
  const decode = axis === 'x'
    ? `
  let total = params.nx * (params.ny + 1u) * (params.nz + 1u);
  if (i >= total) { return; }
  let ix = i % params.nx;
  let t = i / params.nx;
  let iy = t % (params.ny + 1u);
  let iz = t / (params.ny + 1u);
  if (iy == 0u || iz == 0u || iy >= params.ny || iz >= params.nz) { return; }
  let edge = edges[i];
  if (edge.isActive == 0u) { return; }
  var ids = array<u32, 4>(
    cubeIndex(ix, iy, iz - 1u),
    cubeIndex(ix, iy - 1u, iz - 1u),
    cubeIndex(ix, iy - 1u, iz),
    cubeIndex(ix, iy, iz),
  );
  let flip = corners[cornerIndex(ix, iy, iz)] != 0u;
`
    : axis === 'y'
      ? `
  let total = (params.nx + 1u) * params.ny * (params.nz + 1u);
  if (i >= total) { return; }
  let ix = i % (params.nx + 1u);
  let t = i / (params.nx + 1u);
  let iy = t % params.ny;
  let iz = t / params.ny;
  if (ix == 0u || iz == 0u || ix >= params.nx || iz >= params.nz) { return; }
  let edge = edges[i];
  if (edge.isActive == 0u) { return; }
  var ids = array<u32, 4>(
    cubeIndex(ix - 1u, iy, iz),
    cubeIndex(ix - 1u, iy, iz - 1u),
    cubeIndex(ix, iy, iz - 1u),
    cubeIndex(ix, iy, iz),
  );
  let flip = corners[cornerIndex(ix, iy, iz)] != 0u;
`
      : `
  let total = (params.nx + 1u) * (params.ny + 1u) * params.nz;
  if (i >= total) { return; }
  let ix = i % (params.nx + 1u);
  let t = i / (params.nx + 1u);
  let iy = t % (params.ny + 1u);
  let iz = t / (params.ny + 1u);
  if (ix == 0u || iy == 0u || ix >= params.nx || iy >= params.ny) { return; }
  let edge = edges[i];
  if (edge.isActive == 0u) { return; }
  var ids = array<u32, 4>(
    cubeIndex(ix, iy - 1u, iz),
    cubeIndex(ix - 1u, iy - 1u, iz),
    cubeIndex(ix - 1u, iy, iz),
    cubeIndex(ix, iy, iz),
  );
  let flip = corners[cornerIndex(ix, iy, iz)] != 0u;
`;

  return /* wgsl */`
${wgslHeader(solidWGSL)}
@group(0) @binding(1) var<storage, read> corners: array<u32>;
@group(0) @binding(2) var<storage, read> edges: array<HermiteEdge>;
@group(0) @binding(3) var<storage, read> cubes: array<CubeVertex>;
@group(0) @binding(4) var<storage, read> offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> triangles: array<TriangleIndex>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
${decode}
  var p0 = cubes[ids[0]].pos.xyz;
  var p1 = cubes[ids[1]].pos.xyz;
  var p2 = cubes[ids[2]].pos.xyz;
  var p3 = cubes[ids[3]].pos.xyz;
  if (cubes[ids[0]].isActive == 0u || cubes[ids[1]].isActive == 0u || cubes[ids[2]].isActive == 0u || cubes[ids[3]].isActive == 0u) {
    return;
  }

  if (flip) {
    let tmp0 = ids[0];
    let tmp1 = ids[1];
    ids[0] = ids[3];
    ids[1] = ids[2];
    ids[2] = tmp1;
    ids[3] = tmp0;
    let pp0 = p0;
    let pp1 = p1;
    p0 = p3;
    p1 = p2;
    p2 = pp1;
    p3 = pp0;
  }

  let useFirst = chooseFirstDiagonal(p0, p1, p2, p3);
  let base = offsets[i];
  if (useFirst) {
    triangles[base + 0u] = TriangleIndex(ids[0], ids[1], ids[2], 0u);
    triangles[base + 1u] = TriangleIndex(ids[0], ids[2], ids[3], 0u);
  } else {
    triangles[base + 0u] = TriangleIndex(ids[1], ids[2], ids[3], 0u);
    triangles[base + 1u] = TriangleIndex(ids[1], ids[3], ids[0], 0u);
  }
}
`;
}

function buildCountShader(solidWGSL: string, workgroupSize: number, axis: 'x' | 'y' | 'z'): string {
  const decode = axis === 'x'
    ? `
  let total = params.nx * (params.ny + 1u) * (params.nz + 1u);
  if (i >= total) { return; }
  let ix = i % params.nx;
  let t = i / params.nx;
  let iy = t % (params.ny + 1u);
  let iz = t / (params.ny + 1u);
  if (iy == 0u || iz == 0u || iy >= params.ny || iz >= params.nz) {
    counts[i] = 0u;
    return;
  }
  let edge = edges[i];
  if (edge.isActive == 0u) {
    counts[i] = 0u;
    return;
  }
  let ids = array<u32, 4>(
    cubeIndex(ix, iy, iz - 1u),
    cubeIndex(ix, iy - 1u, iz - 1u),
    cubeIndex(ix, iy - 1u, iz),
    cubeIndex(ix, iy, iz),
  );
`
    : axis === 'y'
      ? `
  let total = (params.nx + 1u) * params.ny * (params.nz + 1u);
  if (i >= total) { return; }
  let ix = i % (params.nx + 1u);
  let t = i / (params.nx + 1u);
  let iy = t % params.ny;
  let iz = t / params.ny;
  if (ix == 0u || iz == 0u || ix >= params.nx || iz >= params.nz) {
    counts[i] = 0u;
    return;
  }
  let edge = edges[i];
  if (edge.isActive == 0u) {
    counts[i] = 0u;
    return;
  }
  let ids = array<u32, 4>(
    cubeIndex(ix - 1u, iy, iz),
    cubeIndex(ix - 1u, iy, iz - 1u),
    cubeIndex(ix, iy, iz - 1u),
    cubeIndex(ix, iy, iz),
  );
`
      : `
  let total = (params.nx + 1u) * (params.ny + 1u) * params.nz;
  if (i >= total) { return; }
  let ix = i % (params.nx + 1u);
  let t = i / (params.nx + 1u);
  let iy = t % (params.ny + 1u);
  let iz = t / (params.ny + 1u);
  if (ix == 0u || iy == 0u || ix >= params.nx || iy >= params.ny) {
    counts[i] = 0u;
    return;
  }
  let edge = edges[i];
  if (edge.isActive == 0u) {
    counts[i] = 0u;
    return;
  }
  let ids = array<u32, 4>(
    cubeIndex(ix, iy - 1u, iz),
    cubeIndex(ix - 1u, iy - 1u, iz),
    cubeIndex(ix - 1u, iy, iz),
    cubeIndex(ix, iy, iz),
  );
`;

  return /* wgsl */`
${wgslHeader(solidWGSL)}
@group(0) @binding(1) var<storage, read> corners: array<u32>;
@group(0) @binding(2) var<storage, read> edges: array<HermiteEdge>;
@group(0) @binding(3) var<storage, read> cubes: array<CubeVertex>;
@group(0) @binding(4) var<storage, read_write> counts: array<u32>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
${decode}
  if (cubes[ids[0]].isActive == 0u || cubes[ids[1]].isActive == 0u || cubes[ids[2]].isActive == 0u || cubes[ids[3]].isActive == 0u) {
    counts[i] = 0u;
    return;
  }
  counts[i] = 2u;
}
`;
}

export const exampleSphereSolidWGSL = /* wgsl */`
fn solidOccupancy(p: vec3<f32>) -> bool {
  let center = vec3<f32>(0.0, 0.0, 0.0);
  let radius = 1.0;
  return distance(p, center) <= radius;
}
`;

export const exampleCutCornerSphereSolidWGSL = /* wgsl */`
fn solidOccupancy(p: vec3<f32>) -> bool {
  if (p.x > 0 && p.y > 0 && p.z > 0) {
    return false;
  }
  let center = vec3<f32>(0.0, 0.0, 0.0);
  let radius = 1.0;
  return distance(p, center) <= radius;
}
`;

function logDualContourTimings(label: string, timings: StageTiming[], totalMs: number): void {
  let cumulative = 0;
  const rows = timings.map(({ stage, ms }) => {
    cumulative += ms;
    return {
      stage,
      ms: Number(ms.toFixed(2)),
      cumulativeMs: Number(cumulative.toFixed(2)),
    };
  });
  console.groupCollapsed(`[${label}] stage timings (${totalMs.toFixed(2)} ms total)`);
  console.table(rows);
  console.groupEnd();
}
