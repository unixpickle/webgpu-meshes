import { CPUMesh } from './mesh';
import {
  GPUBufferUsage,
  GPUMapMode,
  GPUShaderStage,
  type GPUBindGroup,
  type GPUBindGroupLayout,
  type GPUBuffer,
  type GPUCommandEncoder,
  type GPUCompilationMessage,
  type GPUComputePassEncoder,
  type GPUComputePipeline,
  type GPUDevice,
  type GPUPipelineLayout,
  type GPUShaderModule,
} from './webgpu_types';
import {
  createSolidBindGroup,
  createSolidBindGroupLayout,
  prepareSolidBindings,
  transformSolidWGSL,
  type PreparedSolidBindings,
} from './solid_bindings';
import type { DualContourSolidBinding } from './dual_contouring';
import type { Vec3 } from './vec3';


const DEFAULT_MARCHING_CUBES_BUFFER_SIZE = 1_000_000;
const STATIC_PARAMS_SIZE = 48;
const BATCH_PARAMS_SIZE = 32;
const EDGE_STRIDE = 32;
const TRIANGLE_STRIDE = 16;
const MC_MAX_TRIANGLES_PER_CASE = 5;
const MC_TRIANGLE_DATA_STRIDE = MC_MAX_TRIANGLES_PER_CASE * 6;

type MCTriangle = [number, number, number, number, number, number];
type MCRotation = [number, number, number, number, number, number, number, number];

export interface MarchingCubesResult {
  mesh: CPUMesh;
  metrics: MarchingCubesMetrics;
}

export interface MarchingCubesMetrics {
  totalMs: number;
  stages: MarchingCubesStageTiming[];
}

export interface MarchingCubesStageTiming {
  stage: string;
  ms: number;
}

export interface MarchingCubesWebGPUOptions {
  device: GPUDevice;
  /**
   * WGSL source that defines:
   *   fn solidOccupancy(p: vec3<f32>) -> bool
   * The function is inlined into occupancy-evaluation shaders.
   */
  solidWGSL: string;
  /**
   * Optional read-only buffers exposed to solidWGSL as @group(1) bindings.
   * Bindings are assigned in array order starting at 0.
   */
  solidBindings?: DualContourSolidBinding[];
  min: Vec3;
  max: Vec3;
  delta: number;
  noJitter?: boolean;
  bisectionSteps?: number;
  bufferSize?: number;
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
  xEdgeCount: number;
  yEdgeCount: number;
  zEdgeCount: number;
}

interface BatchLayout {
  cornerRows: number;
  cubeRows: number;
  cornerPlaneSize: number;
  xEdgePlaneSize: number;
  yEdgePlaneSize: number;
  zEdgePlaneSize: number;
  cubePlaneSize: number;
}

interface DispatchRange {
  localZStart: number;
  localZCount: number;
}

interface BatchSpec {
  cornerFill: DispatchRange;
  cubeFill: DispatchRange;
}

interface ShaderBundle {
  corner: string;
  edgeX: string;
  edgeY: string;
  edgeZ: string;
  count: string;
  emit: string;
}

export async function marchingCubesWebGPU(options: MarchingCubesWebGPUOptions): Promise<MarchingCubesResult> {
  const timings: MarchingCubesStageTiming[] = [];
  const startTime = performance.now();
  let stageStartTime = startTime;
  const markStage = (stage: string) => {
    const now = performance.now();
    timings.push({ stage, ms: now - stageStartTime });
    stageStartTime = now;
  };

  const config = normalizeOptions(options);
  const grid = createGridInfo(config.min, config.max, config.delta, config.noJitter);
  const layout = createBatchLayout(grid, config.bufferSize);
  markStage('normalize options + grid');

  const device = config.device;
  const solidBindings = prepareSolidBindings(device, config.solidBindings ?? [], config.label, 'marchingCubesWebGPU()');
  const workgroupSize = config.workgroupSize;
  const deviceLimits = device.limits as Record<string, number | undefined> | undefined;
  const maxStorageBindingSize = deviceLimits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const maxBufferSize = deviceLimits?.maxBufferSize ?? Number.POSITIVE_INFINITY;

  const cornerBufferSize = layout.cornerPlaneSize * layout.cornerRows * 4;
  const xEdgeBufferSize = layout.xEdgePlaneSize * layout.cornerRows * EDGE_STRIDE;
  const yEdgeBufferSize = layout.yEdgePlaneSize * layout.cornerRows * EDGE_STRIDE;
  const zEdgeBufferSize = layout.zEdgePlaneSize * layout.cubeRows * EDGE_STRIDE;
  const cubeCountBufferSize = layout.cubePlaneSize * layout.cubeRows * 4;
  const xEdgeReadbackSize = xEdgeBufferSize;
  const yEdgeReadbackSize = yEdgeBufferSize;
  const zEdgeReadbackSize = zEdgeBufferSize;

  assertBufferFits('corner buffer', cornerBufferSize, maxStorageBindingSize, maxBufferSize, true);
  assertBufferFits('x edge buffer', xEdgeBufferSize, maxStorageBindingSize, maxBufferSize, true);
  assertBufferFits('y edge buffer', yEdgeBufferSize, maxStorageBindingSize, maxBufferSize, true);
  assertBufferFits('z edge buffer', zEdgeBufferSize, maxStorageBindingSize, maxBufferSize, true);
  assertBufferFits('cube triangle count buffer', cubeCountBufferSize, maxStorageBindingSize, maxBufferSize, true);
  assertBufferFits('cube triangle offset buffer', cubeCountBufferSize, maxStorageBindingSize, maxBufferSize, true);
  assertBufferFits('x edge readback buffer', xEdgeReadbackSize, undefined, maxBufferSize, false);
  assertBufferFits('y edge readback buffer', yEdgeReadbackSize, undefined, maxBufferSize, false);
  assertBufferFits('z edge readback buffer', zEdgeReadbackSize, undefined, maxBufferSize, false);
  assertBufferFits('cube triangle count readback buffer', cubeCountBufferSize, undefined, maxBufferSize, false);

  const cornerBuffer = device.createBuffer({
    label: `${config.label}-mc-corners`,
    size: cornerBufferSize,
    usage: GPUBufferUsage.STORAGE,
  });
  const xEdgeBuffer = device.createBuffer({
    label: `${config.label}-mc-x-edges`,
    size: xEdgeBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const yEdgeBuffer = device.createBuffer({
    label: `${config.label}-mc-y-edges`,
    size: yEdgeBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const zEdgeBuffer = device.createBuffer({
    label: `${config.label}-mc-z-edges`,
    size: zEdgeBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const cubeTriangleCountBuffer = device.createBuffer({
    label: `${config.label}-mc-triangle-counts`,
    size: cubeCountBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const cubeTriangleOffsetBuffer = device.createBuffer({
    label: `${config.label}-mc-triangle-offsets`,
    size: cubeCountBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  markStage('create GPU buffers');

  const staticUniformBuffer = device.createBuffer({
    label: `${config.label}-mc-static-params`,
    size: STATIC_PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const cornerRangeBuffer = device.createBuffer({
    label: `${config.label}-mc-corner-range`,
    size: BATCH_PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const cubeRangeBuffer = device.createBuffer({
    label: `${config.label}-mc-cube-range`,
    size: BATCH_PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(staticUniformBuffer, 0, packStaticUniforms(grid, layout, config));

  const cornerBindGroupLayout = createInternalBindGroupLayout(device, [
    { binding: 0, type: 'uniform' },
    { binding: 1, type: 'uniform' },
    { binding: 2, type: 'storage' },
  ]);
  const edgeBindGroupLayout = createInternalBindGroupLayout(device, [
    { binding: 0, type: 'uniform' },
    { binding: 1, type: 'uniform' },
    { binding: 2, type: 'read-only-storage' },
    { binding: 3, type: 'storage' },
  ]);
  const countBindGroupLayout = createInternalBindGroupLayout(device, [
    { binding: 0, type: 'uniform' },
    { binding: 1, type: 'uniform' },
    { binding: 2, type: 'read-only-storage' },
    { binding: 3, type: 'storage' },
  ]);
  const emitBindGroupLayout = createInternalBindGroupLayout(device, [
    { binding: 0, type: 'uniform' },
    { binding: 1, type: 'uniform' },
    { binding: 2, type: 'read-only-storage' },
    { binding: 3, type: 'read-only-storage' },
    { binding: 4, type: 'storage' },
  ]);
  const solidBindGroupLayout = solidBindings.bindings.length === 0 ? null : createSolidBindGroupLayout(device, solidBindings);

  const shaders = buildShaderBundle(config.solidWGSL, solidBindings, workgroupSize);
  const [cornerModule, xEdgeModule, yEdgeModule, zEdgeModule, countModule, emitModule] = await Promise.all([
    createCheckedShaderModule(device, `${config.label}-mc-corner-shader`, shaders.corner),
    createCheckedShaderModule(device, `${config.label}-mc-x-edge-shader`, shaders.edgeX),
    createCheckedShaderModule(device, `${config.label}-mc-y-edge-shader`, shaders.edgeY),
    createCheckedShaderModule(device, `${config.label}-mc-z-edge-shader`, shaders.edgeZ),
    createCheckedShaderModule(device, `${config.label}-mc-count-shader`, shaders.count),
    createCheckedShaderModule(device, `${config.label}-mc-emit-shader`, shaders.emit),
  ]);
  const cornerPipeline = await device.createComputePipelineAsync({
    label: `${config.label}-mc-corner-pipeline`,
    layout: createPipelineLayout(device, cornerBindGroupLayout, solidBindGroupLayout),
    compute: { module: cornerModule, entryPoint: 'main' },
  });
  const xEdgePipeline = await device.createComputePipelineAsync({
    label: `${config.label}-mc-x-edge-pipeline`,
    layout: createPipelineLayout(device, edgeBindGroupLayout, solidBindGroupLayout),
    compute: { module: xEdgeModule, entryPoint: 'main' },
  });
  const yEdgePipeline = await device.createComputePipelineAsync({
    label: `${config.label}-mc-y-edge-pipeline`,
    layout: createPipelineLayout(device, edgeBindGroupLayout, solidBindGroupLayout),
    compute: { module: yEdgeModule, entryPoint: 'main' },
  });
  const zEdgePipeline = await device.createComputePipelineAsync({
    label: `${config.label}-mc-z-edge-pipeline`,
    layout: createPipelineLayout(device, edgeBindGroupLayout, solidBindGroupLayout),
    compute: { module: zEdgeModule, entryPoint: 'main' },
  });
  const countPipeline = await device.createComputePipelineAsync({
    label: `${config.label}-mc-count-pipeline`,
    layout: createPipelineLayout(device, countBindGroupLayout),
    compute: { module: countModule, entryPoint: 'main' },
  });
  const emitPipeline = await device.createComputePipelineAsync({
    label: `${config.label}-mc-emit-pipeline`,
    layout: createPipelineLayout(device, emitBindGroupLayout),
    compute: { module: emitModule, entryPoint: 'main' },
  });
  markStage('build shaders + pipelines');

  const cornerBindGroup = device.createBindGroup({
    layout: cornerBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: staticUniformBuffer } },
      { binding: 1, resource: { buffer: cornerRangeBuffer } },
      { binding: 2, resource: { buffer: cornerBuffer } },
    ],
  });
  const xEdgeBindGroup = device.createBindGroup({
    layout: edgeBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: staticUniformBuffer } },
      { binding: 1, resource: { buffer: cornerRangeBuffer } },
      { binding: 2, resource: { buffer: cornerBuffer } },
      { binding: 3, resource: { buffer: xEdgeBuffer } },
    ],
  });
  const yEdgeBindGroup = device.createBindGroup({
    layout: edgeBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: staticUniformBuffer } },
      { binding: 1, resource: { buffer: cornerRangeBuffer } },
      { binding: 2, resource: { buffer: cornerBuffer } },
      { binding: 3, resource: { buffer: yEdgeBuffer } },
    ],
  });
  const zEdgeBindGroup = device.createBindGroup({
    layout: edgeBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: staticUniformBuffer } },
      { binding: 1, resource: { buffer: cubeRangeBuffer } },
      { binding: 2, resource: { buffer: cornerBuffer } },
      { binding: 3, resource: { buffer: zEdgeBuffer } },
    ],
  });
  const countBindGroup = device.createBindGroup({
    layout: countBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: staticUniformBuffer } },
      { binding: 1, resource: { buffer: cubeRangeBuffer } },
      { binding: 2, resource: { buffer: cornerBuffer } },
      { binding: 3, resource: { buffer: cubeTriangleCountBuffer } },
    ],
  });
  const solidBindGroup = solidBindGroupLayout === null ? null : createSolidBindGroup(device, solidBindGroupLayout, solidBindings);
  markStage('create bind groups');

  const xEdgeReadback = device.createBuffer({
    label: `${config.label}-mc-x-edge-readback`,
    size: xEdgeReadbackSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const yEdgeReadback = device.createBuffer({
    label: `${config.label}-mc-y-edge-readback`,
    size: yEdgeReadbackSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const zEdgeReadback = device.createBuffer({
    label: `${config.label}-mc-z-edge-readback`,
    size: zEdgeReadbackSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const cubeCountReadback = device.createBuffer({
    label: `${config.label}-mc-count-readback`,
    size: cubeCountBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  markStage('create readback buffers');

  const mesh = new CPUMesh([], []);
  const edgeVertexIds = new Map<number, number>();
  let zOffset = 0;
  let cornerRingHead = 0;
  let cubeRingHead = 0;
  let advanceIntoBatch = 0;
  let batchIndex = 0;

  for (;;) {
    const remainingAfterWindow = grid.nz + 1 - (layout.cornerRows + zOffset);
    const batchSpec = createBatchSpec(layout, advanceIntoBatch);
    const cubeCount = batchSpec.cubeFill.localZCount * layout.cubePlaneSize;

    device.queue.writeBuffer(cornerRangeBuffer, 0, packBatchUniforms(zOffset, cornerRingHead, cubeRingHead, batchSpec.cornerFill));
    device.queue.writeBuffer(cubeRangeBuffer, 0, packBatchUniforms(zOffset, cornerRingHead, cubeRingHead, batchSpec.cubeFill));

    const countEncoder = device.createCommandEncoder({ label: `${config.label}-mc-count-encoder-${batchIndex}` });
    {
      const pass = countEncoder.beginComputePass({ label: `${config.label}-mc-count-pass-${batchIndex}` });
      dispatch1D(pass, cornerPipeline, cornerBindGroup, batchSpec.cornerFill.localZCount * layout.cornerPlaneSize, workgroupSize, solidBindGroup);
      dispatch1D(pass, xEdgePipeline, xEdgeBindGroup, batchSpec.cornerFill.localZCount * layout.xEdgePlaneSize, workgroupSize, solidBindGroup);
      dispatch1D(pass, yEdgePipeline, yEdgeBindGroup, batchSpec.cornerFill.localZCount * layout.yEdgePlaneSize, workgroupSize, solidBindGroup);
      dispatch1D(pass, zEdgePipeline, zEdgeBindGroup, batchSpec.cubeFill.localZCount * layout.zEdgePlaneSize, workgroupSize, solidBindGroup);
      dispatch1D(pass, countPipeline, countBindGroup, cubeCount, workgroupSize);
      pass.end();
    }
    if (cubeCount > 0) {
      countEncoder.copyBufferToBuffer(cubeTriangleCountBuffer, 0, cubeCountReadback, 0, cubeCount * 4);
    }
    device.queue.submit([countEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    markStage(`batch ${batchIndex}: fill + count`);

    const countBytesRaw = cubeCount > 0 ? await readBuffer(cubeCountReadback) : new ArrayBuffer(0);
    const cubeTriangleCounts = new Uint32Array(countBytesRaw.slice(0, cubeCount * 4));
    const cubeTriangleOffsets = exclusiveScanCounts(cubeTriangleCounts);
    const triangleCount = cubeTriangleOffsets.totalCount;
    const triangleBufferSize = Math.max(16, triangleCount * TRIANGLE_STRIDE);
    if (triangleBufferSize > maxStorageBindingSize) {
      throw new Error(
        `Triangle output buffer requires ${triangleBufferSize} bytes, which exceeds the device storage-buffer binding limit of ${maxStorageBindingSize} bytes. ` +
        `Use a larger delta / smaller bounds, or reduce bufferSize.`,
      );
    }
    if (triangleBufferSize > maxBufferSize) {
      throw new Error(
        `Triangle output buffer requires ${triangleBufferSize} bytes, which exceeds the device maxBufferSize of ${maxBufferSize} bytes. ` +
        `Use a larger delta or smaller bounds.`,
      );
    }
    markStage(`batch ${batchIndex}: decode counts`);

    if (cubeTriangleOffsets.offsets.byteLength > 0) {
      device.queue.writeBuffer(cubeTriangleOffsetBuffer, 0, cubeTriangleOffsets.offsets);
    }

    const triangleBuffer = device.createBuffer({
      label: `${config.label}-mc-triangles-${batchIndex}`,
      size: triangleBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const triangleReadback = device.createBuffer({
      label: `${config.label}-mc-triangle-readback-${batchIndex}`,
      size: triangleBufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const emitBindGroup = device.createBindGroup({
      layout: emitBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: staticUniformBuffer } },
        { binding: 1, resource: { buffer: cubeRangeBuffer } },
        { binding: 2, resource: { buffer: cornerBuffer } },
        { binding: 3, resource: { buffer: cubeTriangleOffsetBuffer } },
        { binding: 4, resource: { buffer: triangleBuffer } },
      ],
    });

    const emitEncoder = device.createCommandEncoder({ label: `${config.label}-mc-emit-encoder-${batchIndex}` });
    {
      const pass = emitEncoder.beginComputePass({ label: `${config.label}-mc-emit-pass-${batchIndex}` });
      dispatch1D(pass, emitPipeline, emitBindGroup, cubeCount, workgroupSize);
      pass.end();
    }
    copyLogicalRowsToReadback(
      emitEncoder,
      xEdgeBuffer,
      xEdgeReadback,
      layout.xEdgePlaneSize * EDGE_STRIDE,
      layout.cornerRows,
      cornerRingHead,
      batchSpec.cubeFill.localZStart,
      batchSpec.cubeFill.localZCount + 1,
    );
    copyLogicalRowsToReadback(
      emitEncoder,
      yEdgeBuffer,
      yEdgeReadback,
      layout.yEdgePlaneSize * EDGE_STRIDE,
      layout.cornerRows,
      cornerRingHead,
      batchSpec.cubeFill.localZStart,
      batchSpec.cubeFill.localZCount + 1,
    );
    copyLogicalRowsToReadback(
      emitEncoder,
      zEdgeBuffer,
      zEdgeReadback,
      layout.zEdgePlaneSize * EDGE_STRIDE,
      layout.cubeRows,
      cubeRingHead,
      batchSpec.cubeFill.localZStart,
      batchSpec.cubeFill.localZCount,
    );
    emitEncoder.copyBufferToBuffer(triangleBuffer, 0, triangleReadback, 0, triangleBufferSize);
    device.queue.submit([emitEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    markStage(`batch ${batchIndex}: emit`);

    const [xEdgeBytesRaw, yEdgeBytesRaw, zEdgeBytesRaw, triangleBytesRaw] = await Promise.all([
      readBuffer(xEdgeReadback),
      readBuffer(yEdgeReadback),
      readBuffer(zEdgeReadback),
      readBuffer(triangleReadback),
    ]);
    const xRowCount = batchSpec.cubeFill.localZCount + 1;
    const yRowCount = batchSpec.cubeFill.localZCount + 1;
    const zRowCount = batchSpec.cubeFill.localZCount;
    const xEdgeBytes = xEdgeBytesRaw.slice(0, xRowCount * layout.xEdgePlaneSize * EDGE_STRIDE);
    const yEdgeBytes = yEdgeBytesRaw.slice(0, yRowCount * layout.yEdgePlaneSize * EDGE_STRIDE);
    const zEdgeBytes = zEdgeBytesRaw.slice(0, zRowCount * layout.zEdgePlaneSize * EDGE_STRIDE);
    const triangleBytes = triangleBytesRaw.slice(0, triangleCount * TRIANGLE_STRIDE);

    appendBatchXEdgeVertices(mesh, edgeVertexIds, xEdgeBytes, grid, zOffset + batchSpec.cubeFill.localZStart, xRowCount);
    appendBatchYEdgeVertices(mesh, edgeVertexIds, yEdgeBytes, grid, zOffset + batchSpec.cubeFill.localZStart, yRowCount);
    appendBatchZEdgeVertices(mesh, edgeVertexIds, zEdgeBytes, grid, zOffset + batchSpec.cubeFill.localZStart, zRowCount);
    appendBatchTriangles(mesh, edgeVertexIds, decodeTriangles(triangleBytes, triangleCount), triangleCount);
    triangleBuffer.destroy();
    triangleReadback.destroy();
    markStage(`batch ${batchIndex}: CPU mesh assembly`);

    if (remainingAfterWindow === 0) {
      break;
    }
    advanceIntoBatch = Math.min(remainingAfterWindow, layout.cornerRows - 2);
    zOffset += advanceIntoBatch;
    cornerRingHead = (cornerRingHead + advanceIntoBatch) % layout.cornerRows;
    cubeRingHead = (cubeRingHead + advanceIntoBatch) % layout.cubeRows;
    batchIndex++;
  }

  return {
    mesh,
    metrics: {
      totalMs: performance.now() - startTime,
      stages: timings,
    },
  };
}

function normalizeOptions(options: MarchingCubesWebGPUOptions): Required<Omit<MarchingCubesWebGPUOptions, 'device' | 'solidWGSL' | 'solidBindings' | 'min' | 'max' | 'delta'>> & Pick<MarchingCubesWebGPUOptions, 'device' | 'solidWGSL' | 'solidBindings' | 'min' | 'max' | 'delta'> {
  const bisectionSteps = options.bisectionSteps ?? 32;
  if (!Number.isInteger(bisectionSteps) || bisectionSteps < 1) {
    throw new Error('marchingCubesWebGPU(): bisectionSteps must be an integer >= 1.');
  }
  return {
    device: options.device,
    solidWGSL: options.solidWGSL,
    solidBindings: options.solidBindings ?? [],
    min: options.min,
    max: options.max,
    delta: options.delta,
    noJitter: options.noJitter ?? false,
    bisectionSteps,
    bufferSize: options.bufferSize ?? DEFAULT_MARCHING_CUBES_BUFFER_SIZE,
    workgroupSize: options.workgroupSize ?? 256,
    label: options.label ?? 'marching-cubes-webgpu',
  };
}

function createGridInfo(min: Vec3, max: Vec3, delta: number, noJitter: boolean): GridInfo {
  const jitter = noJitter ? 0 : delta * 0.012923982;
  const minExpanded: Vec3 = [min[0] - delta + jitter, min[1] - delta + jitter, min[2] - delta + jitter];
  const maxExpanded: Vec3 = [max[0] + delta + jitter, max[1] + delta + jitter, max[2] + delta + jitter];
  const countX = Math.round((maxExpanded[0] - minExpanded[0]) / delta) + 1;
  const countY = Math.round((maxExpanded[1] - minExpanded[1]) / delta) + 1;
  const countZ = Math.round((maxExpanded[2] - minExpanded[2]) / delta) + 1;
  if (countX < 2 || countY < 2 || countZ < 2) {
    throw new Error('marchingCubesWebGPU(): invalid bounds; expanded grid must contain at least one cube along each axis.');
  }
  const nx = countX - 1;
  const ny = countY - 1;
  const nz = countZ - 1;
  return {
    minCorner: minExpanded,
    maxCorner: [minExpanded[0] + nx * delta, minExpanded[1] + ny * delta, minExpanded[2] + nz * delta],
    delta,
    nx,
    ny,
    nz,
    xEdgeCount: nx * (ny + 1) * (nz + 1),
    yEdgeCount: (nx + 1) * ny * (nz + 1),
    zEdgeCount: (nx + 1) * (ny + 1) * nz,
  };
}

function createBatchLayout(grid: GridInfo, bufferSize: number): BatchLayout {
  const cornerPlaneSize = (grid.nx + 1) * (grid.ny + 1);
  const totalCornerRows = grid.nz + 1;
  const cornerRows = Math.min(Math.max(Math.floor(bufferSize / cornerPlaneSize), 4), totalCornerRows);
  return {
    cornerRows,
    cubeRows: cornerRows - 1,
    cornerPlaneSize,
    xEdgePlaneSize: grid.nx * (grid.ny + 1),
    yEdgePlaneSize: (grid.nx + 1) * grid.ny,
    zEdgePlaneSize: (grid.nx + 1) * (grid.ny + 1),
    cubePlaneSize: grid.nx * grid.ny,
  };
}

function createBatchSpec(layout: BatchLayout, advanceIntoBatch: number): BatchSpec {
  const cornerFill = advanceIntoBatch === 0
    ? { localZStart: 0, localZCount: layout.cornerRows }
    : { localZStart: layout.cornerRows - advanceIntoBatch, localZCount: advanceIntoBatch };
  const cubeFill = advanceIntoBatch === 0
    ? { localZStart: 0, localZCount: layout.cubeRows }
    : { localZStart: layout.cubeRows - advanceIntoBatch, localZCount: advanceIntoBatch };
  return { cornerFill, cubeFill };
}

function packStaticUniforms(grid: GridInfo, layout: BatchLayout, options: ReturnType<typeof normalizeOptions>): ArrayBuffer {
  const buffer = new ArrayBuffer(STATIC_PARAMS_SIZE);
  const f32 = new Float32Array(buffer);
  const u32 = new Uint32Array(buffer);
  f32[0] = grid.minCorner[0];
  f32[1] = grid.minCorner[1];
  f32[2] = grid.minCorner[2];
  f32[3] = grid.delta;
  u32[4] = grid.nx;
  u32[5] = grid.ny;
  u32[6] = grid.nz;
  u32[7] = options.bisectionSteps;
  u32[8] = layout.cornerRows;
  u32[9] = layout.cubeRows;
  return buffer;
}

function packBatchUniforms(
  zOffset: number,
  cornerRingHead: number,
  cubeRingHead: number,
  range: DispatchRange,
): ArrayBuffer {
  const buffer = new ArrayBuffer(BATCH_PARAMS_SIZE);
  const u32 = new Uint32Array(buffer);
  u32[0] = zOffset;
  u32[1] = cornerRingHead;
  u32[2] = cubeRingHead;
  u32[3] = range.localZStart;
  u32[4] = range.localZCount;
  return buffer;
}

function dispatch1D(pass: GPUComputePassEncoder, pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, count: number, workgroupSize: number, solidBindGroup?: GPUBindGroup | null): void {
  if (count <= 0) return;
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  if (solidBindGroup) {
    pass.setBindGroup(1, solidBindGroup);
  }
  pass.dispatchWorkgroups(Math.ceil(count / workgroupSize));
}

async function readBuffer(buffer: GPUBuffer): Promise<ArrayBuffer> {
  await buffer.mapAsync(GPUMapMode.READ);
  const src = buffer.getMappedRange();
  const out = src.slice(0);
  buffer.unmap();
  return out;
}

function createInternalBindGroupLayout(
  device: GPUDevice,
  entries: Array<{ binding: number; type: 'uniform' | 'storage' | 'read-only-storage' }>,
): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    entries: entries.map((entry) => ({
      binding: entry.binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: entry.type },
    })),
  });
}

function createPipelineLayout(device: GPUDevice, primaryLayout: GPUBindGroupLayout, solidBindGroupLayout?: GPUBindGroupLayout | null): GPUPipelineLayout {
  return device.createPipelineLayout({
    bindGroupLayouts: solidBindGroupLayout ? [primaryLayout, solidBindGroupLayout] : [primaryLayout],
  });
}

async function createCheckedShaderModule(device: GPUDevice, label: string, code: string): Promise<GPUShaderModule> {
  const module = device.createShaderModule({ label, code });
  if (typeof module.getCompilationInfo !== 'function') {
    return module;
  }
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === 'error');
  if (errors.length === 0) {
    return module;
  }
  throw new Error(formatShaderCompilationError(label, errors));
}

function formatShaderCompilationError(label: string, errors: readonly GPUCompilationMessage[]): string {
  const lines = [`WGSL compilation failed for ${label}:`];
  for (const error of errors.slice(0, 8)) {
    const location = typeof error.lineNum === 'number'
      ? `line ${error.lineNum}${typeof error.linePos === 'number' ? `:${error.linePos}` : ''}`
      : 'unknown location';
    lines.push(`${location}: ${String(error.message ?? 'Unknown shader compilation error.')}`);
  }
  if (errors.length > 8) {
    lines.push(`...and ${errors.length - 8} more errors.`);
  }
  return lines.join('\n');
}

function assertBufferFits(
  label: string,
  size: number,
  maxStorageBindingSize: number | undefined,
  maxBufferSize: number,
  isStorage: boolean,
): void {
  if (isStorage && maxStorageBindingSize !== undefined && size > maxStorageBindingSize) {
    throw new Error(`${label} requires ${size} bytes, which exceeds the device storage-buffer binding limit of ${maxStorageBindingSize} bytes.`);
  }
  if (size > maxBufferSize) {
    throw new Error(`${label} requires ${size} bytes, which exceeds the device maxBufferSize of ${maxBufferSize} bytes.`);
  }
}

function copyLogicalRowsToReadback(
  encoder: GPUCommandEncoder,
  srcBuffer: GPUBuffer,
  dstBuffer: GPUBuffer,
  rowBytes: number,
  ringRows: number,
  ringHead: number,
  localZStart: number,
  rowCount: number,
): void {
  if (rowCount <= 0) return;
  let remaining = rowCount;
  let currentLocalZ = localZStart;
  let dstOffset = 0;
  while (remaining > 0) {
    const physicalRow = (ringHead + currentLocalZ) % ringRows;
    const segmentRows = Math.min(remaining, ringRows - physicalRow);
    encoder.copyBufferToBuffer(
      srcBuffer,
      physicalRow * rowBytes,
      dstBuffer,
      dstOffset,
      segmentRows * rowBytes,
    );
    currentLocalZ += segmentRows;
    remaining -= segmentRows;
    dstOffset += segmentRows * rowBytes;
  }
}

function appendBatchXEdgeVertices(mesh: CPUMesh, edgeVertexIds: Map<number, number>, bytes: ArrayBuffer, grid: GridInfo, globalZStart: number, rowCount: number): void {
  if (rowCount <= 0) return;
  const view = new DataView(bytes);
  for (let localZ = 0; localZ < rowCount; localZ++) {
    const globalZ = globalZStart + localZ;
    for (let iy = 0; iy <= grid.ny; iy++) {
      for (let ix = 0; ix < grid.nx; ix++) {
        const localIndex = ix + grid.nx * (iy + (grid.ny + 1) * localZ);
        const base = localIndex * EDGE_STRIDE;
        if (view.getUint32(base + 0, true) === 0) continue;
        const edgeIndex = xEdgeGlobalIndex(ix, iy, globalZ, grid);
        if (edgeVertexIds.has(edgeIndex)) continue;
        const vertexId = mesh.addVertex({
          position: [
            view.getFloat32(base + 16, true),
            view.getFloat32(base + 20, true),
            view.getFloat32(base + 24, true),
          ],
          cubeIndex: null,
          original: true,
        });
        edgeVertexIds.set(edgeIndex, vertexId);
      }
    }
  }
}

function appendBatchYEdgeVertices(mesh: CPUMesh, edgeVertexIds: Map<number, number>, bytes: ArrayBuffer, grid: GridInfo, globalZStart: number, rowCount: number): void {
  if (rowCount <= 0) return;
  const view = new DataView(bytes);
  for (let localZ = 0; localZ < rowCount; localZ++) {
    const globalZ = globalZStart + localZ;
    for (let iy = 0; iy < grid.ny; iy++) {
      for (let ix = 0; ix <= grid.nx; ix++) {
        const localIndex = ix + (grid.nx + 1) * (iy + grid.ny * localZ);
        const base = localIndex * EDGE_STRIDE;
        if (view.getUint32(base + 0, true) === 0) continue;
        const edgeIndex = yEdgeGlobalIndex(ix, iy, globalZ, grid);
        if (edgeVertexIds.has(edgeIndex)) continue;
        const vertexId = mesh.addVertex({
          position: [
            view.getFloat32(base + 16, true),
            view.getFloat32(base + 20, true),
            view.getFloat32(base + 24, true),
          ],
          cubeIndex: null,
          original: true,
        });
        edgeVertexIds.set(edgeIndex, vertexId);
      }
    }
  }
}

function appendBatchZEdgeVertices(mesh: CPUMesh, edgeVertexIds: Map<number, number>, bytes: ArrayBuffer, grid: GridInfo, globalZStart: number, rowCount: number): void {
  if (rowCount <= 0) return;
  const view = new DataView(bytes);
  for (let localZ = 0; localZ < rowCount; localZ++) {
    const globalZ = globalZStart + localZ;
    for (let iy = 0; iy <= grid.ny; iy++) {
      for (let ix = 0; ix <= grid.nx; ix++) {
        const localIndex = ix + (grid.nx + 1) * (iy + (grid.ny + 1) * localZ);
        const base = localIndex * EDGE_STRIDE;
        if (view.getUint32(base + 0, true) === 0) continue;
        const edgeIndex = zEdgeGlobalIndex(ix, iy, globalZ, grid);
        if (edgeVertexIds.has(edgeIndex)) continue;
        const vertexId = mesh.addVertex({
          position: [
            view.getFloat32(base + 16, true),
            view.getFloat32(base + 20, true),
            view.getFloat32(base + 24, true),
          ],
          cubeIndex: null,
          original: true,
        });
        edgeVertexIds.set(edgeIndex, vertexId);
      }
    }
  }
}

function decodeTriangles(bytes: ArrayBuffer, count: number): Int32Array {
  const view = new DataView(bytes);
  const safeCount = Math.min(count, Math.floor(bytes.byteLength / TRIANGLE_STRIDE));
  const result = new Int32Array(safeCount * 3);
  for (let i = 0; i < safeCount; i++) {
    const base = i * TRIANGLE_STRIDE;
    const dst = i * 3;
    result[dst] = view.getUint32(base + 0, true);
    result[dst + 1] = view.getUint32(base + 4, true);
    result[dst + 2] = view.getUint32(base + 8, true);
  }
  return result;
}

function appendBatchTriangles(mesh: CPUMesh, edgeVertexIds: Map<number, number>, triangleIndices: Int32Array, triangleCount: number): void {
  for (let i = 0; i < triangleCount; i++) {
    const base = i * 3;
    const a = edgeVertexIds.get(triangleIndices[base]);
    const b = edgeVertexIds.get(triangleIndices[base + 1]);
    const c = edgeVertexIds.get(triangleIndices[base + 2]);
    if (a === undefined || b === undefined || c === undefined) {
      throw new Error('appendBatchTriangles(): missing edge vertex for emitted triangle');
    }
    mesh.addTriangle({ a, b, c });
  }
}

function exclusiveScanCounts(counts: Uint32Array): { offsets: Uint32Array; totalCount: number } {
  const offsets = new Uint32Array(counts.length);
  let running = 0;
  for (let i = 0; i < counts.length; i++) {
    offsets[i] = running;
    running += counts[i];
  }
  return { offsets, totalCount: running };
}

function xEdgeGlobalIndex(ix: number, iy: number, z: number, grid: GridInfo): number {
  return ix + grid.nx * (iy + (grid.ny + 1) * z);
}

function yEdgeGlobalIndex(ix: number, iy: number, z: number, grid: GridInfo): number {
  return grid.xEdgeCount + ix + (grid.nx + 1) * (iy + grid.ny * z);
}

function zEdgeGlobalIndex(ix: number, iy: number, z: number, grid: GridInfo): number {
  return grid.xEdgeCount + grid.yEdgeCount + ix + (grid.nx + 1) * (iy + (grid.ny + 1) * z);
}

function buildShaderBundle(solidWGSL: string, solidBindings: PreparedSolidBindings, workgroupSize: number): ShaderBundle {
  return {
    corner: buildCornerShader(solidWGSL, solidBindings, workgroupSize),
    edgeX: buildEdgeShader(solidWGSL, solidBindings, workgroupSize, 'x'),
    edgeY: buildEdgeShader(solidWGSL, solidBindings, workgroupSize, 'y'),
    edgeZ: buildEdgeShader(solidWGSL, solidBindings, workgroupSize, 'z'),
    count: buildCountShader(workgroupSize),
    emit: buildEmitShader(workgroupSize),
  };
}

function mcCommonPreludeWGSL(): string {
  return /* wgsl */`
struct MCParams {
  minCorner: vec3<f32>,
  delta: f32,
  nx: u32,
  ny: u32,
  nz: u32,
  bisectionSteps: u32,
  cornerRows: u32,
  cubeRows: u32,
  _pad0: u32,
  _pad1: u32,
};

struct BatchParams {
  zOffset: u32,
  cornerRingHead: u32,
  cubeRingHead: u32,
  localZStart: u32,
  localZCount: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
};

struct EdgeVertex {
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

@group(0) @binding(0) var<uniform> params: MCParams;
@group(0) @binding(1) var<uniform> batch: BatchParams;

fn cornerSlot(localZ: u32) -> u32 {
  return (batch.cornerRingHead + localZ) % params.cornerRows;
}

fn cubeSlot(localZ: u32) -> u32 {
  return (batch.cubeRingHead + localZ) % params.cubeRows;
}

fn globalCornerZ(localZ: u32) -> u32 {
  return batch.zOffset + localZ;
}

fn cornerIndexLocal(ix: u32, iy: u32, localZ: u32) -> u32 {
  let sx = params.nx + 1u;
  let sy = params.ny + 1u;
  return ix + sx * (iy + sy * cornerSlot(localZ));
}

fn xEdgeIndexLocal(ix: u32, iy: u32, localZ: u32) -> u32 {
  return ix + params.nx * (iy + (params.ny + 1u) * cornerSlot(localZ));
}

fn yEdgeIndexLocal(ix: u32, iy: u32, localZ: u32) -> u32 {
  return ix + (params.nx + 1u) * (iy + params.ny * cornerSlot(localZ));
}

fn zEdgeIndexLocal(ix: u32, iy: u32, localZ: u32) -> u32 {
  return ix + (params.nx + 1u) * (iy + (params.ny + 1u) * cubeSlot(localZ));
}

fn cornerPositionLocal(ix: u32, iy: u32, localZ: u32) -> vec3<f32> {
  return params.minCorner + vec3<f32>(f32(ix), f32(iy), f32(globalCornerZ(localZ))) * params.delta;
}
`;
}

function solidBindingWGSL(solidBindings: PreparedSolidBindings): string {
  return solidBindings.declarationWGSL;
}

function occupancyHelpersWGSL(): string {
  return /* wgsl */`
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

function occupancyHeaderWGSL(solidWGSL: string, solidBindings: PreparedSolidBindings): string {
  return /* wgsl */`
${mcCommonPreludeWGSL()}

${solidBindingWGSL(solidBindings)}

${transformSolidWGSL(solidWGSL, solidBindings)}

${occupancyHelpersWGSL()}
`;
}

function meshHelpersWGSL(): string {
  return /* wgsl */`
fn cubeCaseBits(ix: u32, iy: u32, localZ: u32, corners: ptr<function, array<u32, 8>>) -> u32 {
  (*corners)[0] = select(0u, 1u, cornersRead(cornerIndexLocal(ix, iy, localZ)));
  (*corners)[1] = select(0u, 1u, cornersRead(cornerIndexLocal(ix + 1u, iy, localZ)));
  (*corners)[2] = select(0u, 1u, cornersRead(cornerIndexLocal(ix, iy + 1u, localZ)));
  (*corners)[3] = select(0u, 1u, cornersRead(cornerIndexLocal(ix + 1u, iy + 1u, localZ)));
  (*corners)[4] = select(0u, 1u, cornersRead(cornerIndexLocal(ix, iy, localZ + 1u)));
  (*corners)[5] = select(0u, 1u, cornersRead(cornerIndexLocal(ix + 1u, iy, localZ + 1u)));
  (*corners)[6] = select(0u, 1u, cornersRead(cornerIndexLocal(ix, iy + 1u, localZ + 1u)));
  (*corners)[7] = select(0u, 1u, cornersRead(cornerIndexLocal(ix + 1u, iy + 1u, localZ + 1u)));

  var bits = 0u;
  for (var i = 0u; i < 8u; i++) {
    bits |= ((*corners)[i] << i);
  }
  return bits;
}

fn xEdgeGlobalIndex(ix: u32, iy: u32, z: u32) -> u32 {
  return ix + params.nx * (iy + (params.ny + 1u) * z);
}

fn yEdgeGlobalIndex(ix: u32, iy: u32, z: u32) -> u32 {
  let xCount = params.nx * (params.ny + 1u) * (params.nz + 1u);
  return xCount + ix + (params.nx + 1u) * (iy + params.ny * z);
}

fn zEdgeGlobalIndex(ix: u32, iy: u32, z: u32) -> u32 {
  let xCount = params.nx * (params.ny + 1u) * (params.nz + 1u);
  let yCount = (params.nx + 1u) * params.ny * (params.nz + 1u);
  return xCount + yCount + ix + (params.nx + 1u) * (iy + (params.ny + 1u) * z);
}

fn edgeIdFromCorners(ix: u32, iy: u32, localZ: u32, c0_: u32, c1_: u32) -> u32 {
  let z = globalCornerZ(localZ);
  let c0 = min(c0_, c1_);
  let c1 = max(c0_, c1_);

  if (c0 == 0u && c1 == 1u) { return xEdgeGlobalIndex(ix, iy, z); }
  if (c0 == 2u && c1 == 3u) { return xEdgeGlobalIndex(ix, iy + 1u, z); }
  if (c0 == 4u && c1 == 5u) { return xEdgeGlobalIndex(ix, iy, z + 1u); }
  if (c0 == 6u && c1 == 7u) { return xEdgeGlobalIndex(ix, iy + 1u, z + 1u); }

  if (c0 == 0u && c1 == 2u) { return yEdgeGlobalIndex(ix, iy, z); }
  if (c0 == 1u && c1 == 3u) { return yEdgeGlobalIndex(ix + 1u, iy, z); }
  if (c0 == 4u && c1 == 6u) { return yEdgeGlobalIndex(ix, iy, z + 1u); }
  if (c0 == 5u && c1 == 7u) { return yEdgeGlobalIndex(ix + 1u, iy, z + 1u); }

  if (c0 == 0u && c1 == 4u) { return zEdgeGlobalIndex(ix, iy, z); }
  if (c0 == 1u && c1 == 5u) { return zEdgeGlobalIndex(ix + 1u, iy, z); }
  if (c0 == 2u && c1 == 6u) { return zEdgeGlobalIndex(ix, iy + 1u, z); }
  return zEdgeGlobalIndex(ix + 1u, iy + 1u, z);
}
`;
}

function meshHeaderWGSL(): string {
  return /* wgsl */`
${mcCommonPreludeWGSL()}

${MC_LOOKUP_WGSL}

${meshHelpersWGSL()}
`;
}

function buildCornerShader(solidWGSL: string, solidBindings: PreparedSolidBindings, workgroupSize: number): string {
  return /* wgsl */`
${occupancyHeaderWGSL(solidWGSL, solidBindings)}
@group(0) @binding(2) var<storage, read_write> corners: array<u32>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let sx = params.nx + 1u;
  let sy = params.ny + 1u;
  let total = sx * sy * batch.localZCount;
  if (i >= total) { return; }
  let ix = i % sx;
  let t = i / sx;
  let iy = t % sy;
  let localZ = batch.localZStart + t / sy;
  corners[cornerIndexLocal(ix, iy, localZ)] = select(0u, 1u, solidOccupancy(cornerPositionLocal(ix, iy, localZ)));
}
`;
}

function buildEdgeShader(solidWGSL: string, solidBindings: PreparedSolidBindings, workgroupSize: number, axis: 'x' | 'y' | 'z'): string {
  const decode = axis === 'x'
    ? `
  let total = params.nx * (params.ny + 1u) * batch.localZCount;
  if (i >= total) { return; }
  let ix = i % params.nx;
  let t = i / params.nx;
  let iy = t % (params.ny + 1u);
  let localZ = batch.localZStart + t / (params.ny + 1u);
  let c0 = cornerIndexLocal(ix, iy, localZ);
  let c1 = cornerIndexLocal(ix + 1u, iy, localZ);
  let p0 = cornerPositionLocal(ix, iy, localZ);
  let p1 = cornerPositionLocal(ix + 1u, iy, localZ);
  let edgeIndex = xEdgeIndexLocal(ix, iy, localZ);
`
    : axis === 'y'
      ? `
  let total = (params.nx + 1u) * params.ny * batch.localZCount;
  if (i >= total) { return; }
  let ix = i % (params.nx + 1u);
  let t = i / (params.nx + 1u);
  let iy = t % params.ny;
  let localZ = batch.localZStart + t / params.ny;
  let c0 = cornerIndexLocal(ix, iy, localZ);
  let c1 = cornerIndexLocal(ix, iy + 1u, localZ);
  let p0 = cornerPositionLocal(ix, iy, localZ);
  let p1 = cornerPositionLocal(ix, iy + 1u, localZ);
  let edgeIndex = yEdgeIndexLocal(ix, iy, localZ);
`
      : `
  let total = (params.nx + 1u) * (params.ny + 1u) * batch.localZCount;
  if (i >= total) { return; }
  let ix = i % (params.nx + 1u);
  let t = i / (params.nx + 1u);
  let iy = t % (params.ny + 1u);
  let localZ = batch.localZStart + t / (params.ny + 1u);
  let c0 = cornerIndexLocal(ix, iy, localZ);
  let c1 = cornerIndexLocal(ix, iy, localZ + 1u);
  let p0 = cornerPositionLocal(ix, iy, localZ);
  let p1 = cornerPositionLocal(ix, iy, localZ + 1u);
  let edgeIndex = zEdgeIndexLocal(ix, iy, localZ);
`;

  return /* wgsl */`
${occupancyHeaderWGSL(solidWGSL, solidBindings)}
@group(0) @binding(2) var<storage, read> corners: array<u32>;
@group(0) @binding(3) var<storage, read_write> edges: array<EdgeVertex>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
${decode}
  let occ0 = corners[c0] != 0u;
  let occ1 = corners[c1] != 0u;
  if (occ0 == occ1) {
    edges[edgeIndex].isActive = 0u;
    edges[edgeIndex].pos = vec4<f32>(0.0);
    return;
  }
  let hit = bisectOccupancyEdge(p0, p1, occ0);
  edges[edgeIndex].isActive = 1u;
  edges[edgeIndex].pos = vec4<f32>(hit, 1.0);
}
`;
}

function buildCountShader(workgroupSize: number): string {
  return /* wgsl */`
${meshHeaderWGSL()}
@group(0) @binding(2) var<storage, read> corners: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;

fn cornersRead(index: u32) -> bool {
  return corners[index] != 0u;
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let total = params.nx * params.ny * batch.localZCount;
  if (i >= total) { return; }
  let ix = i % params.nx;
  let t = i / params.nx;
  let iy = t % params.ny;
  let localZ = batch.localZStart + t / params.ny;
  var cornerBits = array<u32, 8>();
  let caseBits = cubeCaseBits(ix, iy, localZ, &cornerBits);
  counts[i] = MC_TRIANGLE_COUNTS[caseBits];
}
`;
}

function buildEmitShader(workgroupSize: number): string {
  return /* wgsl */`
${meshHeaderWGSL()}
@group(0) @binding(2) var<storage, read> corners: array<u32>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> triangles: array<TriangleIndex>;

fn cornersRead(index: u32) -> bool {
  return corners[index] != 0u;
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let total = params.nx * params.ny * batch.localZCount;
  if (i >= total) { return; }
  let ix = i % params.nx;
  let t = i / params.nx;
  let iy = t % params.ny;
  let localZ = batch.localZStart + t / params.ny;
  var cornerBits = array<u32, 8>();
  let caseBits = cubeCaseBits(ix, iy, localZ, &cornerBits);
  let triCount = MC_TRIANGLE_COUNTS[caseBits];
  if (triCount == 0u) { return; }
  let triangleBase = offsets[i];
  let tableBase = caseBits * ${MC_TRIANGLE_DATA_STRIDE}u;
  for (var tri = 0u; tri < triCount; tri++) {
    let triOffset = tableBase + tri * 6u;
    let a = edgeIdFromCorners(ix, iy, localZ, MC_TRIANGLE_CORNERS[triOffset + 0u], MC_TRIANGLE_CORNERS[triOffset + 1u]);
    let b = edgeIdFromCorners(ix, iy, localZ, MC_TRIANGLE_CORNERS[triOffset + 2u], MC_TRIANGLE_CORNERS[triOffset + 3u]);
    let c = edgeIdFromCorners(ix, iy, localZ, MC_TRIANGLE_CORNERS[triOffset + 4u], MC_TRIANGLE_CORNERS[triOffset + 5u]);
    triangles[triangleBase + tri] = TriangleIndex(a, b, c, 0u);
  }
}
`;
}

function mcIntersections(corners: number[]): number {
  let result = 0;
  for (const corner of corners) {
    result |= (1 << corner);
  }
  return result;
}

function composeRotation(a: MCRotation, b: MCRotation): MCRotation {
  return [a[b[0]], a[b[1]], a[b[2]], a[b[3]], a[b[4]], a[b[5]], a[b[6]], a[b[7]]];
}

function applyTriangle(rotation: MCRotation, triangle: MCTriangle): MCTriangle {
  return [
    rotation[triangle[0]],
    rotation[triangle[1]],
    rotation[triangle[2]],
    rotation[triangle[3]],
    rotation[triangle[4]],
    rotation[triangle[5]],
  ];
}

function applyIntersections(rotation: MCRotation, bits: number): number {
  let result = 0;
  for (let c = 0; c < 8; c++) {
    if ((bits & (1 << c)) !== 0) {
      result |= (1 << rotation[c]);
    }
  }
  return result;
}

function allMCRotations(): MCRotation[] {
  const zRotation: MCRotation = [2, 0, 3, 1, 6, 4, 7, 5];
  const xRotation: MCRotation = [2, 3, 6, 7, 0, 1, 4, 5];
  const identity: MCRotation = [0, 1, 2, 3, 4, 5, 6, 7];
  const queue: MCRotation[] = [identity];
  const map = new Map<string, MCRotation>([[identity.join(','), identity]]);

  for (let index = 0; index < queue.length; index++) {
    const next = queue[index];
    for (const op of [zRotation, xRotation]) {
      const rotated = composeRotation(op, next);
      const key = rotated.join(',');
      if (map.has(key)) continue;
      map.set(key, rotated);
      queue.push(rotated);
    }
  }

  const result = [...map.values()];
  result.sort((a, b) => {
    for (let i = 0; i < a.length; i++) {
      if (a[i] < b[i]) return -1;
      if (a[i] > b[i]) return 1;
    }
    return 0;
  });
  return result;
}

function buildMCLookup(): { triangleCounts: Uint32Array; triangleCorners: Uint32Array } {
  const rotations = allMCRotations();
  const seen = new Uint8Array(256);
  const triangleCounts = new Uint32Array(256);
  const triangleCorners = new Uint32Array(256 * MC_TRIANGLE_DATA_STRIDE);

  for (const [baseBits, baseTriangles] of BASE_TRIANGLE_TABLE) {
    for (const rotation of rotations) {
      const bits = applyIntersections(rotation, baseBits);
      if (seen[bits] !== 0) continue;
      seen[bits] = 1;
      triangleCounts[bits] = baseTriangles.length;
      for (let triIndex = 0; triIndex < baseTriangles.length; triIndex++) {
        const triangle = applyTriangle(rotation, baseTriangles[triIndex]);
        const offset = bits * MC_TRIANGLE_DATA_STRIDE + triIndex * 6;
        for (let i = 0; i < 6; i++) {
          triangleCorners[offset + i] = triangle[i];
        }
      }
    }
  }

  for (let i = 0; i < 256; i++) {
    if (seen[i] === 0) {
      throw new Error(`Incomplete marching cubes lookup table; missing case ${i}.`);
    }
  }

  return { triangleCounts, triangleCorners };
}

function buildMCLookupWGSL(): string {
  const triangleCounts = Array.from(MC_LOOKUP.triangleCounts, (x) => `${x}u`).join(', ');
  const triangleCorners = Array.from(MC_LOOKUP.triangleCorners, (x) => `${x}u`).join(', ');
  return `const MC_TRIANGLE_COUNTS = array<u32, 256>(${triangleCounts});\nconst MC_TRIANGLE_CORNERS = array<u32, ${MC_LOOKUP.triangleCorners.length}>(${triangleCorners});`;
}

const BASE_TRIANGLE_TABLE: Array<[number, MCTriangle[]]> = [
  [mcIntersections([]), []],
  [mcIntersections([0]), [[0, 1, 0, 2, 0, 4]]],
  [mcIntersections([0, 1]), [
    [0, 4, 1, 5, 0, 2],
    [1, 5, 1, 3, 0, 2],
  ]],
  [mcIntersections([0, 5]), [
    [0, 1, 0, 2, 0, 4],
    [5, 7, 1, 5, 4, 5],
  ]],
  [mcIntersections([0, 7]), [
    [0, 1, 0, 2, 0, 4],
    [6, 7, 3, 7, 5, 7],
  ]],
  [mcIntersections([1, 2, 3]), [
    [0, 1, 1, 5, 0, 2],
    [0, 2, 1, 5, 2, 6],
    [2, 6, 1, 5, 3, 7],
  ]],
  [mcIntersections([0, 1, 7]), [
    [0, 4, 1, 5, 0, 2],
    [1, 5, 1, 3, 0, 2],
    [6, 7, 3, 7, 5, 7],
  ]],
  [mcIntersections([1, 4, 7]), [
    [4, 6, 4, 5, 0, 4],
    [1, 5, 1, 3, 0, 1],
    [6, 7, 3, 7, 5, 7],
  ]],
  [mcIntersections([0, 1, 2, 3]), [
    [0, 4, 1, 5, 3, 7],
    [0, 4, 3, 7, 2, 6],
  ]],
  [mcIntersections([0, 2, 3, 6]), [
    [0, 1, 4, 6, 0, 4],
    [0, 1, 6, 7, 4, 6],
    [0, 1, 1, 3, 6, 7],
    [1, 3, 3, 7, 6, 7],
  ]],
  [mcIntersections([1, 2, 5, 6]), [
    [0, 2, 2, 3, 6, 7],
    [0, 2, 6, 7, 4, 6],
    [0, 1, 4, 5, 5, 7],
    [5, 7, 1, 3, 0, 1],
  ]],
  [mcIntersections([0, 2, 3, 7]), [
    [0, 4, 0, 1, 2, 6],
    [0, 1, 5, 7, 2, 6],
    [2, 6, 5, 7, 6, 7],
    [0, 1, 1, 3, 5, 7],
  ]],
  [mcIntersections([1, 2, 3, 4]), [
    [0, 1, 1, 5, 0, 2],
    [0, 2, 1, 5, 2, 6],
    [2, 6, 1, 5, 3, 7],
    [4, 5, 0, 4, 4, 6],
  ]],
  [mcIntersections([1, 2, 4, 7]), [
    [0, 1, 1, 5, 1, 3],
    [0, 2, 2, 3, 2, 6],
    [4, 5, 0, 4, 4, 6],
    [5, 7, 6, 7, 3, 7],
  ]],
  [mcIntersections([1, 2, 3, 6]), [
    [0, 2, 0, 1, 4, 6],
    [0, 1, 3, 7, 4, 6],
    [0, 1, 1, 5, 3, 7],
    [4, 6, 3, 7, 6, 7],
  ]],
  [mcIntersections([0, 2, 3, 5, 6]), [
    [0, 1, 4, 6, 0, 4],
    [0, 1, 6, 7, 4, 6],
    [0, 1, 1, 3, 6, 7],
    [1, 3, 3, 7, 6, 7],
    [5, 7, 1, 5, 4, 5],
  ]],
  [mcIntersections([2, 3, 4, 5, 6]), [
    [5, 7, 1, 5, 0, 4],
    [0, 4, 6, 7, 5, 7],
    [0, 2, 6, 7, 0, 4],
    [0, 2, 3, 7, 6, 7],
    [0, 2, 1, 3, 3, 7],
  ]],
  [mcIntersections([0, 4, 5, 6, 7]), [
    [1, 5, 0, 1, 0, 2],
    [0, 2, 2, 6, 1, 5],
    [1, 5, 2, 6, 3, 7],
  ]],
  [mcIntersections([1, 2, 3, 4, 5, 6]), [
    [0, 2, 0, 1, 0, 4],
    [3, 7, 6, 7, 5, 7],
  ]],
  [mcIntersections([1, 2, 3, 4, 6, 7]), [
    [0, 2, 4, 5, 0, 4],
    [0, 2, 5, 7, 4, 5],
    [0, 2, 1, 5, 5, 7],
    [0, 1, 1, 5, 0, 2],
  ]],
  [mcIntersections([2, 3, 4, 5, 6, 7]), [
    [1, 5, 0, 4, 0, 2],
    [1, 3, 1, 5, 0, 2],
  ]],
  [mcIntersections([1, 2, 3, 4, 5, 6, 7]), [
    [0, 2, 0, 1, 0, 4],
  ]],
  [mcIntersections([0, 1, 2, 3, 4, 5, 6, 7]), []],
];

const MC_LOOKUP = buildMCLookup();
const MC_LOOKUP_WGSL = buildMCLookupWGSL();
