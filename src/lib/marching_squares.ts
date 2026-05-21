import {
  createSolidBindGroup,
  createSolidBindGroupLayout,
  prepareSolidBindings,
  transformSolidWGSL,
  type PreparedSolidBindings,
} from './solid_bindings';
import type { DualContourSolidBinding } from './dual_contouring';

const GPUBufferUsageAny: any = (globalThis as any).GPUBufferUsage;
const GPUMapModeAny: any = (globalThis as any).GPUMapMode;
const GPUShaderStageAny: any = (globalThis as any).GPUShaderStage;

const DEFAULT_MARCHING_SQUARES_BUFFER_SIZE = 1_000_000;
const STATIC_PARAMS_SIZE = 48;
const BATCH_PARAMS_SIZE = 32;
const EDGE_STRIDE = 32;
const SEGMENT_STRIDE = 8;
const MS_MAX_SEGMENTS_PER_CASE = 2;
const MS_SEGMENT_DATA_STRIDE = MS_MAX_SEGMENTS_PER_CASE * 4;

type Vec2 = [number, number];
type MSSegment = [number, number, number, number];

export interface IndexedSegments2D {
  /**
   * Packed XY positions. Vertex `i` lives at `positions[2*i + 0/1]`.
   */
  positions: Float32Array;
  /**
   * Packed segment indices. Segment `i` uses `indices[2*i + 0/1]`.
   * The endpoint order preserves the clockwise contour orientation used by model2d.MarchingSquares.
   */
  indices: Uint32Array;
}

export interface MarchingSquaresResult {
  mesh: IndexedSegments2D;
  metrics: MarchingSquaresMetrics;
}

export const exampleCircleSolidWGSL = /* wgsl */`
fn solidOccupancy(p: vec2<f32>) -> bool {
  let center = vec2<f32>(0.0, 0.0);
  let radius = 1.0;
  return distance(p, center) <= radius;
}
`;

export interface MarchingSquaresMetrics {
  totalMs: number;
  stages: MarchingSquaresStageTiming[];
}

export interface MarchingSquaresStageTiming {
  stage: string;
  ms: number;
}

export interface MarchingSquaresWebGPUOptions {
  device: any;
  /**
   * WGSL source that defines:
   *   fn solidOccupancy(p: vec2<f32>) -> bool
   * The function is inlined into occupancy-evaluation shaders.
   */
  solidWGSL: string;
  /**
   * Optional read-only buffers exposed to solidWGSL as @group(1) bindings.
   * Bindings are assigned in array order starting at 0.
   */
  solidBindings?: DualContourSolidBinding[];
  min: Vec2;
  max: Vec2;
  delta: number;
  /**
   * Number of midpoint-bisection steps to run for each edge crossing.
   * `0` matches model2d.MarchingSquares midpoint placement.
   */
  bisectionSteps?: number;
  bufferSize?: number;
  workgroupSize?: number;
  label?: string;
}

interface GridInfo {
  minCorner: Vec2;
  maxCorner: Vec2;
  delta: number;
  nx: number;
  ny: number;
  xEdgeCount: number;
  yEdgeCount: number;
}

interface BatchLayout {
  cornerRows: number;
  squareRows: number;
  cornerRowSize: number;
  xEdgeRowSize: number;
  yEdgeRowSize: number;
  squareRowSize: number;
}

interface DispatchRange {
  localYStart: number;
  localYCount: number;
}

interface BatchSpec {
  cornerFill: DispatchRange;
  squareFill: DispatchRange;
}

interface ShaderBundle {
  corner: string;
  edgeX: string;
  edgeY: string;
  count: string;
  emit: string;
}

export async function marchingSquaresWebGPU(options: MarchingSquaresWebGPUOptions): Promise<MarchingSquaresResult> {
  const timings: MarchingSquaresStageTiming[] = [];
  const startTime = performance.now();
  let stageStartTime = startTime;
  const markStage = (stage: string) => {
    const now = performance.now();
    timings.push({ stage, ms: now - stageStartTime });
    stageStartTime = now;
  };

  const config = normalizeOptions(options);
  const grid = createGridInfo(config.min, config.max, config.delta);
  const layout = createBatchLayout(grid, config.bufferSize);
  markStage('normalize options + grid');

  const device = config.device;
  const solidBindings = prepareSolidBindings(device, config.solidBindings ?? [], config.label, 'marchingSquaresWebGPU()');
  const workgroupSize = config.workgroupSize;
  const deviceLimits = device.limits as Record<string, number | undefined> | undefined;
  const maxStorageBindingSize = deviceLimits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const maxBufferSize = deviceLimits?.maxBufferSize ?? Number.POSITIVE_INFINITY;

  const cornerBufferSize = layout.cornerRowSize * layout.cornerRows * 4;
  const xEdgeBufferSize = layout.xEdgeRowSize * layout.cornerRows * EDGE_STRIDE;
  const yEdgeBufferSize = layout.yEdgeRowSize * layout.squareRows * EDGE_STRIDE;
  const squareCountBufferSize = layout.squareRowSize * layout.squareRows * 4;
  const xEdgeReadbackSize = xEdgeBufferSize;
  const yEdgeReadbackSize = yEdgeBufferSize;

  assertBufferFits('corner buffer', cornerBufferSize, maxStorageBindingSize, maxBufferSize, true);
  assertBufferFits('x edge buffer', xEdgeBufferSize, maxStorageBindingSize, maxBufferSize, true);
  assertBufferFits('y edge buffer', yEdgeBufferSize, maxStorageBindingSize, maxBufferSize, true);
  assertBufferFits('square segment count buffer', squareCountBufferSize, maxStorageBindingSize, maxBufferSize, true);
  assertBufferFits('square segment offset buffer', squareCountBufferSize, maxStorageBindingSize, maxBufferSize, true);
  assertBufferFits('x edge readback buffer', xEdgeReadbackSize, undefined, maxBufferSize, false);
  assertBufferFits('y edge readback buffer', yEdgeReadbackSize, undefined, maxBufferSize, false);
  assertBufferFits('square segment count readback buffer', squareCountBufferSize, undefined, maxBufferSize, false);

  const cornerBuffer = device.createBuffer({
    label: `${config.label}-ms-corners`,
    size: cornerBufferSize,
    usage: GPUBufferUsageAny.STORAGE,
  });
  const xEdgeBuffer = device.createBuffer({
    label: `${config.label}-ms-x-edges`,
    size: xEdgeBufferSize,
    usage: GPUBufferUsageAny.STORAGE | GPUBufferUsageAny.COPY_SRC,
  });
  const yEdgeBuffer = device.createBuffer({
    label: `${config.label}-ms-y-edges`,
    size: yEdgeBufferSize,
    usage: GPUBufferUsageAny.STORAGE | GPUBufferUsageAny.COPY_SRC,
  });
  const squareSegmentCountBuffer = device.createBuffer({
    label: `${config.label}-ms-segment-counts`,
    size: squareCountBufferSize,
    usage: GPUBufferUsageAny.STORAGE | GPUBufferUsageAny.COPY_SRC,
  });
  const squareSegmentOffsetBuffer = device.createBuffer({
    label: `${config.label}-ms-segment-offsets`,
    size: squareCountBufferSize,
    usage: GPUBufferUsageAny.STORAGE | GPUBufferUsageAny.COPY_DST,
  });
  markStage('create GPU buffers');

  const staticUniformBuffer = device.createBuffer({
    label: `${config.label}-ms-static-params`,
    size: STATIC_PARAMS_SIZE,
    usage: GPUBufferUsageAny.UNIFORM | GPUBufferUsageAny.COPY_DST,
  });
  const cornerRangeBuffer = device.createBuffer({
    label: `${config.label}-ms-corner-range`,
    size: BATCH_PARAMS_SIZE,
    usage: GPUBufferUsageAny.UNIFORM | GPUBufferUsageAny.COPY_DST,
  });
  const squareRangeBuffer = device.createBuffer({
    label: `${config.label}-ms-square-range`,
    size: BATCH_PARAMS_SIZE,
    usage: GPUBufferUsageAny.UNIFORM | GPUBufferUsageAny.COPY_DST,
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
  const [cornerModule, xEdgeModule, yEdgeModule, countModule, emitModule] = await Promise.all([
    createCheckedShaderModule(device, `${config.label}-ms-corner-shader`, shaders.corner),
    createCheckedShaderModule(device, `${config.label}-ms-x-edge-shader`, shaders.edgeX),
    createCheckedShaderModule(device, `${config.label}-ms-y-edge-shader`, shaders.edgeY),
    createCheckedShaderModule(device, `${config.label}-ms-count-shader`, shaders.count),
    createCheckedShaderModule(device, `${config.label}-ms-emit-shader`, shaders.emit),
  ]);
  const cornerPipeline = await device.createComputePipelineAsync({
    label: `${config.label}-ms-corner-pipeline`,
    layout: createPipelineLayout(device, cornerBindGroupLayout, solidBindGroupLayout),
    compute: { module: cornerModule, entryPoint: 'main' },
  });
  const xEdgePipeline = await device.createComputePipelineAsync({
    label: `${config.label}-ms-x-edge-pipeline`,
    layout: createPipelineLayout(device, edgeBindGroupLayout, solidBindGroupLayout),
    compute: { module: xEdgeModule, entryPoint: 'main' },
  });
  const yEdgePipeline = await device.createComputePipelineAsync({
    label: `${config.label}-ms-y-edge-pipeline`,
    layout: createPipelineLayout(device, edgeBindGroupLayout, solidBindGroupLayout),
    compute: { module: yEdgeModule, entryPoint: 'main' },
  });
  const countPipeline = await device.createComputePipelineAsync({
    label: `${config.label}-ms-count-pipeline`,
    layout: createPipelineLayout(device, countBindGroupLayout),
    compute: { module: countModule, entryPoint: 'main' },
  });
  const emitPipeline = await device.createComputePipelineAsync({
    label: `${config.label}-ms-emit-pipeline`,
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
      { binding: 1, resource: { buffer: squareRangeBuffer } },
      { binding: 2, resource: { buffer: cornerBuffer } },
      { binding: 3, resource: { buffer: yEdgeBuffer } },
    ],
  });
  const countBindGroup = device.createBindGroup({
    layout: countBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: staticUniformBuffer } },
      { binding: 1, resource: { buffer: squareRangeBuffer } },
      { binding: 2, resource: { buffer: cornerBuffer } },
      { binding: 3, resource: { buffer: squareSegmentCountBuffer } },
    ],
  });
  const solidBindGroup = solidBindGroupLayout === null ? null : createSolidBindGroup(device, solidBindGroupLayout, solidBindings);
  markStage('create bind groups');

  const xEdgeReadback = device.createBuffer({
    label: `${config.label}-ms-x-edge-readback`,
    size: xEdgeReadbackSize,
    usage: GPUBufferUsageAny.COPY_DST | GPUBufferUsageAny.MAP_READ,
  });
  const yEdgeReadback = device.createBuffer({
    label: `${config.label}-ms-y-edge-readback`,
    size: yEdgeReadbackSize,
    usage: GPUBufferUsageAny.COPY_DST | GPUBufferUsageAny.MAP_READ,
  });
  const squareCountReadback = device.createBuffer({
    label: `${config.label}-ms-count-readback`,
    size: squareCountBufferSize,
    usage: GPUBufferUsageAny.COPY_DST | GPUBufferUsageAny.MAP_READ,
  });
  markStage('create readback buffers');

  const positions: number[] = [];
  const indices: number[] = [];
  const edgeVertexIds = new Map<number, number>();
  let yOffset = 0;
  let cornerRingHead = 0;
  let squareRingHead = 0;
  let advanceIntoBatch = 0;
  let batchIndex = 0;

  for (;;) {
    const remainingAfterWindow = grid.ny + 1 - (layout.cornerRows + yOffset);
    const batchSpec = createBatchSpec(layout, advanceIntoBatch);
    const squareCount = batchSpec.squareFill.localYCount * layout.squareRowSize;

    device.queue.writeBuffer(cornerRangeBuffer, 0, packBatchUniforms(yOffset, cornerRingHead, squareRingHead, batchSpec.cornerFill));
    device.queue.writeBuffer(squareRangeBuffer, 0, packBatchUniforms(yOffset, cornerRingHead, squareRingHead, batchSpec.squareFill));

    const countEncoder = device.createCommandEncoder({ label: `${config.label}-ms-count-encoder-${batchIndex}` });
    {
      const pass = countEncoder.beginComputePass({ label: `${config.label}-ms-count-pass-${batchIndex}` });
      dispatch1D(pass, cornerPipeline, cornerBindGroup, batchSpec.cornerFill.localYCount * layout.cornerRowSize, workgroupSize, solidBindGroup);
      dispatch1D(pass, xEdgePipeline, xEdgeBindGroup, batchSpec.cornerFill.localYCount * layout.xEdgeRowSize, workgroupSize, solidBindGroup);
      dispatch1D(pass, yEdgePipeline, yEdgeBindGroup, batchSpec.squareFill.localYCount * layout.yEdgeRowSize, workgroupSize, solidBindGroup);
      dispatch1D(pass, countPipeline, countBindGroup, squareCount, workgroupSize);
      pass.end();
    }
    if (squareCount > 0) {
      countEncoder.copyBufferToBuffer(squareSegmentCountBuffer, 0, squareCountReadback, 0, squareCount * 4);
    }
    device.queue.submit([countEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    markStage(`batch ${batchIndex}: fill + count`);

    const countBytesRaw = squareCount > 0 ? await readBuffer(squareCountReadback) : new ArrayBuffer(0);
    const squareSegmentCounts = new Uint32Array(countBytesRaw.slice(0, squareCount * 4));
    const squareSegmentOffsets = exclusiveScanCounts(squareSegmentCounts);
    const segmentCount = squareSegmentOffsets.totalCount;
    const segmentBufferSize = Math.max(SEGMENT_STRIDE, segmentCount * SEGMENT_STRIDE);
    if (segmentBufferSize > maxStorageBindingSize) {
      throw new Error(
        `Segment output buffer requires ${segmentBufferSize} bytes, which exceeds the device storage-buffer binding limit of ${maxStorageBindingSize} bytes. ` +
        `Use a larger delta / smaller bounds, or reduce bufferSize.`,
      );
    }
    if (segmentBufferSize > maxBufferSize) {
      throw new Error(
        `Segment output buffer requires ${segmentBufferSize} bytes, which exceeds the device maxBufferSize of ${maxBufferSize} bytes. ` +
        `Use a larger delta or smaller bounds.`,
      );
    }
    markStage(`batch ${batchIndex}: decode counts`);

    if (squareSegmentOffsets.offsets.byteLength > 0) {
      device.queue.writeBuffer(squareSegmentOffsetBuffer, 0, squareSegmentOffsets.offsets);
    }

    const segmentBuffer = device.createBuffer({
      label: `${config.label}-ms-segments-${batchIndex}`,
      size: segmentBufferSize,
      usage: GPUBufferUsageAny.STORAGE | GPUBufferUsageAny.COPY_SRC,
    });
    const segmentReadback = device.createBuffer({
      label: `${config.label}-ms-segment-readback-${batchIndex}`,
      size: segmentBufferSize,
      usage: GPUBufferUsageAny.COPY_DST | GPUBufferUsageAny.MAP_READ,
    });
    const emitBindGroup = device.createBindGroup({
      layout: emitBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: staticUniformBuffer } },
        { binding: 1, resource: { buffer: squareRangeBuffer } },
        { binding: 2, resource: { buffer: cornerBuffer } },
        { binding: 3, resource: { buffer: squareSegmentOffsetBuffer } },
        { binding: 4, resource: { buffer: segmentBuffer } },
      ],
    });

    const emitEncoder = device.createCommandEncoder({ label: `${config.label}-ms-emit-encoder-${batchIndex}` });
    {
      const pass = emitEncoder.beginComputePass({ label: `${config.label}-ms-emit-pass-${batchIndex}` });
      dispatch1D(pass, emitPipeline, emitBindGroup, squareCount, workgroupSize);
      pass.end();
    }
    copyLogicalRowsToReadback(
      emitEncoder,
      xEdgeBuffer,
      xEdgeReadback,
      layout.xEdgeRowSize * EDGE_STRIDE,
      layout.cornerRows,
      cornerRingHead,
      batchSpec.squareFill.localYStart,
      batchSpec.squareFill.localYCount + 1,
    );
    copyLogicalRowsToReadback(
      emitEncoder,
      yEdgeBuffer,
      yEdgeReadback,
      layout.yEdgeRowSize * EDGE_STRIDE,
      layout.squareRows,
      squareRingHead,
      batchSpec.squareFill.localYStart,
      batchSpec.squareFill.localYCount,
    );
    emitEncoder.copyBufferToBuffer(segmentBuffer, 0, segmentReadback, 0, segmentBufferSize);
    device.queue.submit([emitEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    markStage(`batch ${batchIndex}: emit`);

    const [xEdgeBytesRaw, yEdgeBytesRaw, segmentBytesRaw] = await Promise.all([
      readBuffer(xEdgeReadback),
      readBuffer(yEdgeReadback),
      readBuffer(segmentReadback),
    ]);
    const xRowCount = batchSpec.squareFill.localYCount + 1;
    const yRowCount = batchSpec.squareFill.localYCount;
    const xEdgeBytes = xEdgeBytesRaw.slice(0, xRowCount * layout.xEdgeRowSize * EDGE_STRIDE);
    const yEdgeBytes = yEdgeBytesRaw.slice(0, yRowCount * layout.yEdgeRowSize * EDGE_STRIDE);
    const segmentBytes = segmentBytesRaw.slice(0, segmentCount * SEGMENT_STRIDE);

    appendBatchXEdgeVertices(positions, edgeVertexIds, xEdgeBytes, grid, yOffset + batchSpec.squareFill.localYStart, xRowCount);
    appendBatchYEdgeVertices(positions, edgeVertexIds, yEdgeBytes, grid, yOffset + batchSpec.squareFill.localYStart, yRowCount);
    appendBatchSegments(indices, edgeVertexIds, decodeSegments(segmentBytes, segmentCount), segmentCount);
    segmentBuffer.destroy();
    segmentReadback.destroy();
    markStage(`batch ${batchIndex}: CPU mesh assembly`);

    if (remainingAfterWindow === 0) {
      break;
    }
    advanceIntoBatch = Math.min(remainingAfterWindow, layout.squareRows);
    yOffset += advanceIntoBatch;
    cornerRingHead = (cornerRingHead + advanceIntoBatch) % layout.cornerRows;
    squareRingHead = (squareRingHead + advanceIntoBatch) % layout.squareRows;
    batchIndex++;
  }

  return {
    mesh: {
      positions: new Float32Array(positions),
      indices: new Uint32Array(indices),
    },
    metrics: {
      totalMs: performance.now() - startTime,
      stages: timings,
    },
  };
}

function normalizeOptions(options: MarchingSquaresWebGPUOptions): Required<Omit<MarchingSquaresWebGPUOptions, 'device' | 'solidWGSL' | 'solidBindings' | 'min' | 'max' | 'delta'>> & Pick<MarchingSquaresWebGPUOptions, 'device' | 'solidWGSL' | 'solidBindings' | 'min' | 'max' | 'delta'> {
  const bisectionSteps = options.bisectionSteps ?? 0;
  if (!Number.isInteger(bisectionSteps) || bisectionSteps < 0) {
    throw new Error('marchingSquaresWebGPU(): bisectionSteps must be an integer >= 0.');
  }
  const workgroupSize = options.workgroupSize ?? 256;
  if (!Number.isInteger(workgroupSize) || workgroupSize < 1) {
    throw new Error('marchingSquaresWebGPU(): workgroupSize must be an integer >= 1.');
  }
  return {
    device: options.device,
    solidWGSL: options.solidWGSL,
    solidBindings: options.solidBindings ?? [],
    min: options.min,
    max: options.max,
    delta: options.delta,
    bisectionSteps,
    bufferSize: options.bufferSize ?? DEFAULT_MARCHING_SQUARES_BUFFER_SIZE,
    workgroupSize,
    label: options.label ?? 'marching-squares-webgpu',
  };
}

function createGridInfo(min: Vec2, max: Vec2, delta: number): GridInfo {
  if (!Number.isFinite(min[0]) || !Number.isFinite(min[1]) || !Number.isFinite(max[0]) || !Number.isFinite(max[1])) {
    throw new Error('marchingSquaresWebGPU(): min and max must contain finite numbers.');
  }
  if (max[0] < min[0] || max[1] < min[1]) {
    throw new Error('marchingSquaresWebGPU(): max must be greater than or equal to min on every axis.');
  }
  if (!Number.isFinite(delta) || !(delta > 0)) {
    throw new Error('marchingSquaresWebGPU(): delta must be greater than 0.');
  }
  const minExpanded: Vec2 = [min[0] - delta, min[1] - delta];
  const maxExpanded: Vec2 = [max[0] + delta, max[1] + delta];
  const countX = Math.floor((maxExpanded[0] - minExpanded[0]) / delta + 1e-6) + 1;
  const countY = Math.floor((maxExpanded[1] - minExpanded[1]) / delta + 1e-6) + 1;
  if (countX < 2 || countY < 2) {
    throw new Error('marchingSquaresWebGPU(): invalid bounds; expanded grid must contain at least one square along each axis.');
  }
  const nx = countX - 1;
  const ny = countY - 1;
  return {
    minCorner: minExpanded,
    maxCorner: [minExpanded[0] + nx * delta, minExpanded[1] + ny * delta],
    delta,
    nx,
    ny,
    xEdgeCount: nx * (ny + 1),
    yEdgeCount: (nx + 1) * ny,
  };
}

function createBatchLayout(grid: GridInfo, bufferSize: number): BatchLayout {
  const cornerRowSize = grid.nx + 1;
  const totalCornerRows = grid.ny + 1;
  const cornerRows = Math.min(Math.max(Math.floor(bufferSize / cornerRowSize), 2), totalCornerRows);
  return {
    cornerRows,
    squareRows: cornerRows - 1,
    cornerRowSize,
    xEdgeRowSize: grid.nx,
    yEdgeRowSize: grid.nx + 1,
    squareRowSize: grid.nx,
  };
}

function createBatchSpec(layout: BatchLayout, advanceIntoBatch: number): BatchSpec {
  const cornerFill = advanceIntoBatch === 0
    ? { localYStart: 0, localYCount: layout.cornerRows }
    : { localYStart: layout.cornerRows - advanceIntoBatch, localYCount: advanceIntoBatch };
  const squareFill = advanceIntoBatch === 0
    ? { localYStart: 0, localYCount: layout.squareRows }
    : { localYStart: layout.squareRows - advanceIntoBatch, localYCount: advanceIntoBatch };
  return { cornerFill, squareFill };
}

function packStaticUniforms(grid: GridInfo, layout: BatchLayout, options: ReturnType<typeof normalizeOptions>): ArrayBuffer {
  const buffer = new ArrayBuffer(STATIC_PARAMS_SIZE);
  const f32 = new Float32Array(buffer);
  const u32 = new Uint32Array(buffer);
  f32[0] = grid.minCorner[0];
  f32[1] = grid.minCorner[1];
  f32[2] = grid.delta;
  u32[4] = grid.nx;
  u32[5] = grid.ny;
  u32[6] = options.bisectionSteps;
  u32[7] = layout.cornerRows;
  u32[8] = layout.squareRows;
  return buffer;
}

function packBatchUniforms(
  yOffset: number,
  cornerRingHead: number,
  squareRingHead: number,
  range: DispatchRange,
): ArrayBuffer {
  const buffer = new ArrayBuffer(BATCH_PARAMS_SIZE);
  const u32 = new Uint32Array(buffer);
  u32[0] = yOffset;
  u32[1] = cornerRingHead;
  u32[2] = squareRingHead;
  u32[3] = range.localYStart;
  u32[4] = range.localYCount;
  return buffer;
}

function dispatch1D(pass: any, pipeline: any, bindGroup: any, count: number, workgroupSize: number, solidBindGroup?: any): void {
  if (count <= 0) return;
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  if (solidBindGroup) {
    pass.setBindGroup(1, solidBindGroup);
  }
  pass.dispatchWorkgroups(Math.ceil(count / workgroupSize));
}

async function readBuffer(buffer: any): Promise<ArrayBuffer> {
  await buffer.mapAsync(GPUMapModeAny.READ);
  const src = buffer.getMappedRange();
  const out = src.slice(0);
  buffer.unmap();
  return out;
}

function createInternalBindGroupLayout(
  device: any,
  entries: Array<{ binding: number; type: 'uniform' | 'storage' | 'read-only-storage' }>,
): any {
  return device.createBindGroupLayout({
    entries: entries.map((entry) => ({
      binding: entry.binding,
      visibility: GPUShaderStageAny.COMPUTE,
      buffer: { type: entry.type },
    })),
  });
}

function createPipelineLayout(device: any, primaryLayout: any, solidBindGroupLayout?: any): any {
  return device.createPipelineLayout({
    bindGroupLayouts: solidBindGroupLayout ? [primaryLayout, solidBindGroupLayout] : [primaryLayout],
  });
}

async function createCheckedShaderModule(device: any, label: string, code: string): Promise<any> {
  const module = device.createShaderModule({ label, code });
  if (typeof module.getCompilationInfo !== 'function') {
    return module;
  }
  const info = await module.getCompilationInfo();
  const errors = (info.messages as Array<any>).filter((message) => message.type === 'error');
  if (errors.length === 0) {
    return module;
  }
  throw new Error(formatShaderCompilationError(label, errors));
}

function formatShaderCompilationError(label: string, errors: Array<any>): string {
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
  encoder: any,
  srcBuffer: any,
  dstBuffer: any,
  rowBytes: number,
  ringRows: number,
  ringHead: number,
  localYStart: number,
  rowCount: number,
): void {
  if (rowCount <= 0) return;
  let remaining = rowCount;
  let currentLocalY = localYStart;
  let dstOffset = 0;
  while (remaining > 0) {
    const physicalRow = (ringHead + currentLocalY) % ringRows;
    const segmentRows = Math.min(remaining, ringRows - physicalRow);
    encoder.copyBufferToBuffer(
      srcBuffer,
      physicalRow * rowBytes,
      dstBuffer,
      dstOffset,
      segmentRows * rowBytes,
    );
    currentLocalY += segmentRows;
    remaining -= segmentRows;
    dstOffset += segmentRows * rowBytes;
  }
}

function appendBatchXEdgeVertices(
  positions: number[],
  edgeVertexIds: Map<number, number>,
  bytes: ArrayBuffer,
  grid: GridInfo,
  globalYStart: number,
  rowCount: number,
): void {
  if (rowCount <= 0) return;
  const view = new DataView(bytes);
  for (let localY = 0; localY < rowCount; localY++) {
    const globalY = globalYStart + localY;
    for (let ix = 0; ix < grid.nx; ix++) {
      const localIndex = ix + grid.nx * localY;
      const base = localIndex * EDGE_STRIDE;
      if (view.getUint32(base + 0, true) === 0) continue;
      const edgeIndex = xEdgeGlobalIndex(ix, globalY, grid);
      if (edgeVertexIds.has(edgeIndex)) continue;
      positions.push(
        view.getFloat32(base + 16, true),
        view.getFloat32(base + 20, true),
      );
      edgeVertexIds.set(edgeIndex, positions.length / 2 - 1);
    }
  }
}

function appendBatchYEdgeVertices(
  positions: number[],
  edgeVertexIds: Map<number, number>,
  bytes: ArrayBuffer,
  grid: GridInfo,
  globalYStart: number,
  rowCount: number,
): void {
  if (rowCount <= 0) return;
  const view = new DataView(bytes);
  for (let localY = 0; localY < rowCount; localY++) {
    const globalY = globalYStart + localY;
    for (let ix = 0; ix <= grid.nx; ix++) {
      const localIndex = ix + (grid.nx + 1) * localY;
      const base = localIndex * EDGE_STRIDE;
      if (view.getUint32(base + 0, true) === 0) continue;
      const edgeIndex = yEdgeGlobalIndex(ix, globalY, grid);
      if (edgeVertexIds.has(edgeIndex)) continue;
      positions.push(
        view.getFloat32(base + 16, true),
        view.getFloat32(base + 20, true),
      );
      edgeVertexIds.set(edgeIndex, positions.length / 2 - 1);
    }
  }
}

function decodeSegments(bytes: ArrayBuffer, count: number): Uint32Array {
  const view = new DataView(bytes);
  const safeCount = Math.min(count, Math.floor(bytes.byteLength / SEGMENT_STRIDE));
  const result = new Uint32Array(safeCount * 2);
  for (let i = 0; i < safeCount; i++) {
    const base = i * SEGMENT_STRIDE;
    const dst = i * 2;
    result[dst] = view.getUint32(base + 0, true);
    result[dst + 1] = view.getUint32(base + 4, true);
  }
  return result;
}

function appendBatchSegments(indices: number[], edgeVertexIds: Map<number, number>, segmentIndices: Uint32Array, segmentCount: number): void {
  for (let i = 0; i < segmentCount; i++) {
    const base = i * 2;
    const a = edgeVertexIds.get(segmentIndices[base]);
    const b = edgeVertexIds.get(segmentIndices[base + 1]);
    if (a === undefined || b === undefined) {
      throw new Error('appendBatchSegments(): missing edge vertex for emitted segment');
    }
    indices.push(a, b);
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

function xEdgeGlobalIndex(ix: number, y: number, grid: GridInfo): number {
  return ix + grid.nx * y;
}

function yEdgeGlobalIndex(ix: number, y: number, grid: GridInfo): number {
  return grid.xEdgeCount + ix + (grid.nx + 1) * y;
}

function buildShaderBundle(solidWGSL: string, solidBindings: PreparedSolidBindings, workgroupSize: number): ShaderBundle {
  return {
    corner: buildCornerShader(solidWGSL, solidBindings, workgroupSize),
    edgeX: buildEdgeShader(solidWGSL, solidBindings, workgroupSize, 'x'),
    edgeY: buildEdgeShader(solidWGSL, solidBindings, workgroupSize, 'y'),
    count: buildCountShader(workgroupSize),
    emit: buildEmitShader(workgroupSize),
  };
}

function msCommonPreludeWGSL(): string {
  return /* wgsl */`
struct MSParams {
  minCorner: vec2<f32>,
  delta: f32,
  _pad0: f32,
  nx: u32,
  ny: u32,
  bisectionSteps: u32,
  cornerRows: u32,
  squareRows: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
};

struct BatchParams {
  yOffset: u32,
  cornerRingHead: u32,
  squareRingHead: u32,
  localYStart: u32,
  localYCount: u32,
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

struct SegmentIndex {
  a: u32,
  b: u32,
};

@group(0) @binding(0) var<uniform> params: MSParams;
@group(0) @binding(1) var<uniform> batch: BatchParams;

fn cornerSlot(localY: u32) -> u32 {
  return (batch.cornerRingHead + localY) % params.cornerRows;
}

fn squareSlot(localY: u32) -> u32 {
  return (batch.squareRingHead + localY) % params.squareRows;
}

fn globalCornerY(localY: u32) -> u32 {
  return batch.yOffset + localY;
}

fn cornerIndexLocal(ix: u32, localY: u32) -> u32 {
  return ix + (params.nx + 1u) * cornerSlot(localY);
}

fn xEdgeIndexLocal(ix: u32, localY: u32) -> u32 {
  return ix + params.nx * cornerSlot(localY);
}

fn yEdgeIndexLocal(ix: u32, localY: u32) -> u32 {
  return ix + (params.nx + 1u) * squareSlot(localY);
}

fn cornerPositionLocal(ix: u32, localY: u32) -> vec2<f32> {
  return params.minCorner + vec2<f32>(f32(ix), f32(globalCornerY(localY))) * params.delta;
}
`;
}

function solidBindingWGSL(solidBindings: PreparedSolidBindings): string {
  return solidBindings.declarationWGSL;
}

function occupancyHelpersWGSL(): string {
  return /* wgsl */`
fn bisectOccupancyEdge(p0: vec2<f32>, p1: vec2<f32>, occ0: bool) -> vec2<f32> {
  if (params.bisectionSteps == 0u) {
    return 0.5 * (p0 + p1);
  }
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
${msCommonPreludeWGSL()}

${solidBindingWGSL(solidBindings)}

${transformSolidWGSL(solidWGSL, solidBindings)}

${occupancyHelpersWGSL()}
`;
}

function meshHelpersWGSL(): string {
  return /* wgsl */`
fn squareCaseBits(ix: u32, localY: u32, corners: ptr<function, array<u32, 4>>) -> u32 {
  (*corners)[0] = select(0u, 1u, cornersRead(cornerIndexLocal(ix, localY)));
  (*corners)[1] = select(0u, 1u, cornersRead(cornerIndexLocal(ix + 1u, localY)));
  (*corners)[2] = select(0u, 1u, cornersRead(cornerIndexLocal(ix, localY + 1u)));
  (*corners)[3] = select(0u, 1u, cornersRead(cornerIndexLocal(ix + 1u, localY + 1u)));

  var bits = 0u;
  for (var i = 0u; i < 4u; i++) {
    bits |= ((*corners)[i] << i);
  }
  return bits;
}

fn xEdgeGlobalIndex(ix: u32, y: u32) -> u32 {
  return ix + params.nx * y;
}

fn yEdgeGlobalIndex(ix: u32, y: u32) -> u32 {
  let xCount = params.nx * (params.ny + 1u);
  return xCount + ix + (params.nx + 1u) * y;
}

fn edgeIdFromCorners(ix: u32, localY: u32, c0_: u32, c1_: u32) -> u32 {
  let y = globalCornerY(localY);
  let c0 = min(c0_, c1_);
  let c1 = max(c0_, c1_);

  if (c0 == 0u && c1 == 1u) { return xEdgeGlobalIndex(ix, y); }
  if (c0 == 2u && c1 == 3u) { return xEdgeGlobalIndex(ix, y + 1u); }
  if (c0 == 0u && c1 == 2u) { return yEdgeGlobalIndex(ix, y); }
  return yEdgeGlobalIndex(ix + 1u, y);
}
`;
}

function meshHeaderWGSL(): string {
  return /* wgsl */`
${msCommonPreludeWGSL()}

${MS_LOOKUP_WGSL}

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
  let total = sx * batch.localYCount;
  if (i >= total) { return; }
  let ix = i % sx;
  let localY = batch.localYStart + i / sx;
  corners[cornerIndexLocal(ix, localY)] = select(0u, 1u, solidOccupancy(cornerPositionLocal(ix, localY)));
}
`;
}

function buildEdgeShader(solidWGSL: string, solidBindings: PreparedSolidBindings, workgroupSize: number, axis: 'x' | 'y'): string {
  const decode = axis === 'x'
    ? `
  let total = params.nx * batch.localYCount;
  if (i >= total) { return; }
  let ix = i % params.nx;
  let localY = batch.localYStart + i / params.nx;
  let c0 = cornerIndexLocal(ix, localY);
  let c1 = cornerIndexLocal(ix + 1u, localY);
  let p0 = cornerPositionLocal(ix, localY);
  let p1 = cornerPositionLocal(ix + 1u, localY);
  let edgeIndex = xEdgeIndexLocal(ix, localY);
`
    : `
  let total = (params.nx + 1u) * batch.localYCount;
  if (i >= total) { return; }
  let ix = i % (params.nx + 1u);
  let localY = batch.localYStart + i / (params.nx + 1u);
  let c0 = cornerIndexLocal(ix, localY);
  let c1 = cornerIndexLocal(ix, localY + 1u);
  let p0 = cornerPositionLocal(ix, localY);
  let p1 = cornerPositionLocal(ix, localY + 1u);
  let edgeIndex = yEdgeIndexLocal(ix, localY);
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
  edges[edgeIndex].pos = vec4<f32>(hit, 0.0, 1.0);
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
  let total = params.nx * batch.localYCount;
  if (i >= total) { return; }
  let ix = i % params.nx;
  let localY = batch.localYStart + i / params.nx;
  var cornerBits = array<u32, 4>();
  let caseBits = squareCaseBits(ix, localY, &cornerBits);
  counts[i] = MS_SEGMENT_COUNTS[caseBits];
}
`;
}

function buildEmitShader(workgroupSize: number): string {
  return /* wgsl */`
${meshHeaderWGSL()}
@group(0) @binding(2) var<storage, read> corners: array<u32>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> segments: array<SegmentIndex>;

fn cornersRead(index: u32) -> bool {
  return corners[index] != 0u;
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let total = params.nx * batch.localYCount;
  if (i >= total) { return; }
  let ix = i % params.nx;
  let localY = batch.localYStart + i / params.nx;
  var cornerBits = array<u32, 4>();
  let caseBits = squareCaseBits(ix, localY, &cornerBits);
  let segCount = MS_SEGMENT_COUNTS[caseBits];
  if (segCount == 0u) { return; }
  let segmentBase = offsets[i];
  let tableBase = caseBits * ${MS_SEGMENT_DATA_STRIDE}u;
  for (var seg = 0u; seg < segCount; seg++) {
    let segOffset = tableBase + seg * 4u;
    let a = edgeIdFromCorners(ix, localY, MS_SEGMENT_CORNERS[segOffset + 0u], MS_SEGMENT_CORNERS[segOffset + 1u]);
    let b = edgeIdFromCorners(ix, localY, MS_SEGMENT_CORNERS[segOffset + 2u], MS_SEGMENT_CORNERS[segOffset + 3u]);
    segments[segmentBase + seg] = SegmentIndex(a, b);
  }
}
`;
}


function msIntersections(corners: number[]): number {
  let result = 0;
  for (const corner of corners) {
    result |= (1 << corner);
  }
  return result;
}

function buildMSLookup(): { segmentCounts: Uint32Array; segmentCorners: Uint32Array } {
  const mapping = new Map<number, MSSegment[]>([
    [msIntersections([]), []],
    [msIntersections([0]), [[0, 2, 0, 1]]],
    [msIntersections([1]), [[0, 1, 1, 3]]],
    [msIntersections([2]), [[2, 3, 0, 2]]],
    [msIntersections([3]), [[1, 3, 2, 3]]],
    [msIntersections([0, 1]), [[0, 2, 1, 3]]],
    [msIntersections([0, 2]), [[2, 3, 0, 1]]],
    [msIntersections([0, 3]), [[0, 2, 0, 1], [1, 3, 2, 3]]],
    [msIntersections([1, 2]), [[0, 1, 1, 3], [2, 3, 0, 2]]],
  ]);

  for (const [bits, segments] of Array.from(mapping.entries())) {
    const inverseBits = 0xf ^ bits;
    if (mapping.has(inverseBits)) continue;
    mapping.set(inverseBits, segments.map(([a, b, c, d]) => [c, d, a, b]));
  }

  if (mapping.size !== 16) {
    throw new Error(`Incomplete marching squares lookup table; found ${mapping.size} cases.`);
  }

  const segmentCounts = new Uint32Array(16);
  const segmentCorners = new Uint32Array(16 * MS_SEGMENT_DATA_STRIDE);
  for (let bits = 0; bits < 16; bits++) {
    const segments = mapping.get(bits);
    if (!segments) {
      throw new Error(`Incomplete marching squares lookup table; missing case ${bits}.`);
    }
    segmentCounts[bits] = segments.length;
    for (let segmentIndex = 0; segmentIndex < segments.length; segmentIndex++) {
      const offset = bits * MS_SEGMENT_DATA_STRIDE + segmentIndex * 4;
      const segment = segments[segmentIndex];
      for (let i = 0; i < 4; i++) {
        segmentCorners[offset + i] = segment[i];
      }
    }
  }
  return { segmentCounts, segmentCorners };
}

function buildMSLookupWGSL(): string {
  const segmentCounts = Array.from(MS_LOOKUP.segmentCounts, (x) => `${x}u`).join(', ');
  const segmentCorners = Array.from(MS_LOOKUP.segmentCorners, (x) => `${x}u`).join(', ');
  return `const MS_SEGMENT_COUNTS = array<u32, 16>(${segmentCounts});\nconst MS_SEGMENT_CORNERS = array<u32, ${MS_LOOKUP.segmentCorners.length}>(${segmentCorners});`;
}

const MS_LOOKUP = buildMSLookup();
const MS_LOOKUP_WGSL = buildMSLookupWGSL();
