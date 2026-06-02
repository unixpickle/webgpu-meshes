import { CPUMesh, type CPUTriangle, type IndexedMesh } from './mesh';
import { qefWGSL } from './qef_wgsl';
import {
  GPUBufferUsage,
  GPUMapMode,
  GPUShaderStage,
  type GPUBindGroup,
  type GPUBindGroupLayout,
  type GPUBuffer,
  type GPUBufferSource,
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
import {
  add,
  addScalar,
  clamp,
  clampVec3,
  dot,
  normalizeSafe,
  orthoBasis,
  scale,
  sub,
  type Vec3,
} from './vec3';


export interface DualContourResult {
  /** Mesh emitted directly from the GPU passes, before CPU repair. */
  initial: IndexedMesh;
  /** Mesh after duplicate resolution and singular edge / singular vertex repair on the CPU. */
  repaired: IndexedMesh;
  /** Timing metrics collected while generating and repairing the mesh. */
  metrics: DualContourMetrics;
}

export interface DualContourMetrics {
  totalMs: number;
  stages: DualContourStageTiming[];
}

export type TriangleMode = 'max-min-area' | 'sharpest' | 'flattest';
export type DualContourSolidBindingKind = 'uniform' | 'storage';

export interface DualContourSolidBinding {
  /**
   * WGSL variable name exposed to solidWGSL.
   */
  name: string;
  /**
   * Buffer address space. All user bindings are read-only in WGSL.
   */
  kind: DualContourSolidBindingKind;
  /**
   * WGSL type for the binding, e.g. `f32`, `MyParams`, or `array<vec4<f32>>`.
   */
  wgslType: string;
  /**
   * Optional WGSL type declarations required by wgslType.
   */
  wgslDefs?: string;
  /**
   * Typed array / ArrayBuffer data to upload, or an existing GPUBuffer.
   */
  source: GPUBufferSource | GPUBuffer;
  /**
   * Optional binding size in bytes when source is a GPUBuffer.
   */
  size?: number;
  /**
   * Optional label for an internally-created GPU buffer.
   */
  label?: string;
}

export interface DualContouringWebGPUOptions {
  device: GPUDevice;
  /**
   * WGSL source that defines:
   *   fn solidOccupancy(p: vec3<f32>) -> bool
   * The function is inlined into every compute shader.
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
  clip?: boolean;
  repair?: boolean;
  cubeMargin?: number;
  repairEpsilon?: number;
  singularValueEpsilon?: number;
  l2Penalty?: number;
  triangleMode?: TriangleMode;
  bisectionSteps?: number;
  normalStep?: number;
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
  cubeMarginWorld: number;
  repairEpsilonWorld: number;
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
  xyEmit: DispatchRange;
  zEmit: DispatchRange;
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

export interface DualContourStageTiming {
  stage: string;
  ms: number;
}

const DEFAULT_DUAL_CONTOUR_BUFFER_SIZE = 1_000_000;
const STATIC_PARAMS_SIZE = 64;
const BATCH_PARAMS_SIZE = 32;
const HERMITE_STRIDE = 48;
const CUBE_STRIDE = 32;
const TRIANGLE_STRIDE = 16;

export async function dualContourWebGPU(options: DualContouringWebGPUOptions): Promise<DualContourResult> {
  const timings: DualContourStageTiming[] = [];
  const startTime = performance.now();
  let stageStartTime = startTime;
  const markStage = (stage: string) => {
    const now = performance.now();
    timings.push({ stage, ms: now - stageStartTime });
    stageStartTime = now;
  };

  const config = normalizeOptions(options);
  const grid = createGridInfo(config.min, config.max, config.delta, config.noJitter, config.cubeMargin, config.repairEpsilon);
  const layout = createBatchLayout(grid, config.bufferSize);
  markStage('normalize options + grid');

  const device = config.device;
  const solidBindings = prepareSolidBindings(device, config.solidBindings ?? [], config.label, 'dualContourWebGPU()');
  const workgroupSize = config.workgroupSize;
  const deviceLimits = device.limits as Record<string, number | undefined> | undefined;
  const maxStorageBindingSize = deviceLimits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const maxBufferSize = deviceLimits?.maxBufferSize ?? Number.POSITIVE_INFINITY;

  const cornerBufferSize = layout.cornerPlaneSize * layout.cornerRows * 4;
  const xEdgeBufferSize = layout.xEdgePlaneSize * layout.cornerRows * HERMITE_STRIDE;
  const yEdgeBufferSize = layout.yEdgePlaneSize * layout.cornerRows * HERMITE_STRIDE;
  const zEdgeBufferSize = layout.zEdgePlaneSize * layout.cubeRows * HERMITE_STRIDE;
  const cubeBufferSize = layout.cubePlaneSize * layout.cubeRows * CUBE_STRIDE;
  const xCountBufferSize = layout.xEdgePlaneSize * layout.cornerRows * 4;
  const yCountBufferSize = layout.yEdgePlaneSize * layout.cornerRows * 4;
  const zCountBufferSize = layout.zEdgePlaneSize * layout.cubeRows * 4;
  const cubeReadbackSize = layout.cubePlaneSize * layout.cubeRows * CUBE_STRIDE;

  assertBufferFits('corner buffer', cornerBufferSize, maxStorageBindingSize, maxBufferSize, true);
  assertBufferFits('x edge buffer', xEdgeBufferSize, maxStorageBindingSize, maxBufferSize, true);
  assertBufferFits('y edge buffer', yEdgeBufferSize, maxStorageBindingSize, maxBufferSize, true);
  assertBufferFits('z edge buffer', zEdgeBufferSize, maxStorageBindingSize, maxBufferSize, true);
  assertBufferFits('cube buffer', cubeBufferSize, maxStorageBindingSize, maxBufferSize, true);
  assertBufferFits('x triangle count buffer', xCountBufferSize, maxStorageBindingSize, maxBufferSize, true);
  assertBufferFits('y triangle count buffer', yCountBufferSize, maxStorageBindingSize, maxBufferSize, true);
  assertBufferFits('z triangle count buffer', zCountBufferSize, maxStorageBindingSize, maxBufferSize, true);
  assertBufferFits('x triangle offset buffer', xCountBufferSize, maxStorageBindingSize, maxBufferSize, true);
  assertBufferFits('y triangle offset buffer', yCountBufferSize, maxStorageBindingSize, maxBufferSize, true);
  assertBufferFits('z triangle offset buffer', zCountBufferSize, maxStorageBindingSize, maxBufferSize, true);
  assertBufferFits('cube readback buffer', cubeReadbackSize, undefined, maxBufferSize, false);
  assertBufferFits('x triangle count readback buffer', xCountBufferSize, undefined, maxBufferSize, false);
  assertBufferFits('y triangle count readback buffer', yCountBufferSize, undefined, maxBufferSize, false);
  assertBufferFits('z triangle count readback buffer', zCountBufferSize, undefined, maxBufferSize, false);

  const cornerBuffer = device.createBuffer({
    label: `${config.label}-corners`,
    size: cornerBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const xEdgeBuffer = device.createBuffer({
    label: `${config.label}-x-edges`,
    size: xEdgeBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const yEdgeBuffer = device.createBuffer({
    label: `${config.label}-y-edges`,
    size: yEdgeBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const zEdgeBuffer = device.createBuffer({
    label: `${config.label}-z-edges`,
    size: zEdgeBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const cubeBuffer = device.createBuffer({
    label: `${config.label}-cubes`,
    size: cubeBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const xTriangleCountBuffer = device.createBuffer({
    label: `${config.label}-x-triangle-counts`,
    size: xCountBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const yTriangleCountBuffer = device.createBuffer({
    label: `${config.label}-y-triangle-counts`,
    size: yCountBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const zTriangleCountBuffer = device.createBuffer({
    label: `${config.label}-z-triangle-counts`,
    size: zCountBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const xTriangleOffsetBuffer = device.createBuffer({
    label: `${config.label}-x-triangle-offsets`,
    size: xCountBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const yTriangleOffsetBuffer = device.createBuffer({
    label: `${config.label}-y-triangle-offsets`,
    size: yCountBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const zTriangleOffsetBuffer = device.createBuffer({
    label: `${config.label}-z-triangle-offsets`,
    size: zCountBufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  markStage('create GPU buffers');

  const staticUniformBuffer = device.createBuffer({
    label: `${config.label}-static-params`,
    size: STATIC_PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const cornerRangeBuffer = device.createBuffer({
    label: `${config.label}-corner-range`,
    size: BATCH_PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const cubeRangeBuffer = device.createBuffer({
    label: `${config.label}-cube-range`,
    size: BATCH_PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const xyEmitRangeBuffer = device.createBuffer({
    label: `${config.label}-xy-emit-range`,
    size: BATCH_PARAMS_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const zEmitRangeBuffer = device.createBuffer({
    label: `${config.label}-z-emit-range`,
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
  const cubeBindGroupLayout = createInternalBindGroupLayout(device, [
    { binding: 0, type: 'uniform' },
    { binding: 1, type: 'uniform' },
    { binding: 2, type: 'read-only-storage' },
    { binding: 3, type: 'read-only-storage' },
    { binding: 4, type: 'read-only-storage' },
    { binding: 5, type: 'read-only-storage' },
    { binding: 6, type: 'storage' },
  ]);
  const countBindGroupLayout = createInternalBindGroupLayout(device, [
    { binding: 0, type: 'uniform' },
    { binding: 1, type: 'uniform' },
    { binding: 2, type: 'read-only-storage' },
    { binding: 3, type: 'read-only-storage' },
    { binding: 4, type: 'storage' },
  ]);
  const emitBindGroupLayout = createInternalBindGroupLayout(device, [
    { binding: 0, type: 'uniform' },
    { binding: 1, type: 'uniform' },
    { binding: 2, type: 'read-only-storage' },
    { binding: 3, type: 'read-only-storage' },
    { binding: 4, type: 'read-only-storage' },
    { binding: 5, type: 'read-only-storage' },
    { binding: 6, type: 'storage' },
  ]);
  const solidBindGroupLayout = solidBindings.bindings.length === 0 ? null : createSolidBindGroupLayout(device, solidBindings);

  const shaders = buildShaderBundle(config.solidWGSL, solidBindings, workgroupSize);
  const [
    cornerModule,
    xEdgeModule,
    yEdgeModule,
    zEdgeModule,
    cubeModule,
    countXModule,
    countYModule,
    countZModule,
    emitXModule,
    emitYModule,
    emitZModule,
  ] = await Promise.all([
    createCheckedShaderModule(device, `${config.label}-corner-shader`, shaders.corner),
    createCheckedShaderModule(device, `${config.label}-x-edge-shader`, shaders.edgeX),
    createCheckedShaderModule(device, `${config.label}-y-edge-shader`, shaders.edgeY),
    createCheckedShaderModule(device, `${config.label}-z-edge-shader`, shaders.edgeZ),
    createCheckedShaderModule(device, `${config.label}-cube-shader`, shaders.cube),
    createCheckedShaderModule(device, `${config.label}-count-x-shader`, shaders.countX),
    createCheckedShaderModule(device, `${config.label}-count-y-shader`, shaders.countY),
    createCheckedShaderModule(device, `${config.label}-count-z-shader`, shaders.countZ),
    createCheckedShaderModule(device, `${config.label}-emit-x-shader`, shaders.emitX),
    createCheckedShaderModule(device, `${config.label}-emit-y-shader`, shaders.emitY),
    createCheckedShaderModule(device, `${config.label}-emit-z-shader`, shaders.emitZ),
  ]);
  const cornerPipeline = await device.createComputePipelineAsync({
    label: `${config.label}-corner-pipeline`,
    layout: createPipelineLayout(device, cornerBindGroupLayout, solidBindGroupLayout),
    compute: { module: cornerModule, entryPoint: 'main' },
  });
  const xEdgePipeline = await device.createComputePipelineAsync({
    label: `${config.label}-x-edge-pipeline`,
    layout: createPipelineLayout(device, edgeBindGroupLayout, solidBindGroupLayout),
    compute: { module: xEdgeModule, entryPoint: 'main' },
  });
  const yEdgePipeline = await device.createComputePipelineAsync({
    label: `${config.label}-y-edge-pipeline`,
    layout: createPipelineLayout(device, edgeBindGroupLayout, solidBindGroupLayout),
    compute: { module: yEdgeModule, entryPoint: 'main' },
  });
  const zEdgePipeline = await device.createComputePipelineAsync({
    label: `${config.label}-z-edge-pipeline`,
    layout: createPipelineLayout(device, edgeBindGroupLayout, solidBindGroupLayout),
    compute: { module: zEdgeModule, entryPoint: 'main' },
  });
  const cubePipeline = await device.createComputePipelineAsync({
    label: `${config.label}-cube-pipeline`,
    layout: createPipelineLayout(device, cubeBindGroupLayout, solidBindGroupLayout),
    compute: { module: cubeModule, entryPoint: 'main' },
  });
  const countXPipeline = await device.createComputePipelineAsync({
    label: `${config.label}-count-x-pipeline`,
    layout: createPipelineLayout(device, countBindGroupLayout),
    compute: { module: countXModule, entryPoint: 'main' },
  });
  const countYPipeline = await device.createComputePipelineAsync({
    label: `${config.label}-count-y-pipeline`,
    layout: createPipelineLayout(device, countBindGroupLayout),
    compute: { module: countYModule, entryPoint: 'main' },
  });
  const countZPipeline = await device.createComputePipelineAsync({
    label: `${config.label}-count-z-pipeline`,
    layout: createPipelineLayout(device, countBindGroupLayout),
    compute: { module: countZModule, entryPoint: 'main' },
  });
  const emitXPipeline = await device.createComputePipelineAsync({
    label: `${config.label}-emit-x-pipeline`,
    layout: createPipelineLayout(device, emitBindGroupLayout),
    compute: { module: emitXModule, entryPoint: 'main' },
  });
  const emitYPipeline = await device.createComputePipelineAsync({
    label: `${config.label}-emit-y-pipeline`,
    layout: createPipelineLayout(device, emitBindGroupLayout),
    compute: { module: emitYModule, entryPoint: 'main' },
  });
  const emitZPipeline = await device.createComputePipelineAsync({
    label: `${config.label}-emit-z-pipeline`,
    layout: createPipelineLayout(device, emitBindGroupLayout),
    compute: { module: emitZModule, entryPoint: 'main' },
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
  const cubeBindGroup = device.createBindGroup({
    layout: cubeBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: staticUniformBuffer } },
      { binding: 1, resource: { buffer: cubeRangeBuffer } },
      { binding: 2, resource: { buffer: cornerBuffer } },
      { binding: 3, resource: { buffer: xEdgeBuffer } },
      { binding: 4, resource: { buffer: yEdgeBuffer } },
      { binding: 5, resource: { buffer: zEdgeBuffer } },
      { binding: 6, resource: { buffer: cubeBuffer } },
    ],
  });
  const countXBindGroup = device.createBindGroup({
    layout: countBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: staticUniformBuffer } },
      { binding: 1, resource: { buffer: xyEmitRangeBuffer } },
      { binding: 2, resource: { buffer: xEdgeBuffer } },
      { binding: 3, resource: { buffer: cubeBuffer } },
      { binding: 4, resource: { buffer: xTriangleCountBuffer } },
    ],
  });
  const countYBindGroup = device.createBindGroup({
    layout: countBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: staticUniformBuffer } },
      { binding: 1, resource: { buffer: xyEmitRangeBuffer } },
      { binding: 2, resource: { buffer: yEdgeBuffer } },
      { binding: 3, resource: { buffer: cubeBuffer } },
      { binding: 4, resource: { buffer: yTriangleCountBuffer } },
    ],
  });
  const countZBindGroup = device.createBindGroup({
    layout: countBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: staticUniformBuffer } },
      { binding: 1, resource: { buffer: zEmitRangeBuffer } },
      { binding: 2, resource: { buffer: zEdgeBuffer } },
      { binding: 3, resource: { buffer: cubeBuffer } },
      { binding: 4, resource: { buffer: zTriangleCountBuffer } },
    ],
  });
  const solidBindGroup = solidBindGroupLayout === null ? null : createSolidBindGroup(device, solidBindGroupLayout, solidBindings);
  markStage('create bind groups');

  const cubeReadback = device.createBuffer({
    label: `${config.label}-cube-readback`,
    size: cubeReadbackSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const xTriangleCountReadback = device.createBuffer({
    label: `${config.label}-x-triangle-counts-readback`,
    size: xCountBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const yTriangleCountReadback = device.createBuffer({
    label: `${config.label}-y-triangle-counts-readback`,
    size: yCountBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const zTriangleCountReadback = device.createBuffer({
    label: `${config.label}-z-triangle-counts-readback`,
    size: zCountBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  markStage('create readback buffers');

  const cpuMesh = new CPUMesh([], []);
  const cubeVertexIds = new Map<number, number>();
  let zOffset = 0;
  let cornerRingHead = 0;
  let cubeRingHead = 0;
  let advanceIntoBatch = 0;
  let batchIndex = 0;

  for (; ;) {
    const remainingAfterWindow = grid.nz + 1 - (layout.cornerRows + zOffset);
    const isFinalBatch = remainingAfterWindow === 0;
    const batchSpec = createBatchSpec(layout, advanceIntoBatch, isFinalBatch);
    const xEmitCount = batchSpec.xyEmit.localZCount * layout.xEdgePlaneSize;
    const yEmitCount = batchSpec.xyEmit.localZCount * layout.yEdgePlaneSize;
    const zEmitCount = batchSpec.zEmit.localZCount * layout.zEdgePlaneSize;

    device.queue.writeBuffer(cornerRangeBuffer, 0, packBatchUniforms(zOffset, cornerRingHead, cubeRingHead, batchSpec.cornerFill));
    device.queue.writeBuffer(cubeRangeBuffer, 0, packBatchUniforms(zOffset, cornerRingHead, cubeRingHead, batchSpec.cubeFill));
    device.queue.writeBuffer(xyEmitRangeBuffer, 0, packBatchUniforms(zOffset, cornerRingHead, cubeRingHead, batchSpec.xyEmit, isFinalBatch));
    device.queue.writeBuffer(zEmitRangeBuffer, 0, packBatchUniforms(zOffset, cornerRingHead, cubeRingHead, batchSpec.zEmit, isFinalBatch));

    const countEncoder = device.createCommandEncoder({ label: `${config.label}-count-encoder-${batchIndex}` });
    {
      const pass = countEncoder.beginComputePass({ label: `${config.label}-count-pass-${batchIndex}` });
      dispatch1D(pass, cornerPipeline, cornerBindGroup, batchSpec.cornerFill.localZCount * layout.cornerPlaneSize, workgroupSize, solidBindGroup);
      dispatch1D(pass, xEdgePipeline, xEdgeBindGroup, batchSpec.cornerFill.localZCount * layout.xEdgePlaneSize, workgroupSize, solidBindGroup);
      dispatch1D(pass, yEdgePipeline, yEdgeBindGroup, batchSpec.cornerFill.localZCount * layout.yEdgePlaneSize, workgroupSize, solidBindGroup);
      dispatch1D(pass, zEdgePipeline, zEdgeBindGroup, batchSpec.cubeFill.localZCount * layout.zEdgePlaneSize, workgroupSize, solidBindGroup);
      dispatch1D(pass, cubePipeline, cubeBindGroup, batchSpec.cubeFill.localZCount * layout.cubePlaneSize, workgroupSize, solidBindGroup);
      dispatch1D(pass, countXPipeline, countXBindGroup, xEmitCount, workgroupSize);
      dispatch1D(pass, countYPipeline, countYBindGroup, yEmitCount, workgroupSize);
      dispatch1D(pass, countZPipeline, countZBindGroup, zEmitCount, workgroupSize);
      pass.end();
    }
    if (xEmitCount > 0) {
      countEncoder.copyBufferToBuffer(xTriangleCountBuffer, 0, xTriangleCountReadback, 0, xEmitCount * 4);
    }
    if (yEmitCount > 0) {
      countEncoder.copyBufferToBuffer(yTriangleCountBuffer, 0, yTriangleCountReadback, 0, yEmitCount * 4);
    }
    if (zEmitCount > 0) {
      countEncoder.copyBufferToBuffer(zTriangleCountBuffer, 0, zTriangleCountReadback, 0, zEmitCount * 4);
    }
    device.queue.submit([countEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    markStage(`batch ${batchIndex}: fill + count`);

    const [xCountBytesRaw, yCountBytesRaw, zCountBytesRaw] = await Promise.all([
      xEmitCount > 0 ? readBuffer(xTriangleCountReadback) : Promise.resolve(new ArrayBuffer(0)),
      yEmitCount > 0 ? readBuffer(yTriangleCountReadback) : Promise.resolve(new ArrayBuffer(0)),
      zEmitCount > 0 ? readBuffer(zTriangleCountReadback) : Promise.resolve(new ArrayBuffer(0)),
    ]);
    const xTriangleCounts = new Uint32Array(xCountBytesRaw.slice(0, xEmitCount * 4));
    const yTriangleCounts = new Uint32Array(yCountBytesRaw.slice(0, yEmitCount * 4));
    const zTriangleCounts = new Uint32Array(zCountBytesRaw.slice(0, zEmitCount * 4));
    const xTriangleOffsets = exclusiveScanCounts(xTriangleCounts);
    const yTriangleOffsets = exclusiveScanCounts(yTriangleCounts, xTriangleOffsets.totalCount);
    const zTriangleOffsets = exclusiveScanCounts(zTriangleCounts, yTriangleOffsets.totalCount);
    const triangleCount = zTriangleOffsets.totalCount;
    const triangleBufferSize = Math.max(16, triangleCount * TRIANGLE_STRIDE);
    if (triangleBufferSize > maxStorageBindingSize) {
      throw new Error(
        `Triangle output buffer requires ${triangleBufferSize} bytes, which exceeds the device storage-buffer binding limit of ${maxStorageBindingSize} bytes. ` +
        `Use a larger delta / smaller bounds, or reduce bufferSize.`
      );
    }
    if (triangleBufferSize > maxBufferSize) {
      throw new Error(
        `Triangle output buffer requires ${triangleBufferSize} bytes, which exceeds the device maxBufferSize of ${maxBufferSize} bytes. ` +
        `Use a larger delta or smaller bounds.`
      );
    }
    markStage(`batch ${batchIndex}: decode counts`);

    if (xTriangleOffsets.offsets.byteLength > 0) {
      device.queue.writeBuffer(xTriangleOffsetBuffer, 0, xTriangleOffsets.offsets);
    }
    if (yTriangleOffsets.offsets.byteLength > 0) {
      device.queue.writeBuffer(yTriangleOffsetBuffer, 0, yTriangleOffsets.offsets);
    }
    if (zTriangleOffsets.offsets.byteLength > 0) {
      device.queue.writeBuffer(zTriangleOffsetBuffer, 0, zTriangleOffsets.offsets);
    }

    const triangleBuffer = device.createBuffer({
      label: `${config.label}-triangles-${batchIndex}`,
      size: triangleBufferSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const triangleReadback = device.createBuffer({
      label: `${config.label}-triangle-readback-${batchIndex}`,
      size: triangleBufferSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const emitXBindGroup = device.createBindGroup({
      layout: emitBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: staticUniformBuffer } },
        { binding: 1, resource: { buffer: xyEmitRangeBuffer } },
        { binding: 2, resource: { buffer: cornerBuffer } },
        { binding: 3, resource: { buffer: xEdgeBuffer } },
        { binding: 4, resource: { buffer: cubeBuffer } },
        { binding: 5, resource: { buffer: xTriangleOffsetBuffer } },
        { binding: 6, resource: { buffer: triangleBuffer } },
      ],
    });
    const emitYBindGroup = device.createBindGroup({
      layout: emitBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: staticUniformBuffer } },
        { binding: 1, resource: { buffer: xyEmitRangeBuffer } },
        { binding: 2, resource: { buffer: cornerBuffer } },
        { binding: 3, resource: { buffer: yEdgeBuffer } },
        { binding: 4, resource: { buffer: cubeBuffer } },
        { binding: 5, resource: { buffer: yTriangleOffsetBuffer } },
        { binding: 6, resource: { buffer: triangleBuffer } },
      ],
    });
    const emitZBindGroup = device.createBindGroup({
      layout: emitBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: staticUniformBuffer } },
        { binding: 1, resource: { buffer: zEmitRangeBuffer } },
        { binding: 2, resource: { buffer: cornerBuffer } },
        { binding: 3, resource: { buffer: zEdgeBuffer } },
        { binding: 4, resource: { buffer: cubeBuffer } },
        { binding: 5, resource: { buffer: zTriangleOffsetBuffer } },
        { binding: 6, resource: { buffer: triangleBuffer } },
      ],
    });

    const emitEncoder = device.createCommandEncoder({ label: `${config.label}-emit-encoder-${batchIndex}` });
    {
      const pass = emitEncoder.beginComputePass({ label: `${config.label}-emit-pass-${batchIndex}` });
      dispatch1D(pass, emitXPipeline, emitXBindGroup, xEmitCount, workgroupSize);
      dispatch1D(pass, emitYPipeline, emitYBindGroup, yEmitCount, workgroupSize);
      dispatch1D(pass, emitZPipeline, emitZBindGroup, zEmitCount, workgroupSize);
      pass.end();
    }
    copyLogicalCubeRowsToReadback(
      emitEncoder,
      cubeBuffer,
      cubeReadback,
      layout,
      cubeRingHead,
      batchSpec.cubeFill.localZStart,
      batchSpec.cubeFill.localZCount,
    );
    emitEncoder.copyBufferToBuffer(triangleBuffer, 0, triangleReadback, 0, triangleBufferSize);
    device.queue.submit([emitEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    markStage(`batch ${batchIndex}: emit`);

    const [cubeBytesRaw, triangleBytesRaw] = await Promise.all([
      batchSpec.cubeFill.localZCount > 0 ? readBuffer(cubeReadback) : Promise.resolve(new ArrayBuffer(0)),
      readBuffer(triangleReadback),
    ]);
    const cubeBytes = cubeBytesRaw.slice(0, batchSpec.cubeFill.localZCount * layout.cubePlaneSize * CUBE_STRIDE);
    const triangleBytes = triangleBytesRaw.slice(0, triangleCount * TRIANGLE_STRIDE);
    appendBatchCubeVertices(
      cpuMesh,
      cubeVertexIds,
      cubeBytes,
      grid,
      zOffset + batchSpec.cubeFill.localZStart,
      batchSpec.cubeFill.localZCount,
    );
    appendBatchTriangles(cpuMesh, cubeVertexIds, decodeTriangles(triangleBytes, triangleCount), triangleCount);
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

  const initial = cpuMesh.compact();
  markStage('compact initial mesh');

  if (config.repair) {
    resolveDuplicateOriginalVertices(cpuMesh, grid);
    markStage('repair: resolve duplicates');
    const edgeRepairEpsilon = grid.repairEpsilonWorld * 0.49;
    repairSingularEdges(cpuMesh, grid, edgeRepairEpsilon, config.clip);
    markStage('repair: singular edges');
    repairSingularVertices(cpuMesh, grid, edgeRepairEpsilon, config.clip);
    markStage('repair: singular vertices');
  } else {
    markStage('repair: skipped');
  }

  const repaired = cpuMesh.compact();
  markStage('compact repaired mesh');
  return {
    initial,
    repaired,
    metrics: {
      totalMs: performance.now() - startTime,
      stages: timings,
    },
  };
}

function normalizeOptions(options: DualContouringWebGPUOptions): Required<Omit<DualContouringWebGPUOptions, 'device' | 'solidWGSL' | 'solidBindings' | 'min' | 'max' | 'delta'>> & Pick<DualContouringWebGPUOptions, 'device' | 'solidWGSL' | 'solidBindings' | 'min' | 'max' | 'delta'> {
  return {
    device: options.device,
    solidWGSL: options.solidWGSL,
    solidBindings: options.solidBindings ?? [],
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
    bufferSize: options.bufferSize ?? DEFAULT_DUAL_CONTOUR_BUFFER_SIZE,
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

function createBatchSpec(layout: BatchLayout, advanceIntoBatch: number, isFinalBatch: boolean): BatchSpec {
  const cornerFill = advanceIntoBatch === 0
    ? { localZStart: 0, localZCount: layout.cornerRows }
    : { localZStart: layout.cornerRows - advanceIntoBatch, localZCount: advanceIntoBatch };
  const cubeFill = advanceIntoBatch === 0
    ? { localZStart: 0, localZCount: layout.cubeRows }
    : { localZStart: layout.cubeRows - advanceIntoBatch, localZCount: advanceIntoBatch };
  const xyEmitStart = advanceIntoBatch === 0 ? 0 : layout.cornerRows - advanceIntoBatch - 1;
  const xyEmitEnd = isFinalBatch ? layout.cornerRows : layout.cornerRows - 1;
  const zEmitStart = advanceIntoBatch === 0 ? 0 : layout.cubeRows - advanceIntoBatch;
  const zEmitCount = advanceIntoBatch === 0 ? layout.cubeRows : advanceIntoBatch;
  return {
    cornerFill,
    cubeFill,
    xyEmit: {
      localZStart: xyEmitStart,
      localZCount: Math.max(0, xyEmitEnd - xyEmitStart),
    },
    zEmit: {
      localZStart: zEmitStart,
      localZCount: zEmitCount,
    },
  };
}

function packStaticUniforms(grid: GridInfo, layout: BatchLayout, options: ReturnType<typeof normalizeOptions>): ArrayBuffer {
  const buffer = new ArrayBuffer(STATIC_PARAMS_SIZE);
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
  u32[14] = layout.cornerRows;
  u32[15] = layout.cubeRows;
  return buffer;
}

function packBatchUniforms(
  zOffset: number,
  cornerRingHead: number,
  cubeRingHead: number,
  range: DispatchRange,
  isFinalBatch = false,
): ArrayBuffer {
  const buffer = new ArrayBuffer(BATCH_PARAMS_SIZE);
  const u32 = new Uint32Array(buffer);
  u32[0] = zOffset;
  u32[1] = cornerRingHead;
  u32[2] = cubeRingHead;
  u32[3] = range.localZStart;
  u32[4] = range.localZCount;
  u32[5] = isFinalBatch ? 1 : 0;
  return buffer;
}

function triangleModeToInt(mode: TriangleMode): number {
  switch (mode) {
    case 'max-min-area': return 0;
    case 'sharpest': return 1;
    case 'flattest': return 2;
  }
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

function copyLogicalCubeRowsToReadback(
  encoder: GPUCommandEncoder,
  cubeBuffer: GPUBuffer,
  cubeReadback: GPUBuffer,
  layout: BatchLayout,
  ringHead: number,
  localZStart: number,
  localZCount: number,
): void {
  if (localZCount <= 0) return;
  let remaining = localZCount;
  let currentLocalZ = localZStart;
  let dstOffset = 0;
  const rowBytes = layout.cubePlaneSize * CUBE_STRIDE;
  while (remaining > 0) {
    const physicalRow = (ringHead + currentLocalZ) % layout.cubeRows;
    const segmentRows = Math.min(remaining, layout.cubeRows - physicalRow);
    encoder.copyBufferToBuffer(
      cubeBuffer,
      physicalRow * rowBytes,
      cubeReadback,
      dstOffset,
      segmentRows * rowBytes,
    );
    currentLocalZ += segmentRows;
    remaining -= segmentRows;
    dstOffset += segmentRows * rowBytes;
  }
}

function appendBatchCubeVertices(
  mesh: CPUMesh,
  cubeVertexIds: Map<number, number>,
  bytes: ArrayBuffer,
  grid: GridInfo,
  globalZStart: number,
  rowCount: number,
): void {
  if (rowCount <= 0) return;
  const view = new DataView(bytes);
  for (let localZ = 0; localZ < rowCount; localZ++) {
    const globalZ = globalZStart + localZ;
    for (let iy = 0; iy < grid.ny; iy++) {
      for (let ix = 0; ix < grid.nx; ix++) {
        const localIndex = ix + grid.nx * (iy + grid.ny * localZ);
        const base = localIndex * CUBE_STRIDE;
        if (view.getUint32(base + 0, true) === 0) continue;
        const cubeIndex = ix + grid.nx * (iy + grid.ny * globalZ);
        if (cubeVertexIds.has(cubeIndex)) continue;
        const vertexId = mesh.addVertex({
          position: [
            view.getFloat32(base + 16, true),
            view.getFloat32(base + 20, true),
            view.getFloat32(base + 24, true),
          ],
          cubeIndex,
          original: true,
        });
        cubeVertexIds.set(cubeIndex, vertexId);
      }
    }
  }
}

function decodeTriangles(bytes: ArrayBuffer, count: number): Int32Array {
  const view = new DataView(bytes);
  const safeCount = Math.min(count, Math.floor(bytes.byteLength / 16));
  const result = new Int32Array(safeCount * 3);
  for (let i = 0; i < safeCount; i++) {
    const base = i * 16;
    const dst = i * 3;
    result[dst] = view.getUint32(base + 0, true);
    result[dst + 1] = view.getUint32(base + 4, true);
    result[dst + 2] = view.getUint32(base + 8, true);
  }
  return result;
}

function appendBatchTriangles(mesh: CPUMesh, cubeVertexIds: Map<number, number>, triangleIndices: Int32Array, triangleCount: number): void {
  for (let i = 0; i < triangleCount; i++) {
    const base = i * 3;
    const aCube = triangleIndices[base];
    const bCube = triangleIndices[base + 1];
    const cCube = triangleIndices[base + 2];
    const a = cubeVertexIds.get(aCube);
    const b = cubeVertexIds.get(bCube);
    const c = cubeVertexIds.get(cCube);
    if (a === undefined || b === undefined || c === undefined) {
      throw new Error('appendBatchTriangles(): missing cube vertex for emitted triangle');
    }
    mesh.addTriangle({ a, b, c });
  }
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
  const groups = new Map<number, Map<number, Map<number, number[]>>>();
  for (let i = 0; i < mesh.vertexCount; i++) {
    if (!mesh.vertexOriginal(i)) continue;
    const cubeIndex = mesh.vertexCubeIndex(i);
    if (cubeIndex === null) continue;
    pushPositionBitGroup(groups, mesh, i, i);
  }

  for (const ids of positionBitGroupValues(groups)) {
    for (let k = 0; k < ids.length; k++) {
      const vid = ids[k];
      const cubeIndex = mesh.vertexCubeIndex(vid);
      if (cubeIndex === null) continue;
      const [min, max] = cubeBoundsFromIndex(cubeIndex, grid);
      mesh.setVertexPosition(vid, clampVec3(mesh.vertexPosition(vid), addScalar(min, grid.cubeMarginWorld), addScalar(max, -grid.cubeMarginWorld)));
    }

    const remaining = new Map<number, Map<number, Map<number, number[]>>>();
    for (const vid of ids) {
      pushPositionBitGroup(remaining, mesh, vid, vid);
    }

    for (const same of positionBitGroupValues(remaining)) {
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
        if (mesh.triangleExists(ta)) {
          toClamp.add(mesh.triangleA(ta));
          toClamp.add(mesh.triangleB(ta));
          toClamp.add(mesh.triangleC(ta));
        }
        if (mesh.triangleExists(tb)) {
          toClamp.add(mesh.triangleA(tb));
          toClamp.add(mesh.triangleB(tb));
          toClamp.add(mesh.triangleC(tb));
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
        if (!mesh.triangleExists(triId)) continue;
        const other = thirdVertexOfTriangleId(mesh, triId, refreshed.u, refreshed.v);
        if (other < 0) continue;
        let t1: CPUTriangle = { a: other, b: refreshed.u, c: newVertexId };
        let t2: CPUTriangle = { a: other, b: newVertexId, c: refreshed.v };
        const originalOrientation = segmentOrientationById(mesh, triId, other, refreshed.u);
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
          if (!mesh.triangleExists(triId)) continue;
          toClamp.add(mesh.triangleA(triId));
          toClamp.add(mesh.triangleB(triId));
          toClamp.add(mesh.triangleC(triId));
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
        if (!mesh.triangleExists(triId)) continue;
        const [o1, o2] = otherSegmentOfTriangleById(mesh, triId, v);
        const p1 = mesh.vertexPosition(o1);
        const p2 = mesh.vertexPosition(o2);
        const dotv = clamp(dot(normalizeSafe(sub(p1, center), [1, 0, 0]), normalizeSafe(sub(p2, center), [0, 1, 0])), -1.0, 1.0);
        const theta = Math.acos(dotv);
        dir = add(dir, scale(triangleNormalById(mesh, triId), -theta));
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
    if (!mesh.triangleExists(i)) continue;
    const a = mesh.triangleA(i);
    const b = mesh.triangleB(i);
    const c = mesh.triangleC(i);
    appendEdgeRecord(edgeMap, a, b, i);
    appendEdgeRecord(edgeMap, b, c, i);
    appendEdgeRecord(edgeMap, c, a, i);
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
    const other = thirdVertexOfTriangleId(mesh, triId, u, v);
    const triVec = normalizeSafe(sub(mesh.vertexPosition(other), midpoint), b1);
    const x = dot(b1, triVec);
    const y = dot(b2, triVec);
    const theta = Math.atan2(y, x);
    const normal = triangleNormalById(mesh, triId);
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
    const triAId = sortable[i].triId;
    for (let j = i + 1; j < sortable.length; j++) {
      if (segmentOrientationById(mesh, triAId, u, v) !== segmentOrientationById(mesh, sortable[j].triId, u, v)) {
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
    if (!mesh.triangleExists(i)) continue;
    pushMapArray(vertexToTriangles, mesh.triangleA(i), i);
    pushMapArray(vertexToTriangles, mesh.triangleB(i), i);
    pushMapArray(vertexToTriangles, mesh.triangleC(i), i);
  }

  const result: SingularVertexGroup[] = [];
  for (const [vertexId, triIndices] of vertexToTriangles.entries()) {
    if (triIndices.length < 2) continue;
    const dsu = new DisjointSet(triIndices.length);
    const edgeAroundVertex = new Map<number, number[]>();

    for (let localIndex = 0; localIndex < triIndices.length; localIndex++) {
      const triId = triIndices[localIndex];
      if (!mesh.triangleExists(triId)) continue;
      const [o1, o2] = otherSegmentOfTriangleById(mesh, triId, vertexId);
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

function thirdVertexOfTriangleId(mesh: CPUMesh, triId: number, u: number, v: number): number {
  const a = mesh.triangleA(triId);
  const b = mesh.triangleB(triId);
  const c = mesh.triangleC(triId);
  if (a !== u && a !== v) return a;
  if (b !== u && b !== v) return b;
  if (c !== u && c !== v) return c;
  return -1;
}

function otherSegmentOfTriangleById(mesh: CPUMesh, triId: number, v: number): [number, number] {
  const a = mesh.triangleA(triId);
  const b = mesh.triangleB(triId);
  const c = mesh.triangleC(triId);
  if (a === v) return [b, c];
  if (b === v) return [c, a];
  if (c === v) return [a, b];
  throw new Error('vertex is not part of triangle');
}

function segmentOrientationById(mesh: CPUMesh, triId: number, u: number, v: number): boolean {
  const a = mesh.triangleA(triId);
  const b = mesh.triangleB(triId);
  const c = mesh.triangleC(triId);
  if (a === u) return c === v;
  if (b === u) return a === v;
  if (c === u) return b === v;
  throw new Error('segmentOrientation(): first edge endpoint not present in triangle');
}

function segmentOrientation(tri: CPUTriangle, u: number, v: number): boolean {
  const verts = [tri.a, tri.b, tri.c];
  for (let i = 0; i < 3; i++) {
    if (verts[i] === u) return verts[(i + 2) % 3] === v;
  }
  throw new Error('segmentOrientation(): first edge endpoint not present in triangle');
}

function triangleNormalById(mesh: CPUMesh, triId: number): Vec3 {
  const a = mesh.triangleA(triId);
  const b = mesh.triangleB(triId);
  const c = mesh.triangleC(triId);
  return triangleNormalFromIds(mesh, a, b, c);
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

function triangleNormalFromIds(mesh: CPUMesh, aId: number, bId: number, cId: number): Vec3 {
  const ax = mesh.vertexX(aId);
  const ay = mesh.vertexY(aId);
  const az = mesh.vertexZ(aId);
  const abx = mesh.vertexX(bId) - ax;
  const aby = mesh.vertexY(bId) - ay;
  const abz = mesh.vertexZ(bId) - az;
  const acx = mesh.vertexX(cId) - ax;
  const acy = mesh.vertexY(cId) - ay;
  const acz = mesh.vertexZ(cId) - az;
  const nx = aby * acz - abz * acy;
  const ny = abz * acx - abx * acz;
  const nz = abx * acy - aby * acx;
  const norm2 = nx * nx + ny * ny + nz * nz;
  if (norm2 <= 1e-20) {
    return [1, 0, 0];
  }
  const invNorm = 1 / Math.sqrt(norm2);
  return [nx * invNorm, ny * invNorm, nz * invNorm];
}

function pushPositionBitGroup(
  groups: Map<number, Map<number, Map<number, number[]>>>,
  mesh: CPUMesh,
  vertexId: number,
  value: number,
): void {
  const xBits = float32Bits(mesh.vertexX(vertexId));
  const yBits = float32Bits(mesh.vertexY(vertexId));
  const zBits = float32Bits(mesh.vertexZ(vertexId));

  let yzMap = groups.get(xBits);
  if (!yzMap) {
    yzMap = new Map<number, Map<number, number[]>>();
    groups.set(xBits, yzMap);
  }

  let zMap = yzMap.get(yBits);
  if (!zMap) {
    zMap = new Map<number, number[]>();
    yzMap.set(yBits, zMap);
  }

  let arr = zMap.get(zBits);
  if (!arr) {
    arr = [];
    zMap.set(zBits, arr);
  }
  arr.push(value);
}

function positionBitGroupValues(groups: Map<number, Map<number, Map<number, number[]>>>): number[][] {
  const result: number[][] = [];
  for (const yzMap of groups.values()) {
    for (const zMap of yzMap.values()) {
      for (const ids of zMap.values()) {
        if (ids.length > 1) result.push(ids);
      }
    }
  }
  return result;
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

function buildShaderBundle(solidWGSL: string, solidBindings: PreparedSolidBindings, workgroupSize: number): ShaderBundle {
  return {
    corner: buildCornerShader(solidWGSL, solidBindings, workgroupSize),
    edgeX: buildEdgeShader(solidWGSL, solidBindings, workgroupSize, 'x'),
    edgeY: buildEdgeShader(solidWGSL, solidBindings, workgroupSize, 'y'),
    edgeZ: buildEdgeShader(solidWGSL, solidBindings, workgroupSize, 'z'),
    cube: buildCubeShader(solidWGSL, solidBindings, workgroupSize),
    countX: buildCountShader(solidWGSL, solidBindings, workgroupSize, 'x'),
    countY: buildCountShader(solidWGSL, solidBindings, workgroupSize, 'y'),
    countZ: buildCountShader(solidWGSL, solidBindings, workgroupSize, 'z'),
    emitX: buildEmitShader(solidWGSL, solidBindings, workgroupSize, 'x'),
    emitY: buildEmitShader(solidWGSL, solidBindings, workgroupSize, 'y'),
    emitZ: buildEmitShader(solidWGSL, solidBindings, workgroupSize, 'z'),
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

function wgslHeader(solidWGSL: string, solidBindings: PreparedSolidBindings): string {
  const surfaceHelpers = surfaceHelpersWGSL(solidWGSL);
  const solidBindingWGSL = buildSolidBindingWGSL(solidBindings);
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
  cornerRows: u32,
  cubeRows: u32,
};

struct BatchParams {
  zOffset: u32,
  cornerRingHead: u32,
  cubeRingHead: u32,
  localZStart: u32,
  localZCount: u32,
  isFinalBatch: u32,
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
  globalIndex: u32,
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
@group(0) @binding(1) var<uniform> batch: BatchParams;

${solidBindingWGSL}

${transformSolidWGSL(solidWGSL, solidBindings)}

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

fn cubeBufferIndex(ix: u32, iy: u32, localZ: u32) -> u32 {
  return ix + params.nx * (iy + params.ny * cubeSlot(localZ));
}

fn cubeGlobalIndex(ix: u32, iy: u32, localZ: u32) -> u32 {
  return ix + params.nx * (iy + params.ny * globalCornerZ(localZ));
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

fn cubeMinLocal(ix: u32, iy: u32, localZ: u32) -> vec3<f32> {
  return cornerPositionLocal(ix, iy, localZ);
}

fn cubeMaxLocal(ix: u32, iy: u32, localZ: u32) -> vec3<f32> {
  return cornerPositionLocal(ix + 1u, iy + 1u, localZ + 1u);
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

function buildCornerShader(solidWGSL: string, solidBindings: PreparedSolidBindings, workgroupSize: number): string {
  return /* wgsl */`
${wgslHeader(solidWGSL, solidBindings)}
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
${wgslHeader(solidWGSL, solidBindings)}
@group(0) @binding(2) var<storage, read> corners: array<u32>;
@group(0) @binding(3) var<storage, read_write> edges: array<HermiteEdge>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
${decode}
  let occ0 = corners[c0] != 0u;
  let occ1 = corners[c1] != 0u;
  if (occ0 == occ1) {
    edges[edgeIndex].isActive = 0u;
    edges[edgeIndex].pos = vec4<f32>(0.0);
    edges[edgeIndex].normal = vec4<f32>(0.0);
    return;
  }
  let hit = bisectOccupancyEdge(p0, p1, occ0);
  edges[edgeIndex].isActive = 1u;
  edges[edgeIndex].pos = vec4<f32>(hit, 1.0);
  edges[edgeIndex].normal = vec4<f32>(estimateNormal(hit), 0.0);
}
`;
}

function buildCubeShader(solidWGSL: string, solidBindings: PreparedSolidBindings, workgroupSize: number): string {
  return /* wgsl */`
${wgslHeader(solidWGSL, solidBindings)}
@group(0) @binding(2) var<storage, read> corners: array<u32>;
@group(0) @binding(3) var<storage, read> xEdges: array<HermiteEdge>;
@group(0) @binding(4) var<storage, read> yEdges: array<HermiteEdge>;
@group(0) @binding(5) var<storage, read> zEdges: array<HermiteEdge>;
@group(0) @binding(6) var<storage, read_write> cubes: array<CubeVertex>;

fn cubeCornerValue(ix: u32, iy: u32, localZ: u32, dx: u32, dy: u32, dz: u32) -> bool {
  return corners[cornerIndexLocal(ix + dx, iy + dy, localZ + dz)] != 0u;
}

fn clipToCube(p: vec3<f32>, ix: u32, iy: u32, localZ: u32) -> vec3<f32> {
  let margin = params.cubeMargin;
  let lo = cubeMinLocal(ix, iy, localZ) + vec3<f32>(margin);
  let hi = cubeMaxLocal(ix, iy, localZ) - vec3<f32>(margin);
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
  let total = params.nx * params.ny * batch.localZCount;
  if (i >= total) { return; }

  let ix = i % params.nx;
  let t = i / params.nx;
  let iy = t % params.ny;
  let localZ = batch.localZStart + t / params.ny;
  let cubeIndex = cubeBufferIndex(ix, iy, localZ);

  let c000 = cubeCornerValue(ix, iy, localZ, 0u, 0u, 0u);
  let c100 = cubeCornerValue(ix, iy, localZ, 1u, 0u, 0u);
  let c010 = cubeCornerValue(ix, iy, localZ, 0u, 1u, 0u);
  let c110 = cubeCornerValue(ix, iy, localZ, 1u, 1u, 0u);
  let c001 = cubeCornerValue(ix, iy, localZ, 0u, 0u, 1u);
  let c101 = cubeCornerValue(ix, iy, localZ, 1u, 0u, 1u);
  let c011 = cubeCornerValue(ix, iy, localZ, 0u, 1u, 1u);
  let c111 = cubeCornerValue(ix, iy, localZ, 1u, 1u, 1u);

  let firstValue = c000;
  let activeCube =
    (c100 != firstValue) || (c010 != firstValue) || (c110 != firstValue) ||
    (c001 != firstValue) || (c101 != firstValue) || (c011 != firstValue) || (c111 != firstValue);
  if (!activeCube) {
    cubes[cubeIndex].isActive = 0u;
    cubes[cubeIndex].globalIndex = 0xffffffffu;
    cubes[cubeIndex].pos = vec4<f32>(0.0);
    return;
  }

  let e0 = xEdges[xEdgeIndexLocal(ix, iy, localZ)];
  let e1 = yEdges[yEdgeIndexLocal(ix, iy, localZ)];
  let e2 = yEdges[yEdgeIndexLocal(ix + 1u, iy, localZ)];
  let e3 = xEdges[xEdgeIndexLocal(ix, iy + 1u, localZ)];
  let e4 = zEdges[zEdgeIndexLocal(ix, iy, localZ)];
  let e5 = zEdges[zEdgeIndexLocal(ix + 1u, iy, localZ)];
  let e6 = zEdges[zEdgeIndexLocal(ix, iy + 1u, localZ)];
  let e7 = zEdges[zEdgeIndexLocal(ix + 1u, iy + 1u, localZ)];
  let e8 = xEdges[xEdgeIndexLocal(ix, iy, localZ + 1u)];
  let e9 = yEdges[yEdgeIndexLocal(ix, iy, localZ + 1u)];
  let e10 = yEdges[yEdgeIndexLocal(ix + 1u, iy, localZ + 1u)];
  let e11 = xEdges[xEdgeIndexLocal(ix, iy + 1u, localZ + 1u)];

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
    cubes[cubeIndex].isActive = 0u;
    cubes[cubeIndex].globalIndex = 0xffffffffu;
    cubes[cubeIndex].pos = vec4<f32>(0.0);
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
    p = clipToCube(p, ix, iy, localZ);
  }

  cubes[cubeIndex].isActive = 1u;
  cubes[cubeIndex].globalIndex = cubeGlobalIndex(ix, iy, localZ);
  cubes[cubeIndex].pos = vec4<f32>(p, 1.0);
}
`;
}

function buildEmitShader(solidWGSL: string, solidBindings: PreparedSolidBindings, workgroupSize: number, axis: 'x' | 'y' | 'z'): string {
  const decode = axis === 'x'
    ? `
  let total = params.nx * (params.ny + 1u) * batch.localZCount;
  if (i >= total) { return; }
  let ix = i % params.nx;
  let t = i / params.nx;
  let iy = t % (params.ny + 1u);
  let localZ = batch.localZStart + t / (params.ny + 1u);
  let globalZ = globalCornerZ(localZ);
  if (iy == 0u || globalZ == 0u || iy >= params.ny || globalZ >= params.nz) { return; }
  let edge = edges[xEdgeIndexLocal(ix, iy, localZ)];
  if (edge.isActive == 0u) { return; }
  var cubeIds = array<u32, 4>(
    cubeBufferIndex(ix, iy, localZ - 1u),
    cubeBufferIndex(ix, iy - 1u, localZ - 1u),
    cubeBufferIndex(ix, iy - 1u, localZ),
    cubeBufferIndex(ix, iy, localZ),
  );
  var globalIds = array<u32, 4>(
    cubeGlobalIndex(ix, iy, localZ - 1u),
    cubeGlobalIndex(ix, iy - 1u, localZ - 1u),
    cubeGlobalIndex(ix, iy - 1u, localZ),
    cubeGlobalIndex(ix, iy, localZ),
  );
  let flip = corners[cornerIndexLocal(ix, iy, localZ)] != 0u;
`
    : axis === 'y'
      ? `
  let total = (params.nx + 1u) * params.ny * batch.localZCount;
  if (i >= total) { return; }
  let ix = i % (params.nx + 1u);
  let t = i / (params.nx + 1u);
  let iy = t % params.ny;
  let localZ = batch.localZStart + t / params.ny;
  let globalZ = globalCornerZ(localZ);
  if (ix == 0u || globalZ == 0u || ix >= params.nx || globalZ >= params.nz) { return; }
  let edge = edges[yEdgeIndexLocal(ix, iy, localZ)];
  if (edge.isActive == 0u) { return; }
  var cubeIds = array<u32, 4>(
    cubeBufferIndex(ix - 1u, iy, localZ),
    cubeBufferIndex(ix - 1u, iy, localZ - 1u),
    cubeBufferIndex(ix, iy, localZ - 1u),
    cubeBufferIndex(ix, iy, localZ),
  );
  var globalIds = array<u32, 4>(
    cubeGlobalIndex(ix - 1u, iy, localZ),
    cubeGlobalIndex(ix - 1u, iy, localZ - 1u),
    cubeGlobalIndex(ix, iy, localZ - 1u),
    cubeGlobalIndex(ix, iy, localZ),
  );
  let flip = corners[cornerIndexLocal(ix, iy, localZ)] != 0u;
`
      : `
  let total = (params.nx + 1u) * (params.ny + 1u) * batch.localZCount;
  if (i >= total) { return; }
  let ix = i % (params.nx + 1u);
  let t = i / (params.nx + 1u);
  let iy = t % (params.ny + 1u);
  let localZ = batch.localZStart + t / (params.ny + 1u);
  if (ix == 0u || iy == 0u || ix >= params.nx || iy >= params.ny) { return; }
  let edge = edges[zEdgeIndexLocal(ix, iy, localZ)];
  if (edge.isActive == 0u) { return; }
  var cubeIds = array<u32, 4>(
    cubeBufferIndex(ix, iy - 1u, localZ),
    cubeBufferIndex(ix - 1u, iy - 1u, localZ),
    cubeBufferIndex(ix - 1u, iy, localZ),
    cubeBufferIndex(ix, iy, localZ),
  );
  var globalIds = array<u32, 4>(
    cubeGlobalIndex(ix, iy - 1u, localZ),
    cubeGlobalIndex(ix - 1u, iy - 1u, localZ),
    cubeGlobalIndex(ix - 1u, iy, localZ),
    cubeGlobalIndex(ix, iy, localZ),
  );
  let flip = corners[cornerIndexLocal(ix, iy, localZ)] != 0u;
`;

  return /* wgsl */`
${wgslHeader(solidWGSL, solidBindings)}
@group(0) @binding(2) var<storage, read> corners: array<u32>;
@group(0) @binding(3) var<storage, read> edges: array<HermiteEdge>;
@group(0) @binding(4) var<storage, read> cubes: array<CubeVertex>;
@group(0) @binding(5) var<storage, read> offsets: array<u32>;
@group(0) @binding(6) var<storage, read_write> triangles: array<TriangleIndex>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
${decode}
  var p0 = cubes[cubeIds[0]].pos.xyz;
  var p1 = cubes[cubeIds[1]].pos.xyz;
  var p2 = cubes[cubeIds[2]].pos.xyz;
  var p3 = cubes[cubeIds[3]].pos.xyz;
  if (
    cubes[cubeIds[0]].isActive == 0u || cubes[cubeIds[0]].globalIndex != globalIds[0] ||
    cubes[cubeIds[1]].isActive == 0u || cubes[cubeIds[1]].globalIndex != globalIds[1] ||
    cubes[cubeIds[2]].isActive == 0u || cubes[cubeIds[2]].globalIndex != globalIds[2] ||
    cubes[cubeIds[3]].isActive == 0u || cubes[cubeIds[3]].globalIndex != globalIds[3]
  ) {
    return;
  }

  if (flip) {
    let tmp0 = globalIds[0];
    let tmp1 = globalIds[1];
    globalIds[0] = globalIds[3];
    globalIds[1] = globalIds[2];
    globalIds[2] = tmp1;
    globalIds[3] = tmp0;
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
    triangles[base + 0u] = TriangleIndex(globalIds[0], globalIds[1], globalIds[2], 0u);
    triangles[base + 1u] = TriangleIndex(globalIds[0], globalIds[2], globalIds[3], 0u);
  } else {
    triangles[base + 0u] = TriangleIndex(globalIds[1], globalIds[2], globalIds[3], 0u);
    triangles[base + 1u] = TriangleIndex(globalIds[1], globalIds[3], globalIds[0], 0u);
  }
}
`;
}

function buildCountShader(solidWGSL: string, solidBindings: PreparedSolidBindings, workgroupSize: number, axis: 'x' | 'y' | 'z'): string {
  const decode = axis === 'x'
    ? `
  let total = params.nx * (params.ny + 1u) * batch.localZCount;
  if (i >= total) { return; }
  let ix = i % params.nx;
  let t = i / params.nx;
  let iy = t % (params.ny + 1u);
  let localZ = batch.localZStart + t / (params.ny + 1u);
  let globalZ = globalCornerZ(localZ);
  if (iy == 0u || globalZ == 0u || iy >= params.ny || globalZ >= params.nz) {
    counts[i] = 0u;
    return;
  }
  let edge = edges[xEdgeIndexLocal(ix, iy, localZ)];
  if (edge.isActive == 0u) {
    counts[i] = 0u;
    return;
  }
  let ids = array<u32, 4>(
    cubeBufferIndex(ix, iy, localZ - 1u),
    cubeBufferIndex(ix, iy - 1u, localZ - 1u),
    cubeBufferIndex(ix, iy - 1u, localZ),
    cubeBufferIndex(ix, iy, localZ),
  );
  let globalIds = array<u32, 4>(
    cubeGlobalIndex(ix, iy, localZ - 1u),
    cubeGlobalIndex(ix, iy - 1u, localZ - 1u),
    cubeGlobalIndex(ix, iy - 1u, localZ),
    cubeGlobalIndex(ix, iy, localZ),
  );
`
    : axis === 'y'
      ? `
  let total = (params.nx + 1u) * params.ny * batch.localZCount;
  if (i >= total) { return; }
  let ix = i % (params.nx + 1u);
  let t = i / (params.nx + 1u);
  let iy = t % params.ny;
  let localZ = batch.localZStart + t / params.ny;
  let globalZ = globalCornerZ(localZ);
  if (ix == 0u || globalZ == 0u || ix >= params.nx || globalZ >= params.nz) {
    counts[i] = 0u;
    return;
  }
  let edge = edges[yEdgeIndexLocal(ix, iy, localZ)];
  if (edge.isActive == 0u) {
    counts[i] = 0u;
    return;
  }
  let ids = array<u32, 4>(
    cubeBufferIndex(ix - 1u, iy, localZ),
    cubeBufferIndex(ix - 1u, iy, localZ - 1u),
    cubeBufferIndex(ix, iy, localZ - 1u),
    cubeBufferIndex(ix, iy, localZ),
  );
  let globalIds = array<u32, 4>(
    cubeGlobalIndex(ix - 1u, iy, localZ),
    cubeGlobalIndex(ix - 1u, iy, localZ - 1u),
    cubeGlobalIndex(ix, iy, localZ - 1u),
    cubeGlobalIndex(ix, iy, localZ),
  );
`
      : `
  let total = (params.nx + 1u) * (params.ny + 1u) * batch.localZCount;
  if (i >= total) { return; }
  let ix = i % (params.nx + 1u);
  let t = i / (params.nx + 1u);
  let iy = t % (params.ny + 1u);
  let localZ = batch.localZStart + t / (params.ny + 1u);
  if (ix == 0u || iy == 0u || ix >= params.nx || iy >= params.ny) {
    counts[i] = 0u;
    return;
  }
  let edge = edges[zEdgeIndexLocal(ix, iy, localZ)];
  if (edge.isActive == 0u) {
    counts[i] = 0u;
    return;
  }
  let ids = array<u32, 4>(
    cubeBufferIndex(ix, iy - 1u, localZ),
    cubeBufferIndex(ix - 1u, iy - 1u, localZ),
    cubeBufferIndex(ix - 1u, iy, localZ),
    cubeBufferIndex(ix, iy, localZ),
  );
  let globalIds = array<u32, 4>(
    cubeGlobalIndex(ix, iy - 1u, localZ),
    cubeGlobalIndex(ix - 1u, iy - 1u, localZ),
    cubeGlobalIndex(ix - 1u, iy, localZ),
    cubeGlobalIndex(ix, iy, localZ),
  );
`;

  return /* wgsl */`
${wgslHeader(solidWGSL, solidBindings)}
@group(0) @binding(2) var<storage, read> edges: array<HermiteEdge>;
@group(0) @binding(3) var<storage, read> cubes: array<CubeVertex>;
@group(0) @binding(4) var<storage, read_write> counts: array<u32>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
${decode}
  if (
    cubes[ids[0]].isActive == 0u || cubes[ids[0]].globalIndex != globalIds[0] ||
    cubes[ids[1]].isActive == 0u || cubes[ids[1]].globalIndex != globalIds[1] ||
    cubes[ids[2]].isActive == 0u || cubes[ids[2]].globalIndex != globalIds[2] ||
    cubes[ids[3]].isActive == 0u || cubes[ids[3]].globalIndex != globalIds[3]
  ) {
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

function buildSolidBindingWGSL(solidBindings: PreparedSolidBindings): string {
  return solidBindings.declarationWGSL;
}
