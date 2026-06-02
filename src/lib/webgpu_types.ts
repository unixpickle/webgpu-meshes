export interface GPUBufferUsageConstants {
  COPY_DST: number;
  COPY_SRC: number;
  MAP_READ: number;
  STORAGE: number;
  UNIFORM: number;
}

export interface GPUMapModeConstants {
  READ: number;
}

export interface GPUShaderStageConstants {
  COMPUTE: number;
}

export const GPUBufferUsage = (globalThis as unknown as { GPUBufferUsage: GPUBufferUsageConstants }).GPUBufferUsage;
export const GPUMapMode = (globalThis as unknown as { GPUMapMode: GPUMapModeConstants }).GPUMapMode;
export const GPUShaderStage = (globalThis as unknown as { GPUShaderStage: GPUShaderStageConstants }).GPUShaderStage;

export type GPUBufferSource = ArrayBuffer | ArrayBufferView<ArrayBufferLike>;
export type GPUWriteBufferSource = ArrayBufferLike | ArrayBufferView<ArrayBufferLike>;

export interface GPUBuffer {
  readonly size: number;
  readonly usage?: number;
  destroy(): void;
  mapAsync(mode: number): Promise<void>;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
}

export type GPUBindingResource =
  | GPUBuffer
  | {
      buffer: GPUBuffer;
      offset?: number;
      size?: number;
    };

export interface GPUBindGroupLayout {
}

export interface GPUPipelineLayout {
}

export interface GPUBindGroup {
}

export interface GPUComputePipeline {
}

export interface GPUCommandBuffer {
}

export interface GPUCompilationMessage {
  readonly type: string;
  readonly message?: string;
  readonly lineNum?: number;
  readonly linePos?: number;
}

export interface GPUCompilationInfo {
  readonly messages: readonly GPUCompilationMessage[];
}

export interface GPUShaderModule {
  getCompilationInfo?(): Promise<GPUCompilationInfo>;
}

export interface GPUQueue {
  writeBuffer(
    buffer: GPUBuffer,
    bufferOffset: number,
    data: GPUWriteBufferSource,
    dataOffset?: number,
    size?: number,
  ): void;
  submit(commandBuffers: GPUCommandBuffer[]): void;
  onSubmittedWorkDone(): Promise<void>;
}

export interface GPUComputePassEncoder {
  setPipeline(pipeline: GPUComputePipeline): void;
  setBindGroup(index: number, bindGroup: GPUBindGroup): void;
  dispatchWorkgroups(workgroupCountX: number): void;
  end(): void;
}

export interface GPUCommandEncoder {
  beginComputePass(descriptor?: { label?: string }): GPUComputePassEncoder;
  copyBufferToBuffer(
    source: GPUBuffer,
    sourceOffset: number,
    destination: GPUBuffer,
    destinationOffset: number,
    size: number,
  ): void;
  finish(): GPUCommandBuffer;
}

export interface GPUDevice {
  readonly limits?: Record<string, number | undefined>;
  readonly lost: Promise<unknown>;
  readonly queue: GPUQueue;
  createBuffer(descriptor: {
    label?: string;
    size: number;
    usage: number;
  }): GPUBuffer;
  createBindGroup(descriptor: {
    label?: string;
    layout: GPUBindGroupLayout;
    entries: Array<{
      binding: number;
      resource: GPUBindingResource;
    }>;
  }): GPUBindGroup;
  createBindGroupLayout(descriptor: {
    label?: string;
    entries: Array<{
      binding: number;
      visibility: number;
      buffer: {
        type: 'uniform' | 'storage' | 'read-only-storage';
      };
    }>;
  }): GPUBindGroupLayout;
  createCommandEncoder(descriptor?: { label?: string }): GPUCommandEncoder;
  createComputePipelineAsync(descriptor: {
    label?: string;
    layout: GPUPipelineLayout;
    compute: {
      module: GPUShaderModule;
      entryPoint: string;
    };
  }): Promise<GPUComputePipeline>;
  createPipelineLayout(descriptor: {
    label?: string;
    bindGroupLayouts: GPUBindGroupLayout[];
  }): GPUPipelineLayout;
  createShaderModule(descriptor: {
    label?: string;
    code: string;
  }): GPUShaderModule;
}
