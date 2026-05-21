const GPUBufferUsageAny: any = (globalThis as any).GPUBufferUsage;
const GPUShaderStageAny: any = (globalThis as any).GPUShaderStage;

export interface SolidBindingInput {
  name: string;
  kind: 'uniform' | 'storage';
  wgslType: string;
  wgslDefs?: string;
  source: BufferSource | any;
  size?: number;
  label?: string;
}

interface ResolvedSolidBinding {
  index: number;
  name: string;
  kind: 'uniform' | 'storage';
  wgslType: string;
  wgslDefs: string;
  label: string | undefined;
  size: number;
  source: BufferSource | any;
  runtimeArray: RuntimeArrayInfo | null;
}

interface RuntimeArrayInfo {
  elementType: string;
  elementStride: number;
}

interface PackedBindingRewrite {
  packedName: string;
  elementOffset: number;
  elementCount: number;
}

export interface PreparedBindingResource {
  binding: number;
  kind: 'uniform' | 'storage';
  buffer: any;
  size: number;
  declaration: string;
}

export interface PreparedSolidBindings {
  bindings: PreparedBindingResource[];
  declarationWGSL: string;
  rewriteState: unknown;
}

export function prepareSolidBindings(
  device: any,
  solidBindings: SolidBindingInput[],
  label: string,
  errorPrefix: string,
): PreparedSolidBindings {
  const resolved = resolveSolidBindings(solidBindings, errorPrefix);
  if (resolved.length === 0) {
    return {
      bindings: [],
      declarationWGSL: '',
      rewriteState: new Map<string, PackedBindingRewrite>(),
    };
  }

  const defs = [...new Set(resolved.map((binding) => binding.wgslDefs).filter((wgslDefs) => wgslDefs.length > 0))];
  const declarations: string[] = [];
  const bindings: PreparedBindingResource[] = [];
  const rewrites = new Map<string, PackedBindingRewrite>();
  const groupOrder: string[] = [];
  const groups = new Map<string, ResolvedSolidBinding[]>();

  for (const binding of resolved) {
    const key = binding.runtimeArray === null ? `standalone:${binding.index}` : `packed:${binding.kind}:${binding.wgslType}`;
    if (!groups.has(key)) {
      groups.set(key, []);
      groupOrder.push(key);
    }
    groups.get(key)!.push(binding);
  }

  let nextBinding = 0;
  for (const key of groupOrder) {
    const group = groups.get(key)!;
    const first = group[0];
    if (group.length === 1 && first.runtimeArray === null) {
      const resource = materializeStandaloneBinding(device, first, label, errorPrefix);
      bindings.push({
        binding: nextBinding,
        kind: first.kind,
        buffer: resource.buffer,
        size: resource.size,
        declaration: buildStandaloneDeclaration(nextBinding, first),
      });
      declarations.push(bindings[bindings.length - 1].declaration);
      nextBinding += 1;
      continue;
    }

    if (first.runtimeArray === null) {
      throw new Error(`${errorPrefix}: internal error: non-packable binding group cannot be coalesced.`);
    }

    const packedName = `_solidPacked${nextBinding}`;
    const packed = materializePackedRuntimeArrayGroup(device, group, label, nextBinding, packedName, errorPrefix);
    bindings.push({
      binding: nextBinding,
      kind: 'storage',
      buffer: packed.buffer,
      size: packed.size,
      declaration: `@group(1) @binding(${nextBinding}) var<storage, read> ${packedName}: array<${first.runtimeArray.elementType}>;`,
    });
    declarations.push(bindings[bindings.length - 1].declaration);
    for (const [name, rewrite] of packed.rewrites) {
      rewrites.set(name, rewrite);
    }
    nextBinding += 1;
  }

  return {
    bindings,
    declarationWGSL: `${defs.join('\n\n')}${defs.length > 0 && declarations.length > 0 ? '\n\n' : ''}${declarations.join('\n')}`,
    rewriteState: rewrites,
  };
}

export function transformSolidWGSL(source: string, solidBindings: PreparedSolidBindings): string {
  return rewriteSolidWGSL(source, solidBindings.rewriteState as Map<string, PackedBindingRewrite>);
}

export function createSolidBindGroupLayout(device: any, solidBindings: PreparedSolidBindings): any {
  return device.createBindGroupLayout({
    entries: solidBindings.bindings.map((binding) => ({
      binding: binding.binding,
      visibility: GPUShaderStageAny.COMPUTE,
      buffer: {
        type: binding.kind === 'uniform' ? 'uniform' : 'read-only-storage',
      },
    })),
  });
}

export function createSolidBindGroup(device: any, layout: any, solidBindings: PreparedSolidBindings): any {
  return device.createBindGroup({
    layout,
    entries: solidBindings.bindings.map((binding) => ({
      binding: binding.binding,
      resource: {
        buffer: binding.buffer,
        size: binding.size,
      },
    })),
  });
}

function resolveSolidBindings(solidBindings: SolidBindingInput[], errorPrefix: string): ResolvedSolidBinding[] {
  const seenNames = new Set<string>();
  return solidBindings.map((binding, bindingIndex) => {
    validateSolidBinding(binding, bindingIndex, seenNames, errorPrefix);
    const wgslType = binding.wgslType.trim();
    const source = binding.source;
    const size = getSolidBindingSize(source, binding.size, bindingIndex, errorPrefix);
    return {
      index: bindingIndex,
      name: binding.name,
      kind: binding.kind,
      wgslType,
      wgslDefs: binding.wgslDefs?.trim() ?? '',
      label: binding.label,
      size,
      source,
      runtimeArray: binding.kind === 'storage' ? parseRuntimeArrayInfo(wgslType) : null,
    };
  });
}

function validateSolidBinding(
  binding: SolidBindingInput,
  bindingIndex: number,
  seenNames: Set<string>,
  errorPrefix: string,
): void {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(binding.name)) {
    throw new Error(`${errorPrefix}: solidBindings[${bindingIndex}] has invalid WGSL identifier "${binding.name}".`);
  }
  if (seenNames.has(binding.name)) {
    throw new Error(`${errorPrefix}: duplicate solid binding name "${binding.name}".`);
  }
  seenNames.add(binding.name);
  if (binding.kind !== 'uniform' && binding.kind !== 'storage') {
    throw new Error(`${errorPrefix}: solidBindings[${bindingIndex}] has invalid kind "${String(binding.kind)}".`);
  }
  if (!binding.wgslType.trim()) {
    throw new Error(`${errorPrefix}: solidBindings[${bindingIndex}] must provide a non-empty wgslType.`);
  }
}

function getSolidBindingSize(source: BufferSource | any, requestedSize: number | undefined, bindingIndex: number, errorPrefix: string): number {
  if (isBufferSource(source)) {
    const size = source.byteLength;
    if (size <= 0) {
      throw new Error(`${errorPrefix}: solidBindings[${bindingIndex}] data source must contain at least one byte.`);
    }
    return size;
  }
  if (!source || typeof source !== 'object' || typeof (source as { size?: unknown }).size !== 'number') {
    throw new Error(`${errorPrefix}: solidBindings[${bindingIndex}] source must be a typed array, ArrayBuffer, or GPUBuffer.`);
  }
  const sourceSize = (source as { size: number }).size;
  const size = requestedSize ?? sourceSize;
  if (!(size > 0) || size > sourceSize) {
    throw new Error(`${errorPrefix}: solidBindings[${bindingIndex}] size must be in (0, buffer.size].`);
  }
  return size;
}

function materializeStandaloneBinding(
  device: any,
  binding: ResolvedSolidBinding,
  label: string,
  errorPrefix: string,
): { buffer: any; size: number } {
  if (isBufferSource(binding.source)) {
    const usage = (binding.kind === 'uniform' ? GPUBufferUsageAny.UNIFORM : GPUBufferUsageAny.STORAGE) | GPUBufferUsageAny.COPY_DST;
    const buffer = device.createBuffer({
      label: binding.label ?? `${label}-solid-binding-${binding.index}-${binding.name}`,
      size: binding.size,
      usage,
    });
    writeSolidBindingBuffer(device, buffer, 0, binding.source);
    return { buffer, size: binding.size };
  }

  const source = binding.source as { usage?: number; size: number };
  const requiredUsage = binding.kind === 'uniform' ? GPUBufferUsageAny.UNIFORM : GPUBufferUsageAny.STORAGE;
  const usage = typeof source.usage === 'number' ? source.usage : 0;
  if ((usage & requiredUsage) === 0) {
    throw new Error(`${errorPrefix}: solidBindings[${binding.index}] GPUBuffer is missing ${binding.kind.toUpperCase()} usage.`);
  }
  return { buffer: source, size: binding.size };
}

function buildStandaloneDeclaration(bindingIndex: number, binding: ResolvedSolidBinding): string {
  return binding.kind === 'uniform'
    ? `@group(1) @binding(${bindingIndex}) var<uniform> ${binding.name}: ${binding.wgslType};`
    : `@group(1) @binding(${bindingIndex}) var<storage, read> ${binding.name}: ${binding.wgslType};`;
}

function materializePackedRuntimeArrayGroup(
  device: any,
  bindings: ResolvedSolidBinding[],
  label: string,
  bindingIndex: number,
  packedName: string,
  errorPrefix: string,
): { buffer: any; size: number; rewrites: Map<string, PackedBindingRewrite> } {
  const totalSize = bindings.reduce((sum, binding) => sum + binding.size, 0);
  const buffer = device.createBuffer({
    label: `${label}-solid-packed-${bindingIndex}`,
    size: totalSize,
    usage: GPUBufferUsageAny.STORAGE | GPUBufferUsageAny.COPY_DST,
  });

  const rewrites = new Map<string, PackedBindingRewrite>();
  let encoder: any = null;
  let currentOffset = 0;
  for (const binding of bindings) {
    const runtimeArray = binding.runtimeArray!;
    if (binding.size % runtimeArray.elementStride !== 0) {
      throw new Error(
        `${errorPrefix}: solidBindings[${binding.index}] size ${binding.size} is not a multiple of ${runtimeArray.elementStride} bytes for ${binding.wgslType}.`,
      );
    }

    if (isBufferSource(binding.source)) {
      writeSolidBindingBuffer(device, buffer, currentOffset, binding.source);
    } else {
      const source = binding.source as { usage?: number };
      const usage = typeof source.usage === 'number' ? source.usage : 0;
      if ((usage & GPUBufferUsageAny.COPY_SRC) === 0) {
        throw new Error(
          `${errorPrefix}: solidBindings[${binding.index}] GPUBuffer must include COPY_SRC usage so it can be packed into ${packedName}.`,
        );
      }
      if (encoder === null) {
        encoder = device.createCommandEncoder({ label: `${label}-solid-pack-copy` });
      }
      encoder.copyBufferToBuffer(binding.source, 0, buffer, currentOffset, binding.size);
    }

    rewrites.set(binding.name, {
      packedName,
      elementOffset: currentOffset / runtimeArray.elementStride,
      elementCount: binding.size / runtimeArray.elementStride,
    });
    currentOffset += binding.size;
  }

  if (encoder !== null) {
    device.queue.submit([encoder.finish()]);
  }

  return { buffer, size: totalSize, rewrites };
}

function rewriteSolidWGSL(source: string, rewrites: Map<string, PackedBindingRewrite>): string {
  if (rewrites.size === 0 || source.length === 0) {
    return source;
  }

  const tokens = lexWGSLTokens(source);
  let result = '';
  let cursor = 0;
  let tokenIndex = 0;
  while (tokenIndex < tokens.length) {
    const token = tokens[tokenIndex];
    if (
      token.kind === 'identifier' &&
      token.text === 'arrayLength' &&
      tokenIndex + 4 < tokens.length &&
      tokens[tokenIndex + 1].text === '(' &&
      tokens[tokenIndex + 2].text === '&' &&
      tokens[tokenIndex + 3].kind === 'identifier'
    ) {
      const rewrite = rewrites.get(tokens[tokenIndex + 3].text);
      if (rewrite !== undefined && tokens[tokenIndex + 4].text === ')') {
        result += source.slice(cursor, token.start);
        result += `${rewrite.elementCount}u`;
        cursor = tokens[tokenIndex + 4].end;
        tokenIndex += 5;
        continue;
      }
    }

    if (
      token.kind === 'identifier' &&
      rewrites.has(token.text) &&
      tokenIndex + 1 < tokens.length &&
      tokens[tokenIndex + 1].text === '['
    ) {
      const rewrite = rewrites.get(token.text)!;
      result += source.slice(cursor, token.start);
      result += `${rewrite.packedName}[${rewrite.elementOffset}u + `;
      cursor = tokens[tokenIndex + 1].end;
      tokenIndex += 2;
      continue;
    }

    tokenIndex += 1;
  }

  result += source.slice(cursor);
  return result;
}

interface WGSLToken {
  kind: 'identifier' | 'punct';
  text: string;
  start: number;
  end: number;
}

function lexWGSLTokens(source: string): WGSLToken[] {
  const tokens: WGSLToken[] = [];
  let i = 0;
  while (i < source.length) {
    const char = source[i];
    if (isWhitespace(char)) {
      i += 1;
      continue;
    }
    if (char === '/' && source[i + 1] === '/') {
      i += 2;
      while (i < source.length && source[i] !== '\n') {
        i += 1;
      }
      continue;
    }
    if (char === '/' && source[i + 1] === '*') {
      i += 2;
      while (i + 1 < source.length && !(source[i] === '*' && source[i + 1] === '/')) {
        i += 1;
      }
      i = Math.min(source.length, i + 2);
      continue;
    }
    if (isIdentifierStart(char)) {
      const start = i;
      i += 1;
      while (i < source.length && isIdentifierContinue(source[i])) {
        i += 1;
      }
      tokens.push({ kind: 'identifier', text: source.slice(start, i), start, end: i });
      continue;
    }
    if ('()[]&'.includes(char)) {
      tokens.push({ kind: 'punct', text: char, start: i, end: i + 1 });
    }
    i += 1;
  }
  return tokens;
}

function parseRuntimeArrayInfo(wgslType: string): RuntimeArrayInfo | null {
  const arrayArgs = parseArrayArguments(wgslType);
  if (arrayArgs === null || arrayArgs.length !== 1) {
    return null;
  }
  const elementType = arrayArgs[0];
  const layout = parsePlainWGSLTypeLayout(elementType);
  if (layout === null) {
    return null;
  }
  return {
    elementType,
    elementStride: roundUp(layout.align, layout.size),
  };
}

function parseArrayArguments(wgslType: string): string[] | null {
  const trimmed = wgslType.trim();
  if (!trimmed.startsWith('array<') || !trimmed.endsWith('>')) {
    return null;
  }
  return splitTopLevelArgs(trimmed.slice(6, -1));
}

interface WGSLTypeLayout {
  align: number;
  size: number;
}

function parsePlainWGSLTypeLayout(typeText: string): WGSLTypeLayout | null {
  const trimmed = typeText.trim();
  switch (trimmed) {
    case 'f16':
      return { align: 2, size: 2 };
    case 'f32':
    case 'i32':
    case 'u32':
      return { align: 4, size: 4 };
  }

  const atomicMatch = /^atomic<(.+)>$/.exec(trimmed);
  if (atomicMatch) {
    const inner = atomicMatch[1].trim();
    if (inner === 'i32' || inner === 'u32') {
      return { align: 4, size: 4 };
    }
    return null;
  }

  const vecMatch = /^vec([234])<(.+)>$/.exec(trimmed);
  if (vecMatch) {
    const length = Number(vecMatch[1]);
    const scalar = parsePlainWGSLTypeLayout(vecMatch[2]);
    if (scalar === null || (scalar.size !== 2 && scalar.size !== 4)) {
      return null;
    }
    if (length === 2) {
      return { align: 2 * scalar.align, size: 2 * scalar.size };
    }
    if (length === 3) {
      return { align: 4 * scalar.align, size: 3 * scalar.size };
    }
    return { align: 4 * scalar.align, size: 4 * scalar.size };
  }

  const matMatch = /^mat([234])x([234])<(.+)>$/.exec(trimmed);
  if (matMatch) {
    const columns = Number(matMatch[1]);
    const rows = Number(matMatch[2]);
    const scalar = parsePlainWGSLTypeLayout(matMatch[3]);
    if (scalar === null || (scalar.size !== 2 && scalar.size !== 4)) {
      return null;
    }
    const columnLayout = parsePlainWGSLTypeLayout(`vec${rows}<${matMatch[3].trim()}>`);
    if (columnLayout === null) {
      return null;
    }
    const columnStride = roundUp(columnLayout.align, columnLayout.size);
    return {
      align: columnLayout.align,
      size: columns * columnStride,
    };
  }

  const arrayArgs = parseArrayArguments(trimmed);
  if (arrayArgs !== null && arrayArgs.length === 2) {
    const elementLayout = parsePlainWGSLTypeLayout(arrayArgs[0]);
    if (elementLayout === null || !/^[0-9]+u?$/.test(arrayArgs[1])) {
      return null;
    }
    const count = Number(arrayArgs[1].replace(/u$/, ''));
    const stride = roundUp(elementLayout.align, elementLayout.size);
    return {
      align: elementLayout.align,
      size: count * stride,
    };
  }

  return null;
}

function splitTopLevelArgs(text: string): string[] {
  const parts: string[] = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (char === '<') {
      depth += 1;
    } else if (char === '>') {
      depth -= 1;
    } else if (char === ',' && depth === 0) {
      parts.push(text.slice(start, i).trim());
      start = i + 1;
    }
  }
  parts.push(text.slice(start).trim());
  return parts.filter((part) => part.length > 0);
}

function writeSolidBindingBuffer(device: any, buffer: any, offset: number, source: BufferSource): void {
  if (source instanceof ArrayBuffer) {
    device.queue.writeBuffer(buffer, offset, source);
  } else {
    device.queue.writeBuffer(buffer, offset, source.buffer, source.byteOffset, source.byteLength);
  }
}

function isBufferSource(value: unknown): value is BufferSource {
  return value instanceof ArrayBuffer || ArrayBuffer.isView(value);
}

function roundUp(alignment: number, value: number): number {
  return Math.ceil(value / alignment) * alignment;
}

function isWhitespace(char: string): boolean {
  return char === ' ' || char === '\n' || char === '\r' || char === '\t';
}

function isIdentifierStart(char: string): boolean {
  return /[A-Za-z_]/.test(char);
}

function isIdentifierContinue(char: string): boolean {
  return /[A-Za-z0-9_]/.test(char);
}
