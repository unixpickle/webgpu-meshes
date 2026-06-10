import './styles.css';

import {
  dualContourWebGPU,
  exampleCircleSolidWGSL,
  exampleSphereSolidWGSL,
  marchingCubesWebGPU,
  marchingSquaresWebGPU,
  meshToBinarySTL,
  segmentsToSVGBlob,
  type Vec3,
} from './lib';

type Vec2 = [number, number];
type ExportTarget = 'stl' | 'svg';
type MeshingMode = 'dual' | 'mc';

interface DemoState {
  solidWGSL: string;
  buffersJSON: string;
  boundsMin: string;
  boundsMax: string;
  delta: string;
  filename: string;
  meshingMode: MeshingMode;
  clip: boolean;
  repair: boolean;
  searchIterations: string;
}

const gpuNavigator = navigator as Navigator & {
  gpu?: {
    requestAdapter(): Promise<any>;
  };
};

const root = document.querySelector<HTMLDivElement>('#app');
if (!root) {
  throw new Error('Missing #app root element.');
}

root.innerHTML = `
  <main class="shell">
    <section class="hero">
      <p class="eyebrow">WebGPU Mesher</p>
      <h1 id="hero-title">Turn a WGSL solid into STL or SVG.</h1>
      <p id="hero-lede" class="lede">
        Use the STL tab for 3D solids with a <code>SolidVector</code> from <code>vec3&lt;f32&gt;</code>, or the SVG tab for 2D outlines from <code>vec2&lt;f32&gt;</code>.
      </p>
    </section>

    <section class="panel">
      <fieldset class="target-picker">
        <legend>Export target</legend>
        <label class="mode-option">
          <input id="target-stl" type="radio" name="export-target" value="stl" checked />
          <span>3D STL</span>
        </label>
        <label class="mode-option">
          <input id="target-svg" type="radio" name="export-target" value="svg" />
          <span>2D SVG</span>
        </label>
      </fieldset>

      <fieldset id="mode-picker" class="mode-picker">
        <legend>3D meshing algorithm</legend>
        <label class="mode-option">
          <input id="mode-dual" type="radio" name="meshing-mode" value="dual" checked />
          <span>Dual Contouring</span>
        </label>
        <label class="mode-option">
          <input id="mode-mc" type="radio" name="meshing-mode" value="mc" />
          <span>Marching Cubes</span>
        </label>
      </fieldset>

      <label class="field field-large">
        <span>Solid shader (WGSL)</span>
        <textarea id="solid-wgsl" spellcheck="false"></textarea>
      </label>

      <label class="field field-large">
        <span>Buffers (JSON)</span>
        <textarea id="buffers-json" rows="4" spellcheck="false"></textarea>
      </label>

      <div class="grid">
        <label class="field">
          <span id="bounds-min-label">Bounding box min</span>
          <textarea id="bounds-min" rows="2" spellcheck="false"></textarea>
        </label>
        <label class="field">
          <span id="bounds-max-label">Bounding box max</span>
          <textarea id="bounds-max" rows="2" spellcheck="false"></textarea>
        </label>
      </div>

      <div class="controls">
        <label class="field field-inline">
          <span>Grid delta</span>
          <input id="delta" type="number" min="0.000001" step="0.01" />
        </label>
        <label class="field field-inline">
          <span>Filename</span>
          <input id="filename" type="text" />
        </label>
        <div id="dc-options" class="algorithm-options algorithm-options-inline">
          <label class="toggle">
            <input id="clip" type="checkbox" />
            <span>Clip</span>
          </label>
          <label class="toggle">
            <input id="repair" type="checkbox" />
            <span>CPU repair</span>
          </label>
        </div>
        <div id="search-options" class="algorithm-options" hidden>
          <label class="field field-inline">
            <span id="search-iterations-label">Search iterations</span>
            <input id="search-iterations" type="number" min="0" step="1" />
          </label>
        </div>
        <button id="generate" type="button">Generate STL</button>
      </div>

      <pre id="status" class="status" aria-live="polite"></pre>
    </section>
  </main>
`;

const defaultStates: Record<ExportTarget, DemoState> = {
  stl: {
    solidWGSL: exampleSphereSolidWGSL.trim(),
    buffersJSON: '{}',
    boundsMin: '-1.25, -1.25, -1.25',
    boundsMax: '1.25, 1.25, 1.25',
    delta: '0.1',
    filename: 'wgsl-mesh.stl',
    meshingMode: 'dual',
    clip: false,
    repair: true,
    searchIterations: '8',
  },
  svg: {
    solidWGSL: exampleCircleSolidWGSL.trim(),
    buffersJSON: '{}',
    boundsMin: '-1.25, -1.25',
    boundsMax: '1.25, 1.25',
    delta: '0.03',
    filename: 'wgsl-outline.svg',
    meshingMode: 'mc',
    clip: false,
    repair: false,
    searchIterations: '8',
  },
};

const targetStates: Record<ExportTarget, DemoState> = {
  stl: { ...defaultStates.stl },
  svg: { ...defaultStates.svg },
};

const heroTitle = getElement<HTMLElement>('hero-title');
const heroLede = getElement<HTMLElement>('hero-lede');
const targetSTLInput = getElement<HTMLInputElement>('target-stl');
const targetSVGInput = getElement<HTMLInputElement>('target-svg');
const modePicker = getElement<HTMLElement>('mode-picker');
const solidTextarea = getElement<HTMLTextAreaElement>('solid-wgsl');
const buffersTextarea = getElement<HTMLTextAreaElement>('buffers-json');
const boundsMinLabel = getElement<HTMLElement>('bounds-min-label');
const boundsMaxLabel = getElement<HTMLElement>('bounds-max-label');
const boundsMinTextarea = getElement<HTMLTextAreaElement>('bounds-min');
const boundsMaxTextarea = getElement<HTMLTextAreaElement>('bounds-max');
const deltaInput = getElement<HTMLInputElement>('delta');
const filenameInput = getElement<HTMLInputElement>('filename');
const modeDualInput = getElement<HTMLInputElement>('mode-dual');
const modeMCInput = getElement<HTMLInputElement>('mode-mc');
const dcOptions = getElement<HTMLElement>('dc-options');
const searchOptions = getElement<HTMLElement>('search-options');
const clipInput = getElement<HTMLInputElement>('clip');
const repairInput = getElement<HTMLInputElement>('repair');
const searchIterationsLabel = getElement<HTMLElement>('search-iterations-label');
const searchIterationsInput = getElement<HTMLInputElement>('search-iterations');
const generateButton = getElement<HTMLButtonElement>('generate');
const statusBox = getElement<HTMLElement>('status');

let devicePromise: Promise<any> | null = null;
let activeTarget: ExportTarget = currentExportTarget();

applyDemoState(targetStates[activeTarget]);
updateTargetUI();
statusBox.textContent = gpuNavigator.gpu
  ? 'Ready. WebGPU detected.'
  : 'WebGPU is not available in this browser.';

targetSTLInput.addEventListener('change', handleTargetChange);
targetSVGInput.addEventListener('change', handleTargetChange);
modeDualInput.addEventListener('change', updateTargetUI);
modeMCInput.addEventListener('change', updateTargetUI);

generateButton.addEventListener('click', async () => {
  generateButton.disabled = true;
  try {
    if (!gpuNavigator.gpu) {
      throw new Error('This browser does not expose WebGPU.');
    }

    const target = currentExportTarget();
    targetStates[target] = captureDemoState();
    const solidWGSL = solidTextarea.value.trim();
    if (!solidWGSL) {
      throw new Error('Enter WGSL source for solidOccupancy().');
    }
    const solidBindings = parseSolidBindingsJSON(buffersTextarea.value);
    const delta = Number(deltaInput.value);
    if (!(delta > 0)) {
      throw new Error('Grid delta must be greater than 0.');
    }

    statusBox.textContent = 'Requesting WebGPU device...';
    const device = await getDevice();
    const start = performance.now();

    if (target === 'svg') {
      const min = parseVec2(boundsMinTextarea.value, 'Bounds min');
      const max = parseVec2(boundsMaxTextarea.value, 'Bounds max');
      validateBounds2D(min, max);
      const searchIterations = parseIntegerInput(searchIterationsInput.value, 'Search iterations', 0);

      statusBox.textContent = 'Running marching squares on the GPU...';
      const result = await marchingSquaresWebGPU({
        device,
        solidWGSL,
        solidBindings,
        min,
        max,
        delta,
        bisectionSteps: searchIterations,
        label: 'webui-marching-squares',
      });
      logStageMetrics('webui-marching-squares', result.metrics);

      const segmentCount = result.mesh.indices.length / 2;
      const vertexCount = result.mesh.positions.length / 2;
      if (segmentCount === 0 || vertexCount === 0) {
        throw new Error(
          `No segments were generated.\n` +
          `Check the shader, bounds, delta, or the search-iteration setting.`
        );
      }

      statusBox.textContent =
        `Marching squares generated ${segmentCount} segments / ${vertexCount} vertices.\n` +
        `Building SVG...`;

      const filename = normalizeFilename(filenameInput.value, '.svg', 'wgsl-outline');
      const blob = segmentsToSVGBlob(result.mesh, {
        title: sanitizeFilenameBase(filenameInput.value, '.svg', 'wgsl-outline'),
      });
      downloadBlob(blob, filename);
      const durationMs = Math.round(performance.now() - start);
      statusBox.textContent =
        `Downloaded the outline as SVG in ${durationMs} ms.\n` +
        `${segmentCount} segments / ${vertexCount} vertices written.`;
      return;
    }

    const min = parseVec3(boundsMinTextarea.value, 'Bounding box min');
    const max = parseVec3(boundsMaxTextarea.value, 'Bounding box max');
    validateBounds3D(min, max);
    const mode = currentMeshingMode();

    let meshLabel = 'generated';
    let mesh;
    if (mode === 'dual') {
      statusBox.textContent = 'Running dual contouring on the GPU...';
      const result = await dualContourWebGPU({
        device,
        solidWGSL,
        solidBindings,
        min,
        max,
        delta,
        clip: clipInput.checked,
        repair: repairInput.checked,
        label: 'webui-dual-contour',
      });
      logStageMetrics('webui-dual-contour', result.metrics);

      const initialTriangleCount = result.initial.indices.length / 3;
      const initialVertexCount = result.initial.positions.length / 3;
      const repairedTriangleCount = result.repaired.indices.length / 3;
      const repairedVertexCount = result.repaired.positions.length / 3;

      mesh = result.repaired;
      meshLabel = repairInput.checked ? 'repaired' : 'initial';
      if (repairInput.checked && repairedTriangleCount === 0 && initialTriangleCount > 0) {
        mesh = result.initial;
        meshLabel = 'initial';
        statusBox.textContent =
          `Repair produced 0 triangles, falling back to the initial mesh.\n` +
          `Initial: ${initialTriangleCount} triangles / ${initialVertexCount} vertices.\n` +
          `Repaired: ${repairedTriangleCount} triangles / ${repairedVertexCount} vertices.`;
      }

      const triangleCount = mesh.indices.length / 3;
      const vertexCount = mesh.positions.length / 3;
      if (triangleCount === 0 || vertexCount === 0) {
        throw new Error(
          `No triangles were generated.\n` +
          `Initial: ${initialTriangleCount} triangles / ${initialVertexCount} vertices.\n` +
          `Repaired: ${repairedTriangleCount} triangles / ${repairedVertexCount} vertices.\n` +
          `Check the shader, bounds, delta, or the repair option.`
        );
      }

      statusBox.textContent =
        `Using the ${meshLabel} mesh.\n` +
        `Initial: ${initialTriangleCount} triangles / ${initialVertexCount} vertices.\n` +
        `Repaired: ${repairedTriangleCount} triangles / ${repairedVertexCount} vertices.\n` +
        `Building STL...`;
    } else {
      const searchIterations = parseIntegerInput(searchIterationsInput.value, 'Search iterations', 1);
      statusBox.textContent = 'Running marching cubes on the GPU...';
      const result = await marchingCubesWebGPU({
        device,
        solidWGSL,
        solidBindings,
        min,
        max,
        delta,
        bisectionSteps: searchIterations,
        label: 'webui-marching-cubes',
      });
      logStageMetrics('webui-marching-cubes', result.metrics);
      mesh = result.mesh.compact();
      const triangleCount = mesh.indices.length / 3;
      const vertexCount = mesh.positions.length / 3;
      if (triangleCount === 0 || vertexCount === 0) {
        throw new Error(
          `No triangles were generated.\n` +
          `Check the shader, bounds, delta, or the search-iteration setting.`
        );
      }
      statusBox.textContent =
        `Marching cubes generated ${triangleCount} triangles / ${vertexCount} vertices.\n` +
        `Building STL...`;
    }

    const triangleCount = mesh.indices.length / 3;
    const vertexCount = mesh.positions.length / 3;
    const filename = normalizeFilename(filenameInput.value, '.stl', 'wgsl-mesh');
    const blob = meshToBinarySTL(mesh, sanitizeFilenameBase(filenameInput.value, '.stl', 'wgsl-mesh'));
    downloadBlob(blob, filename);
    const durationMs = Math.round(performance.now() - start);
    statusBox.textContent =
      `Downloaded the ${meshLabel} mesh as STL in ${durationMs} ms.\n` +
      `${triangleCount} triangles / ${vertexCount} vertices written.`;
  } catch (error) {
    statusBox.textContent = formatError(error);
  } finally {
    generateButton.disabled = false;
  }
});

function getElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing element #${id}.`);
  }
  return element as T;
}

async function getDevice(): Promise<any> {
  if (!devicePromise) {
    devicePromise = requestDevice();
  }
  return devicePromise;
}

async function requestDevice(): Promise<any> {
  const adapter = await gpuNavigator.gpu?.requestAdapter();
  if (!adapter) {
    throw new Error('Unable to acquire a WebGPU adapter.');
  }
  const adapterLimits = adapter.limits as Record<string, number | undefined> | undefined;
  const requiredLimits: Record<string, number> = {};
  if (adapterLimits?.maxStorageBufferBindingSize !== undefined) {
    requiredLimits.maxStorageBufferBindingSize = adapterLimits.maxStorageBufferBindingSize;
  }
  if (adapterLimits?.maxBufferSize !== undefined) {
    requiredLimits.maxBufferSize = adapterLimits.maxBufferSize;
  }
  return adapter.requestDevice({
    requiredLimits,
  });
}

function currentExportTarget(): ExportTarget {
  return targetSVGInput.checked ? 'svg' : 'stl';
}

function currentMeshingMode(): MeshingMode {
  return modeMCInput.checked ? 'mc' : 'dual';
}

function handleTargetChange(): void {
  const nextTarget = currentExportTarget();
  if (nextTarget === activeTarget) {
    updateTargetUI();
    return;
  }
  targetStates[activeTarget] = captureDemoState();
  activeTarget = nextTarget;
  applyDemoState(targetStates[activeTarget]);
  updateTargetUI();
}

function applyDemoState(state: DemoState): void {
  solidTextarea.value = state.solidWGSL;
  buffersTextarea.value = state.buffersJSON;
  boundsMinTextarea.value = state.boundsMin;
  boundsMaxTextarea.value = state.boundsMax;
  deltaInput.value = state.delta;
  filenameInput.value = state.filename;
  modeDualInput.checked = state.meshingMode === 'dual';
  modeMCInput.checked = state.meshingMode === 'mc';
  clipInput.checked = state.clip;
  repairInput.checked = state.repair;
  searchIterationsInput.value = state.searchIterations;
}

function captureDemoState(): DemoState {
  return {
    solidWGSL: solidTextarea.value,
    buffersJSON: buffersTextarea.value,
    boundsMin: boundsMinTextarea.value,
    boundsMax: boundsMaxTextarea.value,
    delta: deltaInput.value,
    filename: filenameInput.value,
    meshingMode: currentMeshingMode(),
    clip: clipInput.checked,
    repair: repairInput.checked,
    searchIterations: searchIterationsInput.value,
  };
}

function updateTargetUI(): void {
  const target = currentExportTarget();
  const isSTL = target === 'stl';
  const mode = currentMeshingMode();
  const isDual = isSTL && mode === 'dual';
  const showSearch = target === 'svg' || (isSTL && mode === 'mc');

  heroTitle.textContent = isSTL ? 'Turn a WGSL solid into STL.' : 'Turn a WGSL solid into SVG.';
  heroLede.innerHTML = isSTL
    ? 'Define <code>SolidVector</code> helpers for <code>vec3&lt;f32&gt;</code>, set bounds and spacing, then export STL.'
    : 'Define <code>SolidVector</code> helpers for <code>vec2&lt;f32&gt;</code>, set 2D bounds and spacing, then export SVG.';
  boundsMinLabel.textContent = isSTL ? 'Bounding box min' : 'Bounds min';
  boundsMaxLabel.textContent = isSTL ? 'Bounding box max' : 'Bounds max';
  searchIterationsLabel.textContent = isSTL ? 'Search iterations' : 'Search iterations';
  generateButton.textContent = isSTL ? 'Generate STL' : 'Generate SVG';
  modePicker.hidden = !isSTL;
  dcOptions.hidden = !isDual;
  searchOptions.hidden = !showSearch;
  searchIterationsInput.min = isSTL ? '1' : '0';
}

function parseVec2(value: string, label: string): Vec2 {
  const parts = parseNumberList(value, label, 2);
  return [parts[0], parts[1]];
}

function parseVec3(value: string, label: string): Vec3 {
  const parts = parseNumberList(value, label, 3);
  return [parts[0], parts[1], parts[2]];
}

function parseNumberList(value: string, label: string, expectedCount: number): number[] {
  const parts = value
    .split(/[\s,]+/)
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
  if (parts.length !== expectedCount) {
    throw new Error(`${label} must contain exactly ${expectedCount} numbers.`);
  }
  const numbers = parts.map((part) => Number(part));
  if (numbers.some((part) => !Number.isFinite(part))) {
    throw new Error(`${label} contains an invalid number.`);
  }
  return numbers;
}

function validateBounds2D(min: Vec2, max: Vec2): void {
  for (let i = 0; i < 2; i++) {
    if (!(max[i] > min[i])) {
      throw new Error('Each max bound must be greater than the matching min bound.');
    }
  }
}

function validateBounds3D(min: Vec3, max: Vec3): void {
  for (let i = 0; i < 3; i++) {
    if (!(max[i] > min[i])) {
      throw new Error('Each max bound must be greater than the matching min bound.');
    }
  }
}

function parseIntegerInput(value: string, label: string, min: number): number {
  const number = Number(value);
  if (!Number.isInteger(number) || number < min) {
    throw new Error(`${label} must be an integer greater than or equal to ${min}.`);
  }
  return number;
}

function parseSolidBindingsJSON(value: string): Array<{
  name: string;
  kind: 'storage';
  wgslType: 'array<f32>';
  source: Float32Array;
}> {
  const trimmed = value.trim();
  if (!trimmed) {
    return [];
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch (error) {
    throw new Error(`Buffers must be valid JSON. ${formatError(error)}`);
  }

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('Buffers must be a JSON object mapping buffer names to float arrays.');
  }

  const result: Array<{
    name: string;
    kind: 'storage';
    wgslType: 'array<f32>';
    source: Float32Array;
  }> = [];

  for (const [name, rawValue] of Object.entries(parsed as Record<string, unknown>)) {
    if (!Array.isArray(rawValue)) {
      throw new Error(`Buffer "${name}" must be an array of numbers.`);
    }
    const numbers = rawValue.map((item) => {
      if (typeof item !== 'number' || !Number.isFinite(item)) {
        throw new Error(`Buffer "${name}" contains a non-finite value.`);
      }
      return item;
    });
    result.push({
      name,
      kind: 'storage',
      wgslType: 'array<f32>',
      source: new Float32Array(numbers),
    });
  }

  return result;
}

function normalizeFilename(value: string, extension: '.stl' | '.svg', fallbackBase: string): string {
  const trimmed = value.trim() || fallbackBase;
  return trimmed.toLowerCase().endsWith(extension) ? trimmed : `${trimmed}${extension}`;
}

function sanitizeFilenameBase(value: string, extension: '.stl' | '.svg', fallbackBase: string): string {
  const normalized = normalizeFilename(value, extension, fallbackBase);
  return normalized.slice(0, normalized.length - extension.length);
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function formatError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return `${error}`;
}

function logStageMetrics(label: string, metrics: { totalMs: number; stages: Array<{ stage: string; ms: number }> }): void {
  let cumulative = 0;
  const rows = metrics.stages.map(({ stage, ms }) => {
    cumulative += ms;
    return {
      stage,
      ms: Number(ms.toFixed(2)),
      cumulativeMs: Number(cumulative.toFixed(2)),
    };
  });
  console.groupCollapsed(`[${label}] stage timings (${metrics.totalMs.toFixed(2)} ms total)`);
  console.table(rows);
  console.groupEnd();
}
