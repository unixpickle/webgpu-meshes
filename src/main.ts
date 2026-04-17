import './styles.css';

import { dualContourWebGPU, exampleSphereSolidWGSL } from './dual_contouring';
import { marchingCubesWebGPU } from './marching_cubes';
import { meshToBinarySTL } from './stl';
import type { Vec3 } from './vec3';

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
      <h1>Turn a WGSL solid into STL.</h1>
      <p class="lede">
        Define <code>solidOccupancy()</code>, set bounds and spacing, then export the generated mesh as STL.
      </p>
    </section>

    <section class="panel">
      <fieldset class="mode-picker">
        <legend>Meshing algorithm</legend>
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

      <div class="grid">
        <label class="field">
          <span>Bounding box min</span>
          <textarea id="bounds-min" rows="2" spellcheck="false"></textarea>
        </label>
        <label class="field">
          <span>Bounding box max</span>
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
        <div id="mc-options" class="algorithm-options" hidden>
          <label class="field field-inline">
            <span>Search iterations</span>
            <input id="mc-search-iterations" type="number" min="1" step="1" />
          </label>
        </div>
        <button id="generate" type="button">Generate STL</button>
      </div>

      <pre id="status" class="status" aria-live="polite"></pre>
    </section>
  </main>
`;

const solidTextarea = getElement<HTMLTextAreaElement>('solid-wgsl');
const boundsMinTextarea = getElement<HTMLTextAreaElement>('bounds-min');
const boundsMaxTextarea = getElement<HTMLTextAreaElement>('bounds-max');
const deltaInput = getElement<HTMLInputElement>('delta');
const filenameInput = getElement<HTMLInputElement>('filename');
const modeDualInput = getElement<HTMLInputElement>('mode-dual');
const modeMCInput = getElement<HTMLInputElement>('mode-mc');
const dcOptions = getElement<HTMLElement>('dc-options');
const mcOptions = getElement<HTMLElement>('mc-options');
const clipInput = getElement<HTMLInputElement>('clip');
const repairInput = getElement<HTMLInputElement>('repair');
const mcSearchIterationsInput = getElement<HTMLInputElement>('mc-search-iterations');
const generateButton = getElement<HTMLButtonElement>('generate');
const statusBox = getElement<HTMLElement>('status');

solidTextarea.value = exampleSphereSolidWGSL.trim();
boundsMinTextarea.value = '-1.25, -1.25, -1.25';
boundsMaxTextarea.value = '1.25, 1.25, 1.25';
deltaInput.value = '0.1';
filenameInput.value = 'wgsl-mesh.stl';
clipInput.checked = false;
repairInput.checked = true;
mcSearchIterationsInput.value = '8';
statusBox.textContent = gpuNavigator.gpu
  ? 'Ready. WebGPU detected.'
  : 'WebGPU is not available in this browser.';

let devicePromise: Promise<any> | null = null;
updateModeUI();

modeDualInput.addEventListener('change', updateModeUI);
modeMCInput.addEventListener('change', updateModeUI);

generateButton.addEventListener('click', async () => {
  generateButton.disabled = true;
  try {
    if (!gpuNavigator.gpu) {
      throw new Error('This browser does not expose WebGPU.');
    }

    const solidWGSL = solidTextarea.value.trim();
    if (!solidWGSL) {
      throw new Error('Enter WGSL source for solidOccupancy().');
    }

    const min = parseVec3(boundsMinTextarea.value, 'Bounding box min');
    const max = parseVec3(boundsMaxTextarea.value, 'Bounding box max');
    const delta = Number(deltaInput.value);
    if (!(delta > 0)) {
      throw new Error('Grid delta must be greater than 0.');
    }
    validateBounds(min, max);
    const mode = currentMeshingMode();

    statusBox.textContent = 'Requesting WebGPU device...';
    const device = await getDevice();

    const start = performance.now();
    let meshLabel = 'generated';
    let mesh;
    if (mode === 'dual') {
      statusBox.textContent = 'Running dual contouring on the GPU...';
      const result = await dualContourWebGPU({
        device,
        solidWGSL,
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
      const searchIterations = parseIntegerInput(mcSearchIterationsInput.value, 'Search iterations', 1);
      statusBox.textContent = 'Running marching cubes on the GPU...';
      const result = await marchingCubesWebGPU({
        device,
        solidWGSL,
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
    const blob = meshToBinarySTL(mesh, sanitizeSolidName(filenameInput.value));
    downloadBlob(blob, normalizeFilename(filenameInput.value));
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

function parseVec3(value: string, label: string): Vec3 {
  const parts = value
    .split(/[\s,]+/)
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
  if (parts.length !== 3) {
    throw new Error(`${label} must contain exactly three numbers.`);
  }

  const numbers = parts.map((part) => Number(part));
  if (numbers.some((part) => !Number.isFinite(part))) {
    throw new Error(`${label} contains an invalid number.`);
  }
  return [numbers[0], numbers[1], numbers[2]];
}

function validateBounds(min: Vec3, max: Vec3): void {
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

function normalizeFilename(value: string): string {
  const trimmed = value.trim() || 'dual-contour-mesh';
  return trimmed.toLowerCase().endsWith('.stl') ? trimmed : `${trimmed}.stl`;
}

function sanitizeSolidName(value: string): string {
  return normalizeFilename(value).replace(/\.stl$/i, '');
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

function currentMeshingMode(): 'dual' | 'mc' {
  return modeMCInput.checked ? 'mc' : 'dual';
}

function updateModeUI(): void {
  const mode = currentMeshingMode();
  const isDual = mode === 'dual';
  dcOptions.hidden = !isDual;
  mcOptions.hidden = isDual;
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
