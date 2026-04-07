import './styles.css';

import { dualContourWebGPU, exampleSphereSolidWGSL } from './dual_contouring_webgpu';
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
      <h1>Dual contour a WGSL solid into STL.</h1>
      <p class="lede">
        Define <code>solidOccupancy()</code>, set bounds and grid spacing, then download the repaired mesh as an STL file.
      </p>
    </section>

    <section class="panel">
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
        <label class="toggle">
          <input id="repair" type="checkbox" />
          <span>CPU repair</span>
        </label>
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
const repairInput = getElement<HTMLInputElement>('repair');
const generateButton = getElement<HTMLButtonElement>('generate');
const statusBox = getElement<HTMLElement>('status');

solidTextarea.value = exampleSphereSolidWGSL.trim();
boundsMinTextarea.value = '-1.25, -1.25, -1.25';
boundsMaxTextarea.value = '1.25, 1.25, 1.25';
deltaInput.value = '0.1';
filenameInput.value = 'dual-contour-sphere.stl';
repairInput.checked = true;
statusBox.textContent = gpuNavigator.gpu
  ? 'Ready. WebGPU detected.'
  : 'WebGPU is not available in this browser.';

let devicePromise: Promise<any> | null = null;

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

    statusBox.textContent = 'Requesting WebGPU device...';
    const device = await getDevice();

    const start = performance.now();
    statusBox.textContent = 'Running dual contouring on the GPU...';
    const result = await dualContourWebGPU({
      device,
      solidWGSL,
      min,
      max,
      delta,
      repair: repairInput.checked,
      label: 'webui-dual-contour',
    });

    const initialTriangleCount = result.initial.indices.length / 3;
    const initialVertexCount = result.initial.positions.length / 3;
    const repairedTriangleCount = result.repaired.indices.length / 3;
    const repairedVertexCount = result.repaired.positions.length / 3;

    let mesh = result.repaired;
    let meshLabel = repairInput.checked ? 'repaired' : 'initial';
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
  return adapter.requestDevice();
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
