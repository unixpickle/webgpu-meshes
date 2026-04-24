import { existsSync } from 'node:fs';
import http from 'node:http';
import { chromium } from 'playwright';

const WORKGROUP_SIZE = 64;

function readStdin() {
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (chunk) => {
      data += chunk;
    });
    process.stdin.on('end', () => resolve(data));
    process.stdin.on('error', reject);
  });
}

function browserArgs() {
  const args = ['--enable-unsafe-webgpu'];
  if (process.platform === 'linux') {
    args.push('--use-angle=vulkan', '--enable-features=Vulkan');
  }
  return args;
}

function browserExecutablePath() {
  if (process.env.SHAPEKERNEL_CHROME_PATH) {
    return process.env.SHAPEKERNEL_CHROME_PATH;
  }
  if (process.platform === 'darwin') {
    const candidate = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
    if (existsSync(candidate)) {
      return candidate;
    }
  }
  return undefined;
}

function buildShaderSource(request) {
  const bufferDecls = request.buffers.map((buffer, index) => (
    `@group(0) @binding(${index}) var<storage, read> ${buffer.name}: array<f32>;`
  ));
  const pointExpr = request.dim === 2 ? 'packed.xy' : 'packed.xyz';
  const outputType = request.returnType === 'bool' ? 'u32' : 'f32';
  const outputExpr = request.returnType === 'bool'
    ? `select(0u, 1u, ${request.entrypoint}(p))`
    : `${request.entrypoint}(p)`;
  return `
${bufferDecls.join('\n')}
@group(0) @binding(${request.inputBinding}) var<storage, read> test_inputs: array<vec4<f32>>;
@group(0) @binding(${request.outputBinding}) var<storage, read_write> test_outputs: array<${outputType}>;

${request.code}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= ${request.inputCount}u) {
    return;
  }
  let packed = test_inputs[index];
  let p = ${pointExpr};
  test_outputs[index] = ${outputExpr};
}
`;
}

function startServer() {
  const server = http.createServer((_req, res) => {
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
    res.end('<!doctype html><title>shapekernel executor</title>');
  });
  return new Promise((resolve, reject) => {
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      if (!address || typeof address === 'string') {
        reject(new Error('failed to determine local server address'));
        return;
      }
      resolve({
        server,
        url: `http://127.0.0.1:${address.port}/`,
      });
    });
  });
}

function closeServer(server) {
  return new Promise((resolve, reject) => {
    server.close((err) => {
      if (err) {
        reject(err);
      } else {
        resolve();
      }
    });
  });
}

async function executeInBrowser(request) {
  const launch = await startServer();
  let browser;

  try {
    browser = await chromium.launch({
      headless: true,
      args: browserArgs(),
      executablePath: browserExecutablePath(),
    });
    const page = await browser.newPage();
    await page.goto(launch.url);
    return await page.evaluate(async (request) => {
      const gpu = navigator.gpu;
      if (!gpu) {
        throw new Error('navigator.gpu is unavailable in Chromium');
      }

      const adapter = await gpu.requestAdapter();
      if (!adapter) {
        throw new Error('navigator.gpu.requestAdapter() returned null');
      }

      const device = await adapter.requestDevice();
      const shaderModule = device.createShaderModule({ code: request.shaderSource });
      const compilationInfo = await shaderModule.getCompilationInfo();
      const compilationErrors = compilationInfo.messages.filter((message) => message.type === 'error');
      if (compilationErrors.length > 0) {
        const details = compilationErrors.map((message) => (
          `${message.lineNum}:${message.linePos}: ${message.message}`
        )).join('\n');
        throw new Error(`WGSL compilation failed:\n${details}`);
      }

      const pipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: {
          module: shaderModule,
          entryPoint: 'main',
        },
      });

      const bindEntries = [];
      for (let i = 0; i < request.buffers.length; i += 1) {
        const data = new Float32Array(request.buffers[i].values);
        const buffer = device.createBuffer({
          size: data.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buffer, 0, data);
        bindEntries.push({
          binding: i,
          resource: { buffer },
        });
      }

      const packedInputs = new Float32Array(request.inputs.flat());
      const inputBuffer = device.createBuffer({
        size: packedInputs.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(inputBuffer, 0, packedInputs);

      const outputSize = request.inputCount * 4;
      const outputBuffer = device.createBuffer({
        size: outputSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const readbackBuffer = device.createBuffer({
        size: outputSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      bindEntries.push({
        binding: request.inputBinding,
        resource: { buffer: inputBuffer },
      });
      bindEntries.push({
        binding: request.outputBinding,
        resource: { buffer: outputBuffer },
      });

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: bindEntries,
      });

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(request.inputCount / 64));
      pass.end();
      encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputSize);
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      await readbackBuffer.mapAsync(GPUMapMode.READ);

      const copy = readbackBuffer.getMappedRange().slice(0);
      readbackBuffer.unmap();

      if (request.returnType === 'bool') {
        return {
          bools: Array.from(new Uint32Array(copy), (value) => value !== 0),
        };
      }
      return {
        floats: Array.from(new Float32Array(copy)),
      };
    }, {
      ...request,
      shaderSource: buildShaderSource(request),
    });
  } finally {
    if (browser) {
      await browser.close();
    }
    await closeServer(launch.server);
  }
}

async function main() {
  const raw = await readStdin();
  const request = JSON.parse(raw);
  const result = await executeInBrowser(request);
  process.stdout.write(`${JSON.stringify(result)}\n`);
}

main().catch((err) => {
  process.stderr.write(`${err?.stack ?? String(err)}\n`);
  process.exitCode = 1;
});
