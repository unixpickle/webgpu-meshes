import type { IndexedSegments2D } from './marching_squares';

export interface SegmentsToSVGOptions {
  title?: string;
  fill?: string;
  stroke?: string;
  strokeWidth?: number;
  padding?: number;
  background?: string;
}

export function segmentsToSVG(mesh: IndexedSegments2D, options: SegmentsToSVGOptions = {}): string {
  if (mesh.positions.length % 2 !== 0) {
    throw new Error('segmentsToSVG(): mesh positions length must be a multiple of 2.');
  }
  if (mesh.indices.length % 2 !== 0) {
    throw new Error('segmentsToSVG(): mesh indices length must be a multiple of 2.');
  }
  if (mesh.positions.length === 0 || mesh.indices.length === 0) {
    throw new Error('segmentsToSVG(): mesh must contain at least one segment.');
  }

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < mesh.positions.length; i += 2) {
    const x = mesh.positions[i];
    const y = mesh.positions[i + 1];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const spanX = Math.max(maxX - minX, 1e-6);
  const spanY = Math.max(maxY - minY, 1e-6);
  const padding = options.padding ?? Math.max(spanX, spanY) * 0.02;
  const strokeWidth = options.strokeWidth ?? Math.max(Math.max(spanX, spanY) * 0.004, 0.001);
  const viewMinX = minX - padding;
  const viewMinY = minY - padding;
  const viewWidth = spanX + 2 * padding;
  const viewHeight = spanY + 2 * padding;
  const fill = options.fill ?? '#17120f';
  const stroke = options.stroke ?? 'none';
  const background = options.background ?? '#fffaf2';
  const title = options.title?.trim();
  const pathData = buildFilledPathData(mesh);

  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${formatNumber(viewMinX)} ${formatNumber(viewMinY)} ${formatNumber(viewWidth)} ${formatNumber(viewHeight)}">`,
    title ? `  <title>${escapeXML(title)}</title>` : '',
    `  <rect x="${formatNumber(viewMinX)}" y="${formatNumber(viewMinY)}" width="${formatNumber(viewWidth)}" height="${formatNumber(viewHeight)}" fill="${escapeXML(background)}"/>`,
    `  <g transform="translate(0 ${formatNumber(minY + maxY)}) scale(1 -1)">`,
    `    <path d="${pathData}" fill="${escapeXML(fill)}" fill-rule="evenodd" stroke="${escapeXML(stroke)}" stroke-width="${formatNumber(strokeWidth)}" stroke-linecap="round" stroke-linejoin="round"/>`,
    '  </g>',
    '</svg>',
  ].filter((line) => line.length > 0).join('\n');
}

export function segmentsToSVGBlob(mesh: IndexedSegments2D, options: SegmentsToSVGOptions = {}): Blob {
  return new Blob([segmentsToSVG(mesh, options)], { type: 'image/svg+xml;charset=utf-8' });
}

function formatNumber(value: number): string {
  return Number(value.toFixed(6)).toString();
}

function buildFilledPathData(mesh: IndexedSegments2D): string {
  const nextByStart = new Map<number, number[]>();
  for (let segmentIndex = 0; segmentIndex < mesh.indices.length; segmentIndex += 2) {
    const start = mesh.indices[segmentIndex];
    const entries = nextByStart.get(start);
    if (entries) {
      entries.push(segmentIndex);
    } else {
      nextByStart.set(start, [segmentIndex]);
    }
  }

  const visited = new Uint8Array(mesh.indices.length / 2);
  const pathParts: string[] = [];

  for (let segmentIndex = 0; segmentIndex < mesh.indices.length; segmentIndex += 2) {
    const segmentId = segmentIndex / 2;
    if (visited[segmentId] !== 0) continue;

    const loopIndices: number[] = [];
    let currentSegmentIndex = segmentIndex;
    const startVertex = mesh.indices[currentSegmentIndex];

    for (;;) {
      const currentSegmentId = currentSegmentIndex / 2;
      if (visited[currentSegmentId] !== 0) break;
      visited[currentSegmentId] = 1;
      loopIndices.push(currentSegmentIndex);

      const nextStartVertex = mesh.indices[currentSegmentIndex + 1];
      if (nextStartVertex === startVertex) break;
      const candidates = nextByStart.get(nextStartVertex);
      if (!candidates) {
        throw new Error('segmentsToSVG(): encountered an open contour; expected closed loops.');
      }
      const nextSegment = candidates.find((candidate) => visited[candidate / 2] === 0);
      if (nextSegment === undefined) {
        throw new Error('segmentsToSVG(): encountered a broken contour chain while reconstructing loops.');
      }
      currentSegmentIndex = nextSegment;
    }

    if (loopIndices.length === 0) continue;
    const firstVertex = mesh.indices[loopIndices[0]] * 2;
    const commands = [
      `M ${formatNumber(mesh.positions[firstVertex])} ${formatNumber(mesh.positions[firstVertex + 1])}`,
    ];
    for (const loopSegmentIndex of loopIndices) {
      const endVertex = mesh.indices[loopSegmentIndex + 1] * 2;
      commands.push(`L ${formatNumber(mesh.positions[endVertex])} ${formatNumber(mesh.positions[endVertex + 1])}`);
    }
    commands.push('Z');
    pathParts.push(commands.join(' '));
  }

  if (pathParts.length === 0) {
    throw new Error('segmentsToSVG(): failed to reconstruct any closed contours.');
  }
  return pathParts.join(' ');
}

function escapeXML(value: string): string {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&apos;');
}
