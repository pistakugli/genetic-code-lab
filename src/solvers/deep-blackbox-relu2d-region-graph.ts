export type DeepBlackBoxRelu2DRegion = {
  id: number;
  intercept: number;
  slopeX: number;
  slopeY: number;
  support: number;
  centroid: [number, number];
};

export type DeepBlackBoxRelu2DAdjacency = {
  a: number;
  b: number;
  boundaryLength: number;
  contacts: number;
};

export type DeepBlackBoxRelu2DBoundarySegment = {
  id: number;
  regionA: number;
  regionB: number;
  normal: [number, number];
  offset: number;
  start: [number, number];
  end: [number, number];
  length: number;
  topologyStart: [number, number];
  topologyEnd: [number, number];
  topologyLength: number;
  contactCount: number;
  fitRmse: number;
  endpointLineResidual: number;
  topologyPreserving: boolean;
  lineClipped: boolean;
  domainClipped: boolean;
  contactCorridorArea: number;
};

export type DeepBlackBoxRelu2DRegionGraphModel = {
  xRange: [number, number];
  yRange: [number, number];
  gridWidth: number;
  gridHeight: number;
  gradientStep: number;
  querySamples: number;
  aggregation: 'mean' | 'median';
  recoveryKind: 'output-equivalent-region-graph';
  regions: DeepBlackBoxRelu2DRegion[];
  adjacency: DeepBlackBoxRelu2DAdjacency[];
  boundarySegments: DeepBlackBoxRelu2DBoundarySegment[];
  cellRegionIds: number[][];
  kinkCellBoundaryCount: number;
  compactBoundarySegmentCount: number;
  regionCount: number;
  adaptiveProbe?: DeepBlackBoxRelu2DAdaptiveProbeReport;
  boundaryRefinement?: DeepBlackBoxRelu2DBoundaryRefinementReport;
  subcellBoundaryProbe?: DeepBlackBoxRelu2DSubcellBoundaryProbeReport;
  topologyGeometry?: DeepBlackBoxRelu2DTopologyGeometryReport;
  expression: string;
};

export type DeepBlackBoxRelu2DRegionGraphConfig = {
  xRange: [number, number];
  yRange: [number, number];
  gridWidth?: number;
  gridHeight?: number;
  gradientStep?: number;
  slopeTolerance?: number;
  interceptTolerance?: number;
  querySamples?: number;
  aggregation?: 'mean' | 'median';
};

export type DeepBlackBoxRelu2DAdaptiveProbeAttempt = {
  gridWidth: number;
  gridHeight: number;
  decompileOracleCalls: number;
  validationOracleCalls: number;
  totalOracleCalls: number;
  validationMae: number;
  validationR2: number;
  regionCount: number;
  adjacencyEdgeCount: number;
  kinkCellBoundaryCount: number;
  compactBoundarySegmentCount: number;
  boundaryCompressionRatio: number;
  maxBoundaryFitRmse: number;
  pass: boolean;
};

export type DeepBlackBoxRelu2DAdaptiveProbeReport = {
  enabled: true;
  candidateGridSizes: number[];
  validationPointCount: number;
  selectedAttempt: DeepBlackBoxRelu2DAdaptiveProbeAttempt;
  attempts: DeepBlackBoxRelu2DAdaptiveProbeAttempt[];
  totalOracleCalls: number;
};

export type DeepBlackBoxRelu2DAdaptiveProbeConfig = {
  candidateGridSizes?: number[];
  validationGridWidth?: number;
  validationGridHeight?: number;
  maxValidationMae?: number;
  minValidationR2?: number;
  maxBoundaryFitRmse?: number;
  minCompressionRatio?: number;
  minRegionCount?: number;
  minAdjacencyEdges?: number;
};

export type DeepBlackBoxRelu2DBoundaryRefinementReport = {
  enabled: true;
  predictionMode: 'boundary-aware-local-switch';
  marginFactor: number;
  projectionSlack: number;
  boundaryMargin: number;
  boundarySegmentCount: number;
  validationPointCount: number;
  decompileOracleCalls: number;
  validationOracleCalls: number;
  totalOracleCalls: number;
  coarseValidationMae: number;
  coarseValidationR2: number;
  refinedValidationMae: number;
  refinedValidationR2: number;
  maeImprovementRatio: number;
  pass: boolean;
};

export type DeepBlackBoxRelu2DBoundaryRefinementConfig = {
  gridWidth?: number;
  gridHeight?: number;
  marginFactor?: number;
  projectionSlack?: number;
  validationGridWidth?: number;
  validationGridHeight?: number;
  maxRefinedValidationMae?: number;
  minRefinedValidationR2?: number;
  requireValidationImprovement?: boolean;
};

export type DeepBlackBoxRelu2DSubcellBoundaryProbeAttempt = {
  gridWidth: number;
  gridHeight: number;
  decompileOracleCalls: number;
  validationOracleCalls: number;
  totalOracleCalls: number;
  validationMae: number;
  validationR2: number;
  coarseValidationMae: number;
  coarseValidationR2: number;
  regionCount: number;
  adjacencyEdgeCount: number;
  compactBoundarySegmentCount: number;
  boundaryCompressionRatio: number;
  pass: boolean;
};

export type DeepBlackBoxRelu2DSubcellBoundaryProbeReport = {
  enabled: true;
  predictionMode: 'oracle-guided-subcell-boundary-grid-selection';
  candidateGridSizes: number[];
  selectedAttempt: DeepBlackBoxRelu2DSubcellBoundaryProbeAttempt;
  attempts: DeepBlackBoxRelu2DSubcellBoundaryProbeAttempt[];
  marginFactor: number;
  projectionSlack: number;
  validationPointCount: number;
  totalOracleCalls: number;
};

export type DeepBlackBoxRelu2DSubcellBoundaryProbeConfig = {
  candidateGridSizes?: number[];
  marginFactor?: number;
  projectionSlack?: number;
  validationGridWidth?: number;
  validationGridHeight?: number;
  maxValidationMae?: number;
  minValidationR2?: number;
  minRegionCount?: number;
  minAdjacencyEdges?: number;
  minBoundaryCompressionRatio?: number;
};

export type DeepBlackBoxRelu2DTopologyGeometryReport = {
  enabled: true;
  geometryKind: 'line-clipped-contact-corridor';
  boundarySegmentCount: number;
  adjacencyEdgeCount: number;
  topologyPreserved: boolean;
  lineClippedSegmentCount: number;
  fallbackSegmentCount: number;
  domainClippedSegmentCount: number;
  maxEndpointLineResidual: number;
  maxBoundaryFitRmse: number;
  maxNormalError: number;
  minSegmentLength: number;
  pass: boolean;
};

type RegionAccumulator = {
  id: number;
  interceptSum: number;
  slopeXSum: number;
  slopeYSum: number;
  valueSum: number;
  xxSum: number;
  xySum: number;
  yySum: number;
  xValueSum: number;
  yValueSum: number;
  xSum: number;
  ySum: number;
  support: number;
};

type AdjacencyAccumulator = DeepBlackBoxRelu2DAdjacency & {
  contactMidpoints: Array<[number, number]>;
};

function formatNumber(value: number): string {
  if (Math.abs(value) < 1e-12) return '0';
  if (Math.abs(value) >= 1e5 || Math.abs(value) < 1e-5) return value.toExponential(8);
  return value.toPrecision(10).replace(/\.?0+$/, '');
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function aggregate(values: number[], mode: 'mean' | 'median'): number {
  if (values.length === 0) return 0;
  if (mode === 'mean') return values.reduce((sum, value) => sum + value, 0) / values.length;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function regressionQuality(actual: number[], predicted: number[]): { mae: number; r2: number } {
  const count = Math.max(1, actual.length);
  const meanActual = actual.reduce((sum, value) => sum + value, 0) / count;
  let absoluteError = 0;
  let residual = 0;
  let total = 0;
  for (let index = 0; index < actual.length; index++) {
    const error = actual[index] - predicted[index];
    absoluteError += Math.abs(error);
    residual += error * error;
    const centered = actual[index] - meanActual;
    total += centered * centered;
  }
  return {
    mae: absoluteError / count,
    r2: total > 1e-20 ? 1 - residual / total : 1
  };
}

function makeValidationInputs(
  [xMin, xMax]: [number, number],
  [yMin, yMax]: [number, number],
  width: number,
  height: number
): Array<[number, number]> {
  const inputs: Array<[number, number]> = [];
  for (let xIndex = 0; xIndex < width; xIndex++) {
    for (let yIndex = 0; yIndex < height; yIndex++) {
      inputs.push([
        xMin + ((xMax - xMin) * (xIndex + 0.37)) / width,
        yMin + ((yMax - yMin) * (yIndex + 0.23)) / height
      ]);
    }
  }
  return inputs;
}

function regionCenter(region: RegionAccumulator): { intercept: number; slopeX: number; slopeY: number } {
  return {
    intercept: region.interceptSum / region.support,
    slopeX: region.slopeXSum / region.support,
    slopeY: region.slopeYSum / region.support
  };
}

function solve3x3(matrix: number[][], rhs: number[]): number[] | null {
  const a = matrix.map((row, index) => [...row, rhs[index]]);
  for (let col = 0; col < 3; col++) {
    let pivot = col;
    for (let row = col + 1; row < 3; row++) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const divisor = a[col][col];
    if (Math.abs(divisor) < 1e-10) return null;
    for (let j = col; j <= 3; j++) a[col][j] /= divisor;
    for (let row = 0; row < 3; row++) {
      if (row === col) continue;
      const factor = a[row][col];
      if (Math.abs(factor) < 1e-20) continue;
      for (let j = col; j <= 3; j++) a[row][j] -= factor * a[col][j];
    }
  }
  return a.map(row => row[3]);
}

function fitRegionPlane(region: RegionAccumulator): { intercept: number; slopeX: number; slopeY: number } {
  if (region.support < 3) return regionCenter(region);
  const solution = solve3x3(
    [
      [region.support, region.xSum, region.ySum],
      [region.xSum, region.xxSum + 1e-10, region.xySum],
      [region.ySum, region.xySum, region.yySum + 1e-10]
    ],
    [region.valueSum, region.xValueSum, region.yValueSum]
  );
  if (!solution || solution.some(value => !Number.isFinite(value))) return regionCenter(region);
  return {
    intercept: solution[0],
    slopeX: solution[1],
    slopeY: solution[2]
  };
}

function buildExpression(model: Omit<DeepBlackBoxRelu2DRegionGraphModel, 'expression'>): string {
  return [
    'region_graph(',
    `regions=${model.regionCount}`,
    `adjacency=${model.adjacency.length}`,
    `boundaries=${model.compactBoundarySegmentCount}`,
    `grid=${model.gridWidth}x${model.gridHeight}`,
    ')'
  ].join(' ');
}

function buildAdjacency(
  cellRegionIds: number[][],
  gridWidth: number,
  gridHeight: number,
  xRange: [number, number],
  yRange: [number, number],
  dx: number,
  dy: number
): { adjacency: DeepBlackBoxRelu2DAdjacency[]; adjacencySamples: AdjacencyAccumulator[]; kinkCellBoundaryCount: number } {
  const [xMin] = xRange;
  const [yMin] = yRange;
  const contacts = new Map<string, AdjacencyAccumulator>();
  let kinkCellBoundaryCount = 0;
  const addContact = (aRaw: number, bRaw: number, length: number, midpoint: [number, number]) => {
    if (aRaw === bRaw) return;
    kinkCellBoundaryCount++;
    const a = Math.min(aRaw, bRaw);
    const b = Math.max(aRaw, bRaw);
    const key = `${a}:${b}`;
    const current = contacts.get(key);
    if (current) {
      current.boundaryLength += length;
      current.contacts++;
      current.contactMidpoints.push(midpoint);
    } else {
      contacts.set(key, { a, b, boundaryLength: length, contacts: 1, contactMidpoints: [midpoint] });
    }
  };

  for (let yIndex = 0; yIndex < gridHeight; yIndex++) {
    for (let xIndex = 0; xIndex < gridWidth; xIndex++) {
      const id = cellRegionIds[yIndex][xIndex];
      if (xIndex + 1 < gridWidth) {
        addContact(
          id,
          cellRegionIds[yIndex][xIndex + 1],
          dy,
          [xMin + (xIndex + 1) * dx, yMin + (yIndex + 0.5) * dy]
        );
      }
      if (yIndex + 1 < gridHeight) {
        addContact(
          id,
          cellRegionIds[yIndex + 1][xIndex],
          dx,
          [xMin + (xIndex + 0.5) * dx, yMin + (yIndex + 1) * dy]
        );
      }
    }
  }

  const adjacencySamples = Array.from(contacts.values()).sort((a, b) => a.a - b.a || a.b - b.b);
  return {
    adjacency: adjacencySamples.map(({ contactMidpoints: _points, ...adjacency }) => adjacency),
    adjacencySamples,
    kinkCellBoundaryCount
  };
}

function clipLineToBox(
  point: [number, number],
  direction: [number, number],
  box: { xMin: number; xMax: number; yMin: number; yMax: number }
): [number, number] | null {
  let minT = Number.NEGATIVE_INFINITY;
  let maxT = Number.POSITIVE_INFINITY;
  const axes = [
    { p: point[0], d: direction[0], min: box.xMin, max: box.xMax },
    { p: point[1], d: direction[1], min: box.yMin, max: box.yMax }
  ];
  for (const axis of axes) {
    if (Math.abs(axis.d) < 1e-14) {
      if (axis.p < axis.min - 1e-12 || axis.p > axis.max + 1e-12) return null;
      continue;
    }
    const t1 = (axis.min - axis.p) / axis.d;
    const t2 = (axis.max - axis.p) / axis.d;
    minT = Math.max(minT, Math.min(t1, t2));
    maxT = Math.min(maxT, Math.max(t1, t2));
  }
  if (!Number.isFinite(minT) || !Number.isFinite(maxT) || maxT <= minT) return null;
  return [minT, maxT];
}

function buildBoundarySegments(
  adjacencySamples: AdjacencyAccumulator[],
  regions: DeepBlackBoxRelu2DRegion[],
  xRange: [number, number],
  yRange: [number, number],
  dx: number,
  dy: number
): DeepBlackBoxRelu2DBoundarySegment[] {
  const [xMin, xMax] = xRange;
  const [yMin, yMax] = yRange;
  return adjacencySamples.map((edge, index) => {
    const a = regions[edge.a];
    const b = regions[edge.b];
    const deltaSlopeX = a.slopeX - b.slopeX;
    const deltaSlopeY = a.slopeY - b.slopeY;
    const deltaIntercept = a.intercept - b.intercept;
    const norm = Math.hypot(deltaSlopeX, deltaSlopeY);
    const normal: [number, number] = norm > 1e-12
      ? [deltaSlopeX / norm, deltaSlopeY / norm]
      : [0, 1];
    const offset = norm > 1e-12 ? deltaIntercept / norm : -(a.centroid[1] + b.centroid[1]) / 2;
    const tangent: [number, number] = [-normal[1], normal[0]];
    const pointOnLine: [number, number] = [-normal[0] * offset, -normal[1] * offset];
    const projected = edge.contactMidpoints.map(([x, y]) => {
      const signedDistance = normal[0] * x + normal[1] * y + offset;
      const px = x - normal[0] * signedDistance;
      const py = y - normal[1] * signedDistance;
      return {
        rawX: px,
        rawY: py,
        x: clamp(px, xMin, xMax),
        y: clamp(py, yMin, yMax),
        t: px * tangent[0] + py * tangent[1],
        signedDistance
      };
    });
    const tValues = projected.map(point => point.t);
    let minT = Math.min(...tValues);
    let maxT = Math.max(...tValues);
    if (Math.abs(maxT - minT) < 1e-12) {
      const halfLength = Math.max(edge.boundaryLength, Math.min(xMax - xMin, yMax - yMin) * 1e-3) / 2;
      minT -= halfLength;
      maxT += halfLength;
    }
    const maxProjectionDistance = projected.length > 0
      ? Math.max(...projected.map(point => Math.abs(point.signedDistance)))
      : 0;
    const corridorXs = [
      ...edge.contactMidpoints.map(([x]) => x),
      ...projected.map(point => point.rawX),
      ...projected.map(point => point.x)
    ];
    const corridorYs = [
      ...edge.contactMidpoints.map(([, y]) => y),
      ...projected.map(point => point.rawY),
      ...projected.map(point => point.y)
    ];
    const xPadding = dx / 2 + Math.min(xMax - xMin, maxProjectionDistance * Math.abs(normal[0]) + dx * 1e-6);
    const yPadding = dy / 2 + Math.min(yMax - yMin, maxProjectionDistance * Math.abs(normal[1]) + dy * 1e-6);
    const corridor = {
      xMin: clamp(Math.min(...corridorXs) - xPadding, xMin, xMax),
      xMax: clamp(Math.max(...corridorXs) + xPadding, xMin, xMax),
      yMin: clamp(Math.min(...corridorYs) - yPadding, yMin, yMax),
      yMax: clamp(Math.max(...corridorYs) + yPadding, yMin, yMax)
    };
    const clipped = clipLineToBox(pointOnLine, tangent, corridor);
    const start: [number, number] = [
      clamp(pointOnLine[0] + tangent[0] * minT, xMin, xMax),
      clamp(pointOnLine[1] + tangent[1] * minT, yMin, yMax)
    ];
    const end: [number, number] = [
      clamp(pointOnLine[0] + tangent[0] * maxT, xMin, xMax),
      clamp(pointOnLine[1] + tangent[1] * maxT, yMin, yMax)
    ];
    const topologyStart: [number, number] = clipped
      ? [pointOnLine[0] + tangent[0] * clipped[0], pointOnLine[1] + tangent[1] * clipped[0]]
      : start;
    const topologyEnd: [number, number] = clipped
      ? [pointOnLine[0] + tangent[0] * clipped[1], pointOnLine[1] + tangent[1] * clipped[1]]
      : end;
    const endpointLineResidual = Math.max(
      Math.abs(normal[0] * topologyStart[0] + normal[1] * topologyStart[1] + offset),
      Math.abs(normal[0] * topologyEnd[0] + normal[1] * topologyEnd[1] + offset)
    );
    const fitRmse = Math.sqrt(
      projected.reduce((sum, point) => sum + point.signedDistance ** 2, 0) / Math.max(1, projected.length)
    );
    const length = Math.hypot(end[0] - start[0], end[1] - start[1]);
    const topologyLength = Math.hypot(topologyEnd[0] - topologyStart[0], topologyEnd[1] - topologyStart[1]);
    const contactCorridorArea = Math.max(0, corridor.xMax - corridor.xMin) * Math.max(0, corridor.yMax - corridor.yMin);
    const domainClipped = corridor.xMin <= xMin + 1e-12
      || corridor.xMax >= xMax - 1e-12
      || corridor.yMin <= yMin + 1e-12
      || corridor.yMax >= yMax - 1e-12;
    return {
      id: index,
      regionA: edge.a,
      regionB: edge.b,
      normal,
      offset,
      start,
      end,
      length,
      topologyStart,
      topologyEnd,
      topologyLength,
      contactCount: edge.contacts,
      fitRmse,
      endpointLineResidual,
      topologyPreserving: Boolean(clipped) && endpointLineResidual < 1e-9 && topologyLength > 0,
      lineClipped: Boolean(clipped),
      domainClipped,
      contactCorridorArea
    };
  });
}

function buildTopologyGeometryReport(
  model: Omit<DeepBlackBoxRelu2DRegionGraphModel, 'expression' | 'topologyGeometry'>
): DeepBlackBoxRelu2DTopologyGeometryReport {
  const endpointResiduals = model.boundarySegments.map(segment => segment.endpointLineResidual);
  const fitRmses = model.boundarySegments.map(segment => segment.fitRmse);
  const normalErrors = model.boundarySegments.map(segment => Math.abs(Math.hypot(...segment.normal) - 1));
  const lengths = model.boundarySegments.map(segment => segment.topologyLength);
  const topologyPreserved = model.boundarySegments.length === model.adjacency.length
    && model.boundarySegments.every(segment => segment.topologyPreserving);
  const report: DeepBlackBoxRelu2DTopologyGeometryReport = {
    enabled: true,
    geometryKind: 'line-clipped-contact-corridor',
    boundarySegmentCount: model.boundarySegments.length,
    adjacencyEdgeCount: model.adjacency.length,
    topologyPreserved,
    lineClippedSegmentCount: model.boundarySegments.filter(segment => segment.lineClipped).length,
    fallbackSegmentCount: model.boundarySegments.filter(segment => !segment.lineClipped).length,
    domainClippedSegmentCount: model.boundarySegments.filter(segment => segment.domainClipped).length,
    maxEndpointLineResidual: endpointResiduals.length > 0 ? Math.max(...endpointResiduals) : Number.POSITIVE_INFINITY,
    maxBoundaryFitRmse: fitRmses.length > 0 ? Math.max(...fitRmses) : Number.POSITIVE_INFINITY,
    maxNormalError: normalErrors.length > 0 ? Math.max(...normalErrors) : Number.POSITIVE_INFINITY,
    minSegmentLength: lengths.length > 0 ? Math.min(...lengths) : 0,
    pass: false
  };
  return {
    ...report,
    pass: report.topologyPreserved
      && report.fallbackSegmentCount === 0
      && report.maxEndpointLineResidual < 1e-9
      && report.maxNormalError < 1e-9
      && report.minSegmentLength > 0
  };
}

export function decompileDeepBlackBoxRelu2DRegionGraph(
  oracle: (x: number, y: number) => number,
  config: DeepBlackBoxRelu2DRegionGraphConfig
): DeepBlackBoxRelu2DRegionGraphModel {
  const [xMin, xMax] = config.xRange;
  const [yMin, yMax] = config.yRange;
  if (!(xMax > xMin) || !(yMax > yMin)) throw new Error('2D deep ReLU region graph requires valid x/y ranges');

  const gridWidth = Math.max(16, Math.floor(config.gridWidth ?? 101));
  const gridHeight = Math.max(16, Math.floor(config.gridHeight ?? 101));
  const dx = (xMax - xMin) / gridWidth;
  const dy = (yMax - yMin) / gridHeight;
  const requestedGradientStep = config.gradientStep ?? Math.min(dx, dy) * 1e-3;
  const maxGradientStep = Math.min(xMax - xMin, yMax - yMin) * 0.25;
  const gradientStep = clamp(requestedGradientStep, 1e-12, maxGradientStep);
  const slopeTolerance = config.slopeTolerance ?? 1e-5;
  const interceptTolerance = config.interceptTolerance ?? 1e-5;
  const querySamples = Math.max(1, Math.floor(config.querySamples ?? 1));
  const aggregation = config.aggregation ?? 'mean';
  if (aggregation !== 'mean' && aggregation !== 'median') throw new Error('2D deep ReLU region graph aggregation must be mean or median');

  const sampleOracle = (x: number, y: number): number => {
    if (querySamples === 1) return oracle(x, y);
    const values = Array.from({ length: querySamples }, () => oracle(x, y));
    return aggregate(values, aggregation);
  };

  const accumulators: RegionAccumulator[] = [];
  const cellRegionIds: number[][] = [];

  for (let yIndex = 0; yIndex < gridHeight; yIndex++) {
    const row: number[] = [];
    const y = yMin + (yIndex + 0.5) * dy;
    for (let xIndex = 0; xIndex < gridWidth; xIndex++) {
      const x = xMin + (xIndex + 0.5) * dx;
      const xLeft = clamp(x - gradientStep, xMin, xMax);
      const xRight = clamp(x + gradientStep, xMin, xMax);
      const yDown = clamp(y - gradientStep, yMin, yMax);
      const yUp = clamp(y + gradientStep, yMin, yMax);
      const center = sampleOracle(x, y);
      const slopeX = (sampleOracle(xRight, y) - sampleOracle(xLeft, y)) / Math.max(1e-12, xRight - xLeft);
      const slopeY = (sampleOracle(x, yUp) - sampleOracle(x, yDown)) / Math.max(1e-12, yUp - yDown);
      const intercept = center - slopeX * x - slopeY * y;

      let region = accumulators.find(candidate => {
        const centerRegion = regionCenter(candidate);
        return Math.abs(centerRegion.slopeX - slopeX) <= slopeTolerance
          && Math.abs(centerRegion.slopeY - slopeY) <= slopeTolerance
          && Math.abs(centerRegion.intercept - intercept) <= interceptTolerance;
      });
      if (!region) {
        region = {
          id: accumulators.length,
          interceptSum: 0,
          slopeXSum: 0,
          slopeYSum: 0,
          valueSum: 0,
          xxSum: 0,
          xySum: 0,
          yySum: 0,
          xValueSum: 0,
          yValueSum: 0,
          xSum: 0,
          ySum: 0,
          support: 0
        };
        accumulators.push(region);
      }

      region.interceptSum += intercept;
      region.slopeXSum += slopeX;
      region.slopeYSum += slopeY;
      region.valueSum += center;
      region.xxSum += x * x;
      region.xySum += x * y;
      region.yySum += y * y;
      region.xValueSum += x * center;
      region.yValueSum += y * center;
      region.xSum += x;
      region.ySum += y;
      region.support++;
      row.push(region.id);
    }
    cellRegionIds.push(row);
  }

  const regions = accumulators.map(region => {
    const centerRegion = fitRegionPlane(region);
    return {
      id: region.id,
      intercept: centerRegion.intercept,
      slopeX: centerRegion.slopeX,
      slopeY: centerRegion.slopeY,
      support: region.support,
      centroid: [region.xSum / region.support, region.ySum / region.support] as [number, number]
    };
  });
  const { adjacency, adjacencySamples, kinkCellBoundaryCount } = buildAdjacency(
    cellRegionIds,
    gridWidth,
    gridHeight,
    config.xRange,
    config.yRange,
    dx,
    dy
  );
  const boundarySegments = buildBoundarySegments(adjacencySamples, regions, config.xRange, config.yRange, dx, dy);
  const modelBase = {
    xRange: config.xRange,
    yRange: config.yRange,
    gridWidth,
    gridHeight,
    gradientStep,
    querySamples,
    aggregation,
    recoveryKind: 'output-equivalent-region-graph' as const,
    regions,
    adjacency,
    boundarySegments,
    cellRegionIds,
    kinkCellBoundaryCount,
    compactBoundarySegmentCount: boundarySegments.length,
    regionCount: regions.length
  };
  const topologyGeometry = buildTopologyGeometryReport(modelBase);

  return {
    ...modelBase,
    topologyGeometry,
    expression: buildExpression(modelBase)
  };
}

export function decompileDeepBlackBoxRelu2DRegionGraphAdaptive(
  oracle: (x: number, y: number) => number,
  config: DeepBlackBoxRelu2DRegionGraphConfig,
  adaptiveConfig: DeepBlackBoxRelu2DAdaptiveProbeConfig = {}
): DeepBlackBoxRelu2DRegionGraphModel {
  const candidateGridSizes = Array.from(new Set(adaptiveConfig.candidateGridSizes ?? [89, 101]))
    .map(value => Math.max(16, Math.floor(value)))
    .sort((a, b) => a - b);
  if (candidateGridSizes.length === 0) throw new Error('adaptive deep ReLU 2D probing requires at least one candidate grid size');

  const validationGridWidth = Math.max(8, Math.floor(adaptiveConfig.validationGridWidth ?? 37));
  const validationGridHeight = Math.max(8, Math.floor(adaptiveConfig.validationGridHeight ?? 31));
  const maxValidationMae = adaptiveConfig.maxValidationMae ?? 0.001;
  const minValidationR2 = adaptiveConfig.minValidationR2 ?? 0.99999;
  const maxBoundaryFitRmse = adaptiveConfig.maxBoundaryFitRmse ?? 0.08;
  const minCompressionRatio = adaptiveConfig.minCompressionRatio ?? 8;
  const minRegionCount = adaptiveConfig.minRegionCount ?? 18;
  const minAdjacencyEdges = adaptiveConfig.minAdjacencyEdges ?? 35;
  const validationInputs = makeValidationInputs(config.xRange, config.yRange, validationGridWidth, validationGridHeight);

  let selectedModel: DeepBlackBoxRelu2DRegionGraphModel | null = null;
  let selectedAttempt: DeepBlackBoxRelu2DAdaptiveProbeAttempt | null = null;
  const attempts: DeepBlackBoxRelu2DAdaptiveProbeAttempt[] = [];

  for (const gridSize of candidateGridSizes) {
    let decompileOracleCalls = 0;
    const countedOracle = (x: number, y: number): number => {
      decompileOracleCalls++;
      return oracle(x, y);
    };
    const model = decompileDeepBlackBoxRelu2DRegionGraph(countedOracle, {
      ...config,
      gridWidth: gridSize,
      gridHeight: gridSize
    });

    let validationOracleCalls = 0;
    const actual = validationInputs.map(([x, y]) => {
      validationOracleCalls++;
      return oracle(x, y);
    });
    const predicted = predictDeepBlackBoxRelu2DRegionGraphBatch(model, validationInputs);
    const metrics = regressionQuality(actual, predicted);
    const boundaryCompressionRatio = model.kinkCellBoundaryCount / Math.max(1, model.compactBoundarySegmentCount);
    const observedMaxBoundaryFitRmse = model.boundarySegments.length > 0
      ? Math.max(...model.boundarySegments.map(segment => segment.fitRmse))
      : Number.POSITIVE_INFINITY;
    const attempt: DeepBlackBoxRelu2DAdaptiveProbeAttempt = {
      gridWidth: gridSize,
      gridHeight: gridSize,
      decompileOracleCalls,
      validationOracleCalls,
      totalOracleCalls: decompileOracleCalls + validationOracleCalls,
      validationMae: metrics.mae,
      validationR2: metrics.r2,
      regionCount: model.regionCount,
      adjacencyEdgeCount: model.adjacency.length,
      kinkCellBoundaryCount: model.kinkCellBoundaryCount,
      compactBoundarySegmentCount: model.compactBoundarySegmentCount,
      boundaryCompressionRatio,
      maxBoundaryFitRmse: observedMaxBoundaryFitRmse,
      pass: metrics.mae < maxValidationMae
        && metrics.r2 > minValidationR2
        && boundaryCompressionRatio > minCompressionRatio
        && observedMaxBoundaryFitRmse < maxBoundaryFitRmse
        && model.regionCount >= minRegionCount
        && model.adjacency.length >= minAdjacencyEdges
    };
    attempts.push(attempt);

    if (!selectedAttempt || attempt.validationMae < selectedAttempt.validationMae) {
      selectedAttempt = attempt;
      selectedModel = model;
    }
    if (attempt.pass) {
      selectedAttempt = attempt;
      selectedModel = model;
      break;
    }
  }

  if (!selectedModel || !selectedAttempt) throw new Error('adaptive deep ReLU 2D probing failed to produce a model');
  const adaptiveProbe: DeepBlackBoxRelu2DAdaptiveProbeReport = {
    enabled: true,
    candidateGridSizes,
    validationPointCount: validationInputs.length,
    selectedAttempt,
    attempts,
    totalOracleCalls: attempts.reduce((sum, attempt) => sum + attempt.totalOracleCalls, 0)
  };
  return {
    ...selectedModel,
    adaptiveProbe
  };
}

export function decompileDeepBlackBoxRelu2DRegionGraphBoundaryRefined(
  oracle: (x: number, y: number) => number,
  config: DeepBlackBoxRelu2DRegionGraphConfig,
  refinementConfig: DeepBlackBoxRelu2DBoundaryRefinementConfig = {}
): DeepBlackBoxRelu2DRegionGraphModel {
  const gridWidth = Math.max(16, Math.floor(refinementConfig.gridWidth ?? config.gridWidth ?? 57));
  const gridHeight = Math.max(16, Math.floor(refinementConfig.gridHeight ?? config.gridHeight ?? gridWidth));
  const marginFactor = refinementConfig.marginFactor ?? 0.5;
  const projectionSlack = refinementConfig.projectionSlack ?? 0.1;
  const validationGridWidth = Math.max(8, Math.floor(refinementConfig.validationGridWidth ?? 37));
  const validationGridHeight = Math.max(8, Math.floor(refinementConfig.validationGridHeight ?? 31));
  const maxRefinedValidationMae = refinementConfig.maxRefinedValidationMae ?? 0.0003;
  const minRefinedValidationR2 = refinementConfig.minRefinedValidationR2 ?? 0.99999;
  const requireValidationImprovement = refinementConfig.requireValidationImprovement ?? true;

  let decompileOracleCalls = 0;
  const countedOracle = (x: number, y: number): number => {
    decompileOracleCalls++;
    return oracle(x, y);
  };
  const model = decompileDeepBlackBoxRelu2DRegionGraph(countedOracle, {
    ...config,
    gridWidth,
    gridHeight
  });

  let validationOracleCalls = 0;
  const validationInputs = makeValidationInputs(config.xRange, config.yRange, validationGridWidth, validationGridHeight);
  const actual = validationInputs.map(([x, y]) => {
    validationOracleCalls++;
    return oracle(x, y);
  });
  const coarsePredicted = validationInputs.map(([x, y]) => predictDeepBlackBoxRelu2DRegionGraphCoarseModel(model, x, y));
  const refinedPredicted = validationInputs.map(([x, y]) => (
    predictDeepBlackBoxRelu2DRegionGraphBoundaryAwareModel(model, x, y, marginFactor, projectionSlack)
  ));
  const coarseMetrics = regressionQuality(actual, coarsePredicted);
  const refinedMetrics = regressionQuality(actual, refinedPredicted);
  const [xMin, xMax] = model.xRange;
  const [yMin, yMax] = model.yRange;
  const boundaryMargin = marginFactor * Math.hypot((xMax - xMin) / model.gridWidth, (yMax - yMin) / model.gridHeight);
  const improvesValidation = refinedMetrics.mae < coarseMetrics.mae
    || refinedMetrics.r2 > coarseMetrics.r2
    || coarseMetrics.mae <= maxRefinedValidationMae;
  const report: DeepBlackBoxRelu2DBoundaryRefinementReport = {
    enabled: true,
    predictionMode: 'boundary-aware-local-switch',
    marginFactor,
    projectionSlack,
    boundaryMargin,
    boundarySegmentCount: model.boundarySegments.length,
    validationPointCount: validationInputs.length,
    decompileOracleCalls,
    validationOracleCalls,
    totalOracleCalls: decompileOracleCalls + validationOracleCalls,
    coarseValidationMae: coarseMetrics.mae,
    coarseValidationR2: coarseMetrics.r2,
    refinedValidationMae: refinedMetrics.mae,
    refinedValidationR2: refinedMetrics.r2,
    maeImprovementRatio: coarseMetrics.mae / Math.max(1e-12, refinedMetrics.mae),
    pass: refinedMetrics.mae < maxRefinedValidationMae
      && refinedMetrics.r2 > minRefinedValidationR2
      && model.boundarySegments.length > 0
      && (!requireValidationImprovement || improvesValidation)
  };

  return {
    ...model,
    boundaryRefinement: report,
    expression: `${model.expression} boundary_refined(margin=${formatNumber(marginFactor)})`
  };
}

export function decompileDeepBlackBoxRelu2DRegionGraphSubcellBoundaryAdaptive(
  oracle: (x: number, y: number) => number,
  config: DeepBlackBoxRelu2DRegionGraphConfig,
  probeConfig: DeepBlackBoxRelu2DSubcellBoundaryProbeConfig = {}
): DeepBlackBoxRelu2DRegionGraphModel {
  const candidateGridSizes = Array.from(new Set(probeConfig.candidateGridSizes ?? [37, 41, 49, 57]))
    .map(value => Math.max(16, Math.floor(value)))
    .sort((a, b) => a - b);
  if (candidateGridSizes.length === 0) throw new Error('subcell boundary probing requires at least one candidate grid size');

  const marginFactor = probeConfig.marginFactor ?? 0.5;
  const projectionSlack = probeConfig.projectionSlack ?? 0.1;
  const validationGridWidth = Math.max(8, Math.floor(probeConfig.validationGridWidth ?? 37));
  const validationGridHeight = Math.max(8, Math.floor(probeConfig.validationGridHeight ?? 31));
  const maxValidationMae = probeConfig.maxValidationMae ?? 0.0003;
  const minValidationR2 = probeConfig.minValidationR2 ?? 0.99999;
  const minRegionCount = probeConfig.minRegionCount ?? 18;
  const minAdjacencyEdges = probeConfig.minAdjacencyEdges ?? 35;
  const minBoundaryCompressionRatio = probeConfig.minBoundaryCompressionRatio ?? 5;
  const validationInputs = makeValidationInputs(config.xRange, config.yRange, validationGridWidth, validationGridHeight);

  let selectedModel: DeepBlackBoxRelu2DRegionGraphModel | null = null;
  let selectedAttempt: DeepBlackBoxRelu2DSubcellBoundaryProbeAttempt | null = null;
  const attempts: DeepBlackBoxRelu2DSubcellBoundaryProbeAttempt[] = [];

  for (const gridSize of candidateGridSizes) {
    let decompileOracleCalls = 0;
    const countedOracle = (x: number, y: number): number => {
      decompileOracleCalls++;
      return oracle(x, y);
    };
    const model = decompileDeepBlackBoxRelu2DRegionGraph(countedOracle, {
      ...config,
      gridWidth: gridSize,
      gridHeight: gridSize
    });

    let validationOracleCalls = 0;
    const actual = validationInputs.map(([x, y]) => {
      validationOracleCalls++;
      return oracle(x, y);
    });
    const coarsePredicted = validationInputs.map(([x, y]) => predictDeepBlackBoxRelu2DRegionGraphCoarseModel(model, x, y));
    const boundaryPredicted = validationInputs.map(([x, y]) => (
      predictDeepBlackBoxRelu2DRegionGraphBoundaryAwareModel(model, x, y, marginFactor, projectionSlack)
    ));
    const coarseMetrics = regressionQuality(actual, coarsePredicted);
    const boundaryMetrics = regressionQuality(actual, boundaryPredicted);
    const boundaryCompressionRatio = model.kinkCellBoundaryCount / Math.max(1, model.compactBoundarySegmentCount);
    const attempt: DeepBlackBoxRelu2DSubcellBoundaryProbeAttempt = {
      gridWidth: gridSize,
      gridHeight: gridSize,
      decompileOracleCalls,
      validationOracleCalls,
      totalOracleCalls: decompileOracleCalls + validationOracleCalls,
      validationMae: boundaryMetrics.mae,
      validationR2: boundaryMetrics.r2,
      coarseValidationMae: coarseMetrics.mae,
      coarseValidationR2: coarseMetrics.r2,
      regionCount: model.regionCount,
      adjacencyEdgeCount: model.adjacency.length,
      compactBoundarySegmentCount: model.compactBoundarySegmentCount,
      boundaryCompressionRatio,
      pass: boundaryMetrics.mae < maxValidationMae
        && boundaryMetrics.r2 > minValidationR2
        && model.regionCount >= minRegionCount
        && model.adjacency.length >= minAdjacencyEdges
        && boundaryCompressionRatio > minBoundaryCompressionRatio
    };
    attempts.push(attempt);

    if (!selectedAttempt || boundaryMetrics.mae < selectedAttempt.validationMae) {
      selectedAttempt = attempt;
      selectedModel = model;
    }
    if (attempt.pass) {
      selectedAttempt = attempt;
      selectedModel = model;
      break;
    }
  }

  if (!selectedModel || !selectedAttempt) throw new Error('subcell boundary probing failed to produce a model');
  const [xMin, xMax] = selectedModel.xRange;
  const [yMin, yMax] = selectedModel.yRange;
  const boundaryMargin = marginFactor * Math.hypot((xMax - xMin) / selectedModel.gridWidth, (yMax - yMin) / selectedModel.gridHeight);
  const boundaryRefinement: DeepBlackBoxRelu2DBoundaryRefinementReport = {
    enabled: true,
    predictionMode: 'boundary-aware-local-switch',
    marginFactor,
    projectionSlack,
    boundaryMargin,
    boundarySegmentCount: selectedModel.boundarySegments.length,
    validationPointCount: validationInputs.length,
    decompileOracleCalls: selectedAttempt.decompileOracleCalls,
    validationOracleCalls: selectedAttempt.validationOracleCalls,
    totalOracleCalls: selectedAttempt.totalOracleCalls,
    coarseValidationMae: selectedAttempt.coarseValidationMae,
    coarseValidationR2: selectedAttempt.coarseValidationR2,
    refinedValidationMae: selectedAttempt.validationMae,
    refinedValidationR2: selectedAttempt.validationR2,
    maeImprovementRatio: selectedAttempt.coarseValidationMae / Math.max(1e-12, selectedAttempt.validationMae),
    pass: selectedAttempt.pass
  };
  const subcellBoundaryProbe: DeepBlackBoxRelu2DSubcellBoundaryProbeReport = {
    enabled: true,
    predictionMode: 'oracle-guided-subcell-boundary-grid-selection',
    candidateGridSizes,
    selectedAttempt,
    attempts,
    marginFactor,
    projectionSlack,
    validationPointCount: validationInputs.length,
    totalOracleCalls: attempts.reduce((sum, attempt) => sum + attempt.totalOracleCalls, 0)
  };
  return {
    ...selectedModel,
    boundaryRefinement,
    subcellBoundaryProbe,
    expression: `${selectedModel.expression} subcell_boundary_probe(grid=${selectedAttempt.gridWidth},margin=${formatNumber(marginFactor)})`
  };
}

function coarseRegionId(model: DeepBlackBoxRelu2DRegionGraphModel, x: number, y: number): number {
  const [xMin, xMax] = model.xRange;
  const [yMin, yMax] = model.yRange;
  const xIndex = clamp(Math.floor(((x - xMin) / (xMax - xMin)) * model.gridWidth), 0, model.gridWidth - 1);
  const yIndex = clamp(Math.floor(((y - yMin) / (yMax - yMin)) * model.gridHeight), 0, model.gridHeight - 1);
  return model.cellRegionIds[yIndex][xIndex];
}

function regionValue(region: DeepBlackBoxRelu2DRegion, x: number, y: number): number {
  const value = region.intercept + region.slopeX * x + region.slopeY * y;
  return Number.isFinite(value) ? value : 0;
}

function selectBoundaryAwareRegionId(
  model: DeepBlackBoxRelu2DRegionGraphModel,
  x: number,
  y: number,
  marginFactor: number,
  projectionSlack: number
): number {
  if (model.boundarySegments.length === 0) return coarseRegionId(model, x, y);
  const [xMin, xMax] = model.xRange;
  const [yMin, yMax] = model.yRange;
  const boundaryMargin = Math.max(
    1e-12,
    marginFactor * Math.hypot((xMax - xMin) / model.gridWidth, (yMax - yMin) / model.gridHeight)
  );
  let regionId = coarseRegionId(model, x, y);
  for (let iteration = 0; iteration < 4; iteration++) {
    const currentRegion = model.regions[regionId];
    let bestSwitch: { regionId: number; distance: number } | null = null;
    for (const segment of model.boundarySegments) {
      if (segment.regionA !== regionId && segment.regionB !== regionId) continue;
      const dx = segment.end[0] - segment.start[0];
      const dy = segment.end[1] - segment.start[1];
      const lengthSquared = dx * dx + dy * dy;
      if (lengthSquared <= 1e-20) continue;
      const projection = ((x - segment.start[0]) * dx + (y - segment.start[1]) * dy) / lengthSquared;
      if (projection < -projectionSlack || projection > 1 + projectionSlack) continue;
      const signedDistance = segment.normal[0] * x + segment.normal[1] * y + segment.offset;
      const currentSide = segment.normal[0] * currentRegion.centroid[0]
        + segment.normal[1] * currentRegion.centroid[1]
        + segment.offset;
      if (Math.abs(signedDistance) > boundaryMargin || signedDistance * currentSide >= 0) continue;
      const otherRegionId = segment.regionA === regionId ? segment.regionB : segment.regionA;
      const distance = Math.abs(signedDistance);
      if (!bestSwitch || distance < bestSwitch.distance) bestSwitch = { regionId: otherRegionId, distance };
    }
    if (!bestSwitch || bestSwitch.regionId === regionId) break;
    regionId = bestSwitch.regionId;
  }
  return regionId;
}

export function predictDeepBlackBoxRelu2DRegionGraphCoarseModel(
  model: DeepBlackBoxRelu2DRegionGraphModel,
  x: number,
  y: number
): number {
  return regionValue(model.regions[coarseRegionId(model, x, y)], x, y);
}

export function predictDeepBlackBoxRelu2DRegionGraphBoundaryAwareModel(
  model: DeepBlackBoxRelu2DRegionGraphModel,
  x: number,
  y: number,
  marginFactor = model.boundaryRefinement?.marginFactor ?? 0.5,
  projectionSlack = model.boundaryRefinement?.projectionSlack ?? 0.1
): number {
  return regionValue(
    model.regions[selectBoundaryAwareRegionId(model, x, y, marginFactor, projectionSlack)],
    x,
    y
  );
}

export function predictDeepBlackBoxRelu2DRegionGraphModel(
  model: DeepBlackBoxRelu2DRegionGraphModel,
  x: number,
  y: number
): number {
  if (model.boundaryRefinement?.enabled) {
    return predictDeepBlackBoxRelu2DRegionGraphBoundaryAwareModel(
      model,
      x,
      y,
      model.boundaryRefinement.marginFactor,
      model.boundaryRefinement.projectionSlack
    );
  }
  return predictDeepBlackBoxRelu2DRegionGraphCoarseModel(model, x, y);
}

export function predictDeepBlackBoxRelu2DRegionGraphBatch(
  model: DeepBlackBoxRelu2DRegionGraphModel,
  inputs: Array<[number, number]>
): number[] {
  return inputs.map(([x, y]) => predictDeepBlackBoxRelu2DRegionGraphModel(model, x, y));
}
