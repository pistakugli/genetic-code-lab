import {
  DeepBlackBoxReluNdRegionSamplerAssignment,
  DeepBlackBoxReluNdRegionSamplerModel
} from './deep-blackbox-relu-nd-region-sampler';

export type DeepBlackBoxReluNdAdjacencyGeometryEdge = {
  id: number;
  regionA: number;
  regionB: number;
  contacts: number;
  witnessCount: number;
  transitionAgreement: number;
  normal: number[];
  offset: number;
  contactCentroid: number[];
  projectedCentroid: number[];
  nearestAssignmentDistance: number;
  meanAssignmentDistance: number;
  contactRadius: number;
  projectionRmse: number;
  normalizedProjectionRmse: number;
  maxProjectionResidual: number;
  minEndpointSignedGap: number;
  supportScore: number;
  pass: boolean;
};

export type DeepBlackBoxReluNdAdjacencyGeometryReport = {
  enabled: true;
  geometryKind: 'nd-affine-equality-contact-witness';
  dimension: number;
  regionCount: number;
  assignmentCount: number;
  sampledAssignmentCount: number;
  neighborCount: number;
  candidatePairCount: number;
  edgeCount: number;
  passingEdgeCount: number;
  largestComponentSize: number;
  largestComponentRatio: number;
  minContactCount: number;
  minWitnessCount: number;
  minTransitionAgreement: number;
  maxNormalizedProjectionRmse: number;
  maxProjectionRmse: number;
  minSupportScore: number;
  edges: DeepBlackBoxReluNdAdjacencyGeometryEdge[];
  pass: boolean;
};

export type DeepBlackBoxReluNdAdjacencyGeometryConfig = {
  assignmentSampleLimit?: number;
  neighborCount?: number;
  minContactCount?: number;
  minWitnessCount?: number;
  minTransitionAgreement?: number;
  maxNormalizedProjectionRmse?: number;
  minPassingEdges?: number;
  minLargestComponentRatio?: number;
};

type ContactWitness = {
  inputA: number[];
  inputB: number[];
  midpoint: number[];
  distance: number;
};

type EdgeAccumulator = {
  regionA: number;
  regionB: number;
  contacts: ContactWitness[];
};

function dot(a: number[], b: number[]): number {
  let value = 0;
  for (let index = 0; index < a.length; index++) value += a[index] * b[index];
  return value;
}

function distance(a: number[], b: number[]): number {
  let sum = 0;
  for (let index = 0; index < a.length; index++) {
    const delta = a[index] - b[index];
    sum += delta * delta;
  }
  return Math.sqrt(sum);
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function meanVector(values: number[][], dimension: number): number[] {
  const result = new Array(dimension).fill(0);
  for (const value of values) {
    for (let index = 0; index < dimension; index++) result[index] += value[index];
  }
  return result.map(value => value / Math.max(1, values.length));
}

function selectAssignments(
  assignments: DeepBlackBoxReluNdRegionSamplerAssignment[],
  limit: number
): DeepBlackBoxReluNdRegionSamplerAssignment[] {
  if (assignments.length <= limit) return assignments;
  const selected: DeepBlackBoxReluNdRegionSamplerAssignment[] = [];
  const stride = assignments.length / limit;
  for (let index = 0; index < limit; index++) selected.push(assignments[Math.floor(index * stride)]);
  return selected;
}

function insertNeighbor(
  neighbors: Array<{ assignment: DeepBlackBoxReluNdRegionSamplerAssignment; distance: number }>,
  candidate: { assignment: DeepBlackBoxReluNdRegionSamplerAssignment; distance: number },
  maxCount: number
): void {
  let index = 0;
  while (index < neighbors.length && neighbors[index].distance <= candidate.distance) index++;
  if (index >= maxCount) return;
  neighbors.splice(index, 0, candidate);
  if (neighbors.length > maxCount) neighbors.pop();
}

function edgeKey(aRaw: number, bRaw: number): string {
  return `${Math.min(aRaw, bRaw)}:${Math.max(aRaw, bRaw)}`;
}

function buildContactMap(
  assignments: DeepBlackBoxReluNdRegionSamplerAssignment[],
  neighborCount: number
): Map<string, EdgeAccumulator> {
  const edges = new Map<string, EdgeAccumulator>();
  for (let index = 0; index < assignments.length; index++) {
    const source = assignments[index];
    const neighbors: Array<{ assignment: DeepBlackBoxReluNdRegionSamplerAssignment; distance: number }> = [];
    for (let otherIndex = 0; otherIndex < assignments.length; otherIndex++) {
      if (otherIndex === index) continue;
      const other = assignments[otherIndex];
      if (other.regionId === source.regionId) continue;
      insertNeighbor(neighbors, { assignment: other, distance: distance(source.input, other.input) }, neighborCount);
    }
    for (const neighbor of neighbors) {
      const a = Math.min(source.regionId, neighbor.assignment.regionId);
      const b = Math.max(source.regionId, neighbor.assignment.regionId);
      const key = edgeKey(a, b);
      const inputA = source.regionId === a ? source.input : neighbor.assignment.input;
      const inputB = source.regionId === a ? neighbor.assignment.input : source.input;
      const midpoint = inputA.map((value, axis) => (value + inputB[axis]) / 2);
      const current = edges.get(key);
      const witness = { inputA, inputB, midpoint, distance: neighbor.distance };
      if (current) current.contacts.push(witness);
      else edges.set(key, { regionA: a, regionB: b, contacts: [witness] });
    }
  }
  return edges;
}

function connectedComponentSize(regionCount: number, edges: DeepBlackBoxReluNdAdjacencyGeometryEdge[]): number {
  const adjacency = Array.from({ length: regionCount }, () => [] as number[]);
  for (const edge of edges) {
    adjacency[edge.regionA].push(edge.regionB);
    adjacency[edge.regionB].push(edge.regionA);
  }
  const seen = new Array(regionCount).fill(false);
  let largest = 0;
  for (let start = 0; start < regionCount; start++) {
    if (seen[start]) continue;
    let size = 0;
    const stack = [start];
    seen[start] = true;
    while (stack.length > 0) {
      const node = stack.pop() ?? start;
      size++;
      for (const next of adjacency[node]) {
        if (seen[next]) continue;
        seen[next] = true;
        stack.push(next);
      }
    }
    largest = Math.max(largest, size);
  }
  return largest;
}

export function buildDeepBlackBoxReluNdAdjacencyGeometry(
  model: DeepBlackBoxReluNdRegionSamplerModel,
  config: DeepBlackBoxReluNdAdjacencyGeometryConfig = {}
): DeepBlackBoxReluNdAdjacencyGeometryReport {
  const dimension = model.dimension;
  const assignmentSampleLimit = Math.max(64, Math.floor(config.assignmentSampleLimit ?? 5000));
  const neighborCount = Math.max(1, Math.floor(config.neighborCount ?? Math.min(12, Math.max(4, dimension * 3))));
  const minContactCount = Math.max(1, Math.floor(config.minContactCount ?? Math.max(3, dimension + 1)));
  const minWitnessCount = Math.max(1, Math.floor(config.minWitnessCount ?? Math.max(2, dimension)));
  const minTransitionAgreement = config.minTransitionAgreement ?? 0.55;
  const maxNormalizedProjectionRmse = config.maxNormalizedProjectionRmse ?? 0.65;
  const minPassingEdges = Math.max(1, Math.floor(config.minPassingEdges ?? Math.max(3, model.regionCount - 1)));
  const minLargestComponentRatio = config.minLargestComponentRatio ?? 0.5;

  const sampledAssignments = selectAssignments(model.assignments, assignmentSampleLimit);
  const candidates = buildContactMap(sampledAssignments, neighborCount);
  const edges: DeepBlackBoxReluNdAdjacencyGeometryEdge[] = [];

  for (const candidate of candidates.values()) {
    const regionA = model.regions[candidate.regionA];
    const regionB = model.regions[candidate.regionB];
    if (!regionA || !regionB) continue;
    const deltaWeights = regionA.weights.map((value, index) => value - regionB.weights[index]);
    const deltaIntercept = regionA.intercept - regionB.intercept;
    const normalNorm = Math.sqrt(dot(deltaWeights, deltaWeights));
    if (normalNorm < 1e-12) continue;
    const normal = deltaWeights.map(value => value / normalNorm);
    const offset = deltaIntercept / normalNorm;
    const midpointResiduals = candidate.contacts.map(contact => dot(normal, contact.midpoint) + offset);
    const projected = candidate.contacts.map((contact, index) => (
      contact.midpoint.map((value, axis) => value - normal[axis] * midpointResiduals[index])
    ));
    const contactCentroid = meanVector(candidate.contacts.map(contact => contact.midpoint), dimension);
    const projectedCentroid = meanVector(projected, dimension);
    const distances = candidate.contacts.map(contact => contact.distance);
    const witnessFlags = candidate.contacts.map(contact => {
      const signedA = dot(normal, contact.inputA) + offset;
      const signedB = dot(normal, contact.inputB) + offset;
      const denominator = signedA - signedB;
      const t = Math.abs(denominator) > 1e-12 ? signedA / denominator : Number.NaN;
      return Number.isFinite(t) && t >= -0.2 && t <= 1.2 && signedA * signedB <= Math.max(1e-8, Math.abs(denominator) * 0.15);
    });
    const witnessCount = witnessFlags.filter(Boolean).length;
    const transitionAgreement = witnessCount / Math.max(1, candidate.contacts.length);
    const projectionRmse = Math.sqrt(mean(midpointResiduals.map(value => value * value)));
    const maxProjectionResidual = Math.max(...midpointResiduals.map(value => Math.abs(value)));
    const meanAssignmentDistance = mean(distances);
    const normalizedProjectionRmse = projectionRmse / Math.max(1e-12, meanAssignmentDistance);
    const endpointSignedGaps = candidate.contacts.map(contact => (
      Math.abs((dot(normal, contact.inputA) + offset) - (dot(normal, contact.inputB) + offset))
    ));
    const contactRadius = projected.length > 0
      ? Math.max(...projected.map(point => distance(point, projectedCentroid)))
      : 0;
    const supportScore = candidate.contacts.length * transitionAgreement / Math.max(1e-12, meanAssignmentDistance * (1 + normalizedProjectionRmse));
    const pass = candidate.contacts.length >= minContactCount
      && witnessCount >= minWitnessCount
      && transitionAgreement >= minTransitionAgreement
      && normalizedProjectionRmse <= maxNormalizedProjectionRmse;
    edges.push({
      id: edges.length,
      regionA: candidate.regionA,
      regionB: candidate.regionB,
      contacts: candidate.contacts.length,
      witnessCount,
      transitionAgreement,
      normal,
      offset,
      contactCentroid,
      projectedCentroid,
      nearestAssignmentDistance: Math.min(...distances),
      meanAssignmentDistance,
      contactRadius,
      projectionRmse,
      normalizedProjectionRmse,
      maxProjectionResidual,
      minEndpointSignedGap: Math.min(...endpointSignedGaps),
      supportScore,
      pass
    });
  }

  const passingEdges = edges
    .filter(edge => edge.pass)
    .sort((a, b) => b.supportScore - a.supportScore)
    .map((edge, index) => ({ ...edge, id: index }));
  const largestComponentSize = connectedComponentSize(model.regionCount, passingEdges);
  const largestComponentRatio = largestComponentSize / Math.max(1, model.regionCount);
  const maxObservedNormalizedProjectionRmse = passingEdges.length > 0
    ? Math.max(...passingEdges.map(edge => edge.normalizedProjectionRmse))
    : Number.POSITIVE_INFINITY;
  const maxProjectionRmse = passingEdges.length > 0
    ? Math.max(...passingEdges.map(edge => edge.projectionRmse))
    : Number.POSITIVE_INFINITY;
  const minSupportScore = passingEdges.length > 0
    ? Math.min(...passingEdges.map(edge => edge.supportScore))
    : 0;
  const pass = passingEdges.length >= minPassingEdges
    && largestComponentRatio >= minLargestComponentRatio
    && maxObservedNormalizedProjectionRmse <= maxNormalizedProjectionRmse;

  return {
    enabled: true,
    geometryKind: 'nd-affine-equality-contact-witness',
    dimension,
    regionCount: model.regionCount,
    assignmentCount: model.assignments.length,
    sampledAssignmentCount: sampledAssignments.length,
    neighborCount,
    candidatePairCount: candidates.size,
    edgeCount: passingEdges.length,
    passingEdgeCount: passingEdges.length,
    largestComponentSize,
    largestComponentRatio,
    minContactCount,
    minWitnessCount,
    minTransitionAgreement,
    maxNormalizedProjectionRmse: maxObservedNormalizedProjectionRmse,
    maxProjectionRmse,
    minSupportScore,
    edges: passingEdges,
    pass
  };
}
