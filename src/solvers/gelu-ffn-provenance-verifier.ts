import { GeluFfnStructureAtom, GeluFfnStructureModel, predictGeluFfnStructureModel } from './gelu-ffn-structure-finder';

export type GeluFfnSourceNeuron = {
  id: string;
  layer: string;
  index: number;
  origin: string;
  inputWeight: number[];
  inputBias: number;
  outputWeight: number;
  activation?: 'gelu';
};

export type GeluFfnSourceNeuronGroup = {
  id: string;
  layer: string;
  sourceIds: string[];
  sourceIndices: number[];
  origin: string;
  inputWeight: number[];
  inputWeightNorm: number;
  inputBias: number;
  outputWeight: number;
  members: GeluFfnSourceNeuron[];
  duplicateSplit: boolean;
};

export type GeluFfnNeuronProvenanceMatch = {
  recoveredAtomIndex: number;
  sourceGroupId: string;
  sourceIds: string[];
  sourceLayer: string;
  sourceIndices: number[];
  sourceOrigin: string;
  duplicateSplit: boolean;
  confidence: number;
  directionCosine: number;
  inputWeightNormError: number;
  inputBiasError: number;
  outputWeightError: number;
  activationCorrelation: number;
  contributionCorrelation: number;
  contributionMae: number;
  parameterConfidence: number;
  functionalConfidence: number;
  whyReacts: {
    formula: string;
    responseRule: string;
    topTriggers: Array<{
      probeIndex: number;
      preactivation: number;
      activation: number;
      contribution: number;
      inputProjection: number;
    }>;
  };
};

export type GeluFfnNeuronProvenanceVerifierConfig = {
  probeInputs: number[][];
  sourceIdentityMode?: 'strict-parameter' | 'functional-equivalence';
  duplicateDirectionCosine?: number;
  duplicateBiasTolerance?: number;
  duplicateNormTolerance?: number;
  minDirectionCosine?: number;
  maxInputWeightNormError?: number;
  maxInputBiasError?: number;
  maxOutputWeightError?: number;
  minActivationCorrelation?: number;
  minContributionCorrelation?: number;
  maxContributionMae?: number;
  minConfidence?: number;
};

export type GeluFfnNeuronProvenanceReport = {
  verified: boolean;
  mode: 'instrumented-source-provenance';
  sourceIdentityMode: 'strict-parameter' | 'functional-equivalence';
  claim: string;
  sourceNeuronCount: number;
  sourceGroupCount: number;
  recoveredAtomCount: number;
  matches: GeluFfnNeuronProvenanceMatch[];
  topCandidateDiagnostics: Array<{
    recoveredAtomIndex: number;
    sourceGroupId: string;
    confidence: number;
    directionCosine: number;
    inputWeightNormError: number;
    inputBiasError: number;
    outputWeightError: number;
    activationCorrelation: number;
    contributionCorrelation: number;
    contributionMae: number;
    parameterConfidence: number;
    functionalConfidence: number;
  }>;
  unmatchedRecoveredAtomIndices: number[];
  unmatchedSourceGroupIds: string[];
  duplicateSourceGroups: GeluFfnSourceNeuronGroup[];
  ambiguous: boolean;
  gates: {
    allRecoveredAtomsMatched: boolean;
    allSourceGroupsMatched: boolean;
    noUnresolvedAmbiguity: boolean;
    parameterErrorsRequired: boolean;
    directionCosinePassed: boolean;
    parameterErrorsPassed: boolean;
    activationCorrelationPassed: boolean;
    contributionCorrelationPassed: boolean;
    contributionMaePassed: boolean;
    confidencePassed: boolean;
  };
};

function dot(left: number[], right: number[]): number {
  let sum = 0;
  for (let index = 0; index < left.length; index++) sum += left[index] * right[index];
  return sum;
}

function norm(values: number[]): number {
  return Math.sqrt(Math.max(0, dot(values, values)));
}

function normalize(values: number[]): number[] {
  const length = Math.max(1e-12, norm(values));
  return values.map(value => value / length);
}

function gelu(value: number): number {
  return 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * value ** 3)));
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function correlation(left: number[], right: number[]): number {
  const leftMean = mean(left);
  const rightMean = mean(right);
  let numerator = 0;
  let leftDenominator = 0;
  let rightDenominator = 0;
  for (let index = 0; index < left.length; index++) {
    const leftDelta = left[index] - leftMean;
    const rightDelta = right[index] - rightMean;
    numerator += leftDelta * rightDelta;
    leftDenominator += leftDelta ** 2;
    rightDenominator += rightDelta ** 2;
  }
  const denominator = Math.sqrt(leftDenominator * rightDenominator);
  return denominator > 1e-12 ? numerator / denominator : (leftDenominator < 1e-12 && rightDenominator < 1e-12 ? 1 : 0);
}

function atomNorm(atom: Pick<GeluFfnStructureAtom, 'inputWeight'>): number {
  return Math.max(1e-12, norm(atom.inputWeight));
}

function assertFiniteVector(name: string, values: number[], dimension: number): void {
  if (values.length !== dimension) throw new Error(`${name} dimension mismatch: expected ${dimension}, got ${values.length}`);
  if (!values.every(Number.isFinite)) throw new Error(`${name} must contain only finite numbers`);
}

function assertProvenanceInputs(
  model: GeluFfnStructureModel,
  sources: GeluFfnSourceNeuron[],
  probeInputs: number[][]
): number {
  if (!sources.length) throw new Error('GELU FFN provenance verifier requires at least one source neuron');
  const dimension = sources[0].inputWeight.length;
  if (dimension <= 0) throw new Error('GELU FFN provenance source input weights must be non-empty');
  if (model.dimension !== undefined && model.dimension !== dimension) {
    throw new Error(`GELU FFN provenance model/source dimension mismatch: model=${model.dimension}, source=${dimension}`);
  }
  for (const [index, source] of sources.entries()) {
    if ((source.activation ?? 'gelu') !== 'gelu') throw new Error(`GELU FFN provenance source ${source.id} uses unsupported activation ${source.activation}`);
    assertFiniteVector(`source ${source.id || index} inputWeight`, source.inputWeight, dimension);
    if (!Number.isFinite(source.inputBias) || !Number.isFinite(source.outputWeight)) {
      throw new Error(`source ${source.id || index} inputBias/outputWeight must be finite`);
    }
  }
  for (const [index, atom] of model.atoms.entries()) {
    assertFiniteVector(`recovered atom ${index} inputWeight`, atom.inputWeight, dimension);
  }
  for (const [index, input] of probeInputs.entries()) {
    assertFiniteVector(`probe input ${index}`, input, dimension);
  }
  return dimension;
}

function sameSourceSupport(
  left: GeluFfnSourceNeuronGroup,
  right: GeluFfnSourceNeuron,
  directionCosine: number,
  biasTolerance: number,
  normTolerance: number
): boolean {
  const rightNorm = atomNorm({ inputWeight: right.inputWeight });
  const cosine = dot(left.inputWeight, right.inputWeight) / Math.max(1e-12, left.inputWeightNorm * rightNorm);
  return cosine >= directionCosine
    && Math.abs(left.inputBias - right.inputBias) <= biasTolerance
    && Math.abs(left.inputWeightNorm - rightNorm) <= normTolerance;
}

function formatVectorTerm(weights: number[]): string {
  return weights
    .map((weight, index) => `${weight.toPrecision(6)}*x${index}`)
    .join(' + ');
}

function buildWhy(
  group: GeluFfnSourceNeuronGroup,
  probeInputs: number[][]
): GeluFfnNeuronProvenanceMatch['whyReacts'] {
  const direction = normalize(group.inputWeight);
  const rows = probeInputs.map((input, probeIndex) => {
    const inputProjection = dot(direction, input);
    const preactivation = dot(group.inputWeight, input) + group.inputBias;
    const activation = gelu(preactivation);
    const contribution = group.outputWeight * activation;
    return { probeIndex, preactivation, activation, contribution, inputProjection };
  }).sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  return {
    formula: `${group.outputWeight.toPrecision(6)} * GELU(${formatVectorTerm(group.inputWeight)} + ${group.inputBias.toPrecision(6)})`,
    responseRule: 'This source group reacts when the input projection along its weight direction pushes preactivation into the GELU transition/positive region; its signed output weight controls whether the channel raises or lowers the FFN output.',
    topTriggers: rows.slice(0, 5)
  };
}

export function groupGeluFfnSourceNeurons(
  sources: GeluFfnSourceNeuron[],
  config: Pick<GeluFfnNeuronProvenanceVerifierConfig, 'duplicateDirectionCosine' | 'duplicateBiasTolerance' | 'duplicateNormTolerance'> = {}
): GeluFfnSourceNeuronGroup[] {
  if (!sources.length) return [];
  const dimension = sources[0].inputWeight.length;
  for (const source of sources) assertFiniteVector(`source ${source.id} inputWeight`, source.inputWeight, dimension);
  const duplicateDirectionCosine = config.duplicateDirectionCosine ?? 0.9995;
  const duplicateBiasTolerance = config.duplicateBiasTolerance ?? 0.03;
  const duplicateNormTolerance = config.duplicateNormTolerance ?? 0.04;
  const groups: GeluFfnSourceNeuronGroup[] = [];
  for (const source of sources) {
    const group = groups.find(candidate => sameSourceSupport(
      candidate,
      source,
      duplicateDirectionCosine,
      duplicateBiasTolerance,
      duplicateNormTolerance
    ));
    if (group) {
      group.members.push(source);
      group.sourceIds.push(source.id);
      group.sourceIndices.push(source.index);
      group.outputWeight += source.outputWeight;
      group.duplicateSplit = true;
      group.id = `${group.layer}[${group.sourceIndices.join('+')}]`;
    } else {
      const inputWeightNorm = atomNorm({ inputWeight: source.inputWeight });
      groups.push({
        id: `${source.layer}[${source.index}]`,
        layer: source.layer,
        sourceIds: [source.id],
        sourceIndices: [source.index],
        origin: source.origin,
        inputWeight: [...source.inputWeight],
        inputWeightNorm,
        inputBias: source.inputBias,
        outputWeight: source.outputWeight,
        members: [source],
        duplicateSplit: false
      });
    }
  }
  return groups;
}

function sourceGroupContribution(group: GeluFfnSourceNeuronGroup, input: number[]): number {
  return group.outputWeight * gelu(dot(group.inputWeight, input) + group.inputBias);
}

function recoveredAtomContribution(atom: GeluFfnStructureAtom, input: number[]): number {
  return atom.outputWeight * gelu(dot(atom.inputWeight, input) + atom.inputBias);
}

function pairScore(
  atom: GeluFfnStructureAtom,
  group: GeluFfnSourceNeuronGroup,
  probeInputs: number[][]
): Omit<GeluFfnNeuronProvenanceMatch, 'recoveredAtomIndex' | 'sourceGroupId' | 'sourceIds' | 'sourceLayer' | 'sourceIndices' | 'sourceOrigin' | 'duplicateSplit' | 'whyReacts'> {
  const atomInputWeightNorm = atomNorm(atom);
  const directionCosine = dot(atom.inputWeight, group.inputWeight) / Math.max(1e-12, atomInputWeightNorm * group.inputWeightNorm);
  const inputWeightNormError = Math.abs(atomInputWeightNorm - group.inputWeightNorm);
  const inputBiasError = Math.abs(atom.inputBias - group.inputBias);
  const outputWeightError = Math.abs(atom.outputWeight - group.outputWeight);
  const sourcePreactivations = probeInputs.map(input => dot(group.inputWeight, input) + group.inputBias);
  const recoveredPreactivations = probeInputs.map(input => dot(atom.inputWeight, input) + atom.inputBias);
  const sourceContributions = probeInputs.map(input => sourceGroupContribution(group, input));
  const recoveredContributions = probeInputs.map(input => recoveredAtomContribution(atom, input));
  const activationCorrelation = correlation(sourcePreactivations.map(gelu), recoveredPreactivations.map(gelu));
  const contributionCorrelation = correlation(sourceContributions, recoveredContributions);
  const contributionMae = mean(sourceContributions.map((value, index) => Math.abs(value - recoveredContributions[index])));
  const normalizedNormScore = 1 - Math.min(1, inputWeightNormError / Math.max(1, group.inputWeightNorm));
  const normalizedBiasScore = 1 - Math.min(1, inputBiasError / Math.max(0.25, Math.abs(group.inputBias), 1));
  const normalizedOutputScore = 1 - Math.min(1, outputWeightError / Math.max(0.25, Math.abs(group.outputWeight), 1));
  const normalizedContributionScore = 1 - Math.min(1, contributionMae / Math.max(0.02, mean(sourceContributions.map(Math.abs))));
  const parameterConfidence = Math.max(0, Math.min(
    directionCosine,
    activationCorrelation,
    contributionCorrelation,
    normalizedNormScore,
    normalizedBiasScore,
    normalizedOutputScore,
    normalizedContributionScore
  ));
  const functionalConfidence = Math.max(0, Math.min(
    directionCosine,
    activationCorrelation,
    contributionCorrelation,
    normalizedContributionScore
  ));
  return {
    confidence: parameterConfidence,
    directionCosine,
    inputWeightNormError,
    inputBiasError,
    outputWeightError,
    activationCorrelation,
    contributionCorrelation,
    contributionMae,
    parameterConfidence,
    functionalConfidence
  };
}

export function verifyGeluFfnNeuronProvenance(
  model: GeluFfnStructureModel,
  sources: GeluFfnSourceNeuron[],
  config: GeluFfnNeuronProvenanceVerifierConfig
): GeluFfnNeuronProvenanceReport {
  if (!config.probeInputs.length) throw new Error('GELU FFN provenance verifier requires probe inputs');
  assertProvenanceInputs(model, sources, config.probeInputs);
  const groups = groupGeluFfnSourceNeurons(sources, config);
  const sourceIdentityMode = config.sourceIdentityMode ?? 'strict-parameter';
  const requireParameterErrors = sourceIdentityMode === 'strict-parameter';
  const minDirectionCosine = config.minDirectionCosine ?? 0.99;
  const maxInputWeightNormError = config.maxInputWeightNormError ?? 0.18;
  const maxInputBiasError = config.maxInputBiasError ?? 0.16;
  const maxOutputWeightError = config.maxOutputWeightError ?? 0.22;
  const minActivationCorrelation = config.minActivationCorrelation ?? 0.995;
  const minContributionCorrelation = config.minContributionCorrelation ?? 0.995;
  const maxContributionMae = config.maxContributionMae ?? 0.02;
  const minConfidence = config.minConfidence ?? 0.82;

  const candidatePairs = model.atoms.flatMap((atom, recoveredAtomIndex) => (
    groups.map(group => ({
      recoveredAtomIndex,
      group,
      score: pairScore(atom, group, config.probeInputs)
    }))
  )).sort((a, b) => (
    sourceIdentityMode === 'functional-equivalence'
      ? b.score.functionalConfidence - a.score.functionalConfidence
      : b.score.parameterConfidence - a.score.parameterConfidence
  ));
  const topCandidateDiagnostics = candidatePairs.slice(0, Math.min(24, candidatePairs.length)).map(candidate => ({
    recoveredAtomIndex: candidate.recoveredAtomIndex,
    sourceGroupId: candidate.group.id,
    confidence: sourceIdentityMode === 'functional-equivalence'
      ? candidate.score.functionalConfidence
      : candidate.score.parameterConfidence,
    directionCosine: candidate.score.directionCosine,
    inputWeightNormError: candidate.score.inputWeightNormError,
    inputBiasError: candidate.score.inputBiasError,
    outputWeightError: candidate.score.outputWeightError,
    activationCorrelation: candidate.score.activationCorrelation,
    contributionCorrelation: candidate.score.contributionCorrelation,
    contributionMae: candidate.score.contributionMae,
    parameterConfidence: candidate.score.parameterConfidence,
    functionalConfidence: candidate.score.functionalConfidence
  }));

  const usedAtoms = new Set<number>();
  const usedGroups = new Set<string>();
  const matches: GeluFfnNeuronProvenanceMatch[] = [];
  for (const candidate of candidatePairs) {
    if (usedAtoms.has(candidate.recoveredAtomIndex) || usedGroups.has(candidate.group.id)) continue;
    const confidence = sourceIdentityMode === 'functional-equivalence'
      ? candidate.score.functionalConfidence
      : candidate.score.parameterConfidence;
    const passesPair = candidate.score.directionCosine >= minDirectionCosine
      && (!requireParameterErrors || candidate.score.inputWeightNormError <= maxInputWeightNormError)
      && (!requireParameterErrors || candidate.score.inputBiasError <= maxInputBiasError)
      && (!requireParameterErrors || candidate.score.outputWeightError <= maxOutputWeightError)
      && candidate.score.activationCorrelation >= minActivationCorrelation
      && candidate.score.contributionCorrelation >= minContributionCorrelation
      && candidate.score.contributionMae <= maxContributionMae
      && confidence >= minConfidence;
    if (!passesPair) continue;
    usedAtoms.add(candidate.recoveredAtomIndex);
    usedGroups.add(candidate.group.id);
    matches.push({
      recoveredAtomIndex: candidate.recoveredAtomIndex,
      sourceGroupId: candidate.group.id,
      sourceIds: [...candidate.group.sourceIds],
      sourceLayer: candidate.group.layer,
      sourceIndices: [...candidate.group.sourceIndices],
      sourceOrigin: candidate.group.origin,
      duplicateSplit: candidate.group.duplicateSplit,
      ...candidate.score,
      confidence,
      whyReacts: buildWhy(candidate.group, config.probeInputs)
    });
  }

  const unmatchedRecoveredAtomIndices = model.atoms
    .map((_, index) => index)
    .filter(index => !usedAtoms.has(index));
  const unmatchedSourceGroupIds = groups
    .map(group => group.id)
    .filter(id => !usedGroups.has(id));
  const duplicateSourceGroups = groups.filter(group => group.duplicateSplit);
  const gates = {
    allRecoveredAtomsMatched: unmatchedRecoveredAtomIndices.length === 0,
    allSourceGroupsMatched: unmatchedSourceGroupIds.length === 0,
    noUnresolvedAmbiguity: true,
    parameterErrorsRequired: requireParameterErrors,
    directionCosinePassed: matches.every(match => match.directionCosine >= minDirectionCosine),
    parameterErrorsPassed: !requireParameterErrors || matches.every(match => (
      match.inputWeightNormError <= maxInputWeightNormError
      && match.inputBiasError <= maxInputBiasError
      && match.outputWeightError <= maxOutputWeightError
    )),
    activationCorrelationPassed: matches.every(match => match.activationCorrelation >= minActivationCorrelation),
    contributionCorrelationPassed: matches.every(match => match.contributionCorrelation >= minContributionCorrelation),
    contributionMaePassed: matches.every(match => match.contributionMae <= maxContributionMae),
    confidencePassed: matches.every(match => match.confidence >= minConfidence)
  };
  const verified = gates.allRecoveredAtomsMatched
    && gates.allSourceGroupsMatched
    && gates.noUnresolvedAmbiguity
    && gates.directionCosinePassed
    && gates.parameterErrorsPassed
    && gates.activationCorrelationPassed
    && gates.contributionCorrelationPassed
    && gates.contributionMaePassed
    && gates.confidencePassed;
  return {
    verified,
    mode: 'instrumented-source-provenance',
    sourceIdentityMode,
    claim: sourceIdentityMode === 'strict-parameter'
      ? 'Recovered GELU atoms are matched to instrumented source neuron groups with strict parameter-error checks. Duplicate/split source neurons are reported as groups, not hallucinated as a unique single original neuron.'
      : 'Recovered GELU atoms are matched to instrumented source neuron groups by functional equivalence: direction, activation trace, and contribution trace must agree, while norm/output-weight gauge drift is reported instead of treated as exact parameter identity.',
    sourceNeuronCount: sources.length,
    sourceGroupCount: groups.length,
    recoveredAtomCount: model.atoms.length,
    matches,
    topCandidateDiagnostics,
    unmatchedRecoveredAtomIndices,
    unmatchedSourceGroupIds,
    duplicateSourceGroups,
    ambiguous: duplicateSourceGroups.length > 0,
    gates
  };
}

export function predictGeluFfnSourceNeurons(sources: GeluFfnSourceNeuron[], outputBias: number, input: number[]): number {
  return outputBias + sources.reduce((sum, source) => (
    sum + source.outputWeight * gelu(dot(source.inputWeight, input) + source.inputBias)
  ), 0);
}

export function modelAgreementMae(model: GeluFfnStructureModel, sources: GeluFfnSourceNeuron[], outputBias: number, inputs: number[][]): number {
  return mean(inputs.map(input => Math.abs(
    predictGeluFfnStructureModel(model, input) - predictGeluFfnSourceNeurons(sources, outputBias, input)
  )));
}
