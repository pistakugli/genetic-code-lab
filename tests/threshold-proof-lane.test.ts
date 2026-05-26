import { resolveThresholdProofLane } from '../src/solvers/threshold-proof-lane';

const ENV_KEYS = [
  'GCL_THRESHOLD_SELECTION_POLICY',
  'GCL_THRESHOLD_COMPOSE_EXPERIMENT',
  'GCL_THRESHOLD_ADAPTIVE_DUEL_EXPERIMENT',
  'GCL_THRESHOLD_UPPER_COMPOSE'
] as const;

function withThresholdEnv(
  env: Partial<Record<typeof ENV_KEYS[number], string>>,
  run: () => void
): void {
  const previous = Object.fromEntries(ENV_KEYS.map(key => [key, process.env[key]]));
  try {
    for (const key of ENV_KEYS) delete process.env[key];
    Object.assign(process.env, env);
    run();
  } finally {
    for (const key of ENV_KEYS) {
      const value = previous[key];
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
}

describe('resolveThresholdProofLane', () => {
  test('uses adaptive-duel threshold compose as the default production lane', () => {
    withThresholdEnv({}, () => {
      const lane = resolveThresholdProofLane();

      expect(lane.selectionPolicy).toBe('weighted');
      expect(lane.composeExperiment).toBe(true);
      expect(lane.composeRequested).toBe('adaptive');
      expect(lane.composeResolved).toBe('adaptive');
      expect(lane.upperCompose).toBe('adaptive');
      expect(lane.composeChooser).toBe('duel');
      expect(lane.composeCandidateSet).toEqual([
        'cross',
        'mirrored-cross',
        'triad-witness',
        'witnessed-or'
      ]);
      expect(lane.composeConditionalCandidateSet).toEqual(['guarded-upper-repair']);
      expect(lane.composeIncumbent).toBe('paired-witness');
      expect(lane.composeVetoPolicy).toBe('paired-witness-incumbent');
      expect(lane.composeExperimentClass).toBe('adaptive-duel');
      expect(lane.laneLabel).toBe('compose-adaptive-duel');
      expect(lane.proofClass).toBe('compose-experiment');
    });
  });

  test('keeps weighted lane honest when threshold compose is explicitly disabled', () => {
    withThresholdEnv({ GCL_THRESHOLD_COMPOSE_EXPERIMENT: '0' }, () => {
      const lane = resolveThresholdProofLane();

      expect(lane.selectionPolicy).toBe('weighted');
      expect(lane.composeExperiment).toBe(false);
      expect(lane.composeRequested).toBeNull();
      expect(lane.composeResolved).toBeNull();
      expect(lane.upperCompose).toBeNull();
      expect(lane.composeCandidateSet).toEqual([]);
      expect(lane.laneLabel).toBe('honest-weighted');
      expect(lane.proofClass).toBe('honest');
    });
  });

  test('resolves adaptive compose to paired-witness outside adaptive duel', () => {
    withThresholdEnv({
      GCL_THRESHOLD_COMPOSE_EXPERIMENT: '1',
      GCL_THRESHOLD_ADAPTIVE_DUEL_EXPERIMENT: '0'
    }, () => {
      const lane = resolveThresholdProofLane();

      expect(lane.composeRequested).toBe('adaptive');
      expect(lane.composeResolved).toBe('paired-witness');
      expect(lane.upperCompose).toBe('paired-witness');
      expect(lane.composeChooser).toBe('static');
      expect(lane.composeCandidateSet).toEqual(['paired-witness']);
      expect(lane.composeExperimentClass).toBe('adaptive-routing');
      expect(lane.laneLabel).toBe('compose-paired-witness');
      expect(lane.proofClass).toBe('compose-experiment');
    });
  });

  test('keeps paired-witness as the adaptive-duel incumbent', () => {
    withThresholdEnv({
      GCL_THRESHOLD_COMPOSE_EXPERIMENT: '1',
      GCL_THRESHOLD_ADAPTIVE_DUEL_EXPERIMENT: '1'
    }, () => {
      const lane = resolveThresholdProofLane();

      expect(lane.composeRequested).toBe('adaptive');
      expect(lane.composeResolved).toBe('adaptive');
      expect(lane.composeChooser).toBe('duel');
      expect(lane.composeCandidateSet).toEqual([
        'cross',
        'mirrored-cross',
        'triad-witness',
        'witnessed-or'
      ]);
      expect(lane.composeConditionalCandidateSet).toEqual(['guarded-upper-repair']);
      expect(lane.composeIncumbent).toBe('paired-witness');
      expect(lane.composeVetoPolicy).toBe('paired-witness-incumbent');
      expect(lane.composeExperimentClass).toBe('adaptive-duel');
      expect(lane.laneLabel).toBe('compose-adaptive-duel');
    });
  });

  test('labels selection plus compose as a compound experiment', () => {
    withThresholdEnv({
      GCL_THRESHOLD_SELECTION_POLICY: 'exact',
      GCL_THRESHOLD_COMPOSE_EXPERIMENT: '1',
      GCL_THRESHOLD_UPPER_COMPOSE: 'paired_witness'
    }, () => {
      const lane = resolveThresholdProofLane();

      expect(lane.selectionPolicy).toBe('exact');
      expect(lane.composeRequested).toBe('paired-witness');
      expect(lane.composeResolved).toBe('paired-witness');
      expect(lane.proofClass).toBe('compound-experiment');
      expect(lane.laneLabel).toBe('selection-exact+compose-paired-witness');
    });
  });
});
