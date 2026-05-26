import fs from 'fs';
import path from 'path';
import vm from 'vm';
import ts from 'typescript';

type Helpers = Record<string, (...args: any[]) => any>;

function loadCalibrationHelpers(): Helpers {
  const scriptPath = path.join(__dirname, '..', 'scripts', 'compose-routing-calibration.ts');
  const source = fs.readFileSync(scriptPath, 'utf8').replace(/^#!.*\n/, '');
  const mainInvocation = source.indexOf('\ntry {\n    main();');

  expect(mainInvocation).toBeGreaterThan(0);

  const helperSource = `
${source.slice(0, mainInvocation)}

;(globalThis as any).__helpers = {
  parseNumber,
  parseNumberList,
  parseModeList,
  parseStrategyList,
  parseSelectionPolicy,
  mean,
  median,
  parsePayload,
  flattenRows,
  sumComposeDebug,
  pickWinners,
  buildWinnerRecords,
  buildAggregateSummary
};
`;

  const compiled = ts.transpileModule(helperSource, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
      esModuleInterop: true
    }
  }).outputText;

  const sandbox: any = {
    __dirname: path.join(__dirname, '..', 'scripts'),
    console,
    globalThis: {},
    process: {
      env: {},
      argv: [],
      stdout: { write: jest.fn() },
      stderr: { write: jest.fn() }
    },
    require: (request: string) => {
      if (request === 'path') return require('path');
      if (request === 'child_process') return { spawnSync: jest.fn() };
      throw new Error(`Unexpected require from calibration helper test: ${request}`);
    },
    exports: {},
    module: { exports: {} }
  };
  sandbox.globalThis = sandbox;

  vm.runInNewContext(compiled, sandbox, { filename: scriptPath });
  return sandbox.__helpers;
}

function row(overrides: Record<string, any>): any {
  return {
    seed: 11,
    mode: 'bridge',
    proofLane: {
      selectionPolicy: 'exact',
      selectionExperiment: true,
      composeExperiment: true,
      composeRequested: 'adaptive',
      composeResolved: 'paired-witness',
      upperCompose: 'paired-witness',
      adaptiveDuelExperiment: true,
      composeChooser: 'duel',
      composeCandidateSet: ['cross', 'mirrored-cross', 'triad-witness', 'witnessed-or'],
      composeConditionalCandidateSet: ['guarded-upper-repair'],
      composeIncumbent: 'paired-witness',
      composeVetoPolicy: 'paired-witness-incumbent',
      composeExperimentClass: 'adaptive-duel',
      laneLabel: 'compose-adaptive-duel',
      proofClass: 'compound-experiment'
    },
    parallelEvaluators: 8,
    distributedIslands: 1,
    topology: 'stable',
    fitness: 0.91,
    exactAccuracy: 0.88,
    length: 123,
    lowerBoundary: 0.75,
    upperBoundary: 0.5,
    boundary: { total: 4, fp4: 3, fn5: 1 },
    composeDebug: {
      thresholdComposeFocusCount: 10,
      thresholdRoleComposeCount: 7,
      thresholdUpperComposeOpportunityCount: 10,
      thresholdAdaptiveDuelCount: 5,
      thresholdAdaptiveCrossCount: 2,
      thresholdAdaptiveMirroredCrossCount: 0,
      thresholdAdaptiveTriadWitnessCount: 0,
      thresholdAdaptivePairedWitnessCount: 3,
      thresholdAdaptiveGuardedCrossCount: 0,
      thresholdAdaptiveGuardedUpperRepairCount: 1,
      thresholdAdaptiveWitnessedOrCount: 4,
      thresholdAdaptivePairedWitnessIncumbentCount: 3,
      thresholdAdaptivePairedWitnessVetoCount: 2,
      thresholdAdaptiveCrossVetoCount: 1,
      thresholdAdaptiveMirroredCrossVetoCount: 0,
      thresholdAdaptiveTriadWitnessVetoCount: 0,
      thresholdAdaptiveGuardedCrossVetoCount: 0,
      thresholdAdaptiveGuardedUpperRepairVetoCount: 0,
      thresholdAdaptiveWitnessedOrVetoCount: 1,
      thresholdAdaptiveVetoExactRegressionCount: 0,
      thresholdAdaptiveVetoUpperExactRegressionCount: 0,
      thresholdAdaptiveVetoLowerExactRegressionCount: 0,
      thresholdAdaptiveVetoUpperBoundaryRegressionCount: 0,
      thresholdAdaptiveVetoLowerBoundaryRegressionCount: 1,
      thresholdAdaptiveVetoBoundaryRegressionCount: 0,
      thresholdAdaptiveVetoPressureRegressionCount: 0,
      thresholdAdaptiveVetoWeightedRegressionCount: 0,
      thresholdAdaptiveVetoNoMaterialGainCount: 1,
      thresholdStaticPairedWitnessComposeCount: 0
    },
    ...overrides
  };
}

describe('compose routing calibration helpers', () => {
  const helpers = loadCalibrationHelpers();

  test('normalizes CLI list inputs without invoking calibration runtime', () => {
    expect(helpers.parseNumber('42', 9)).toBe(42);
    expect(helpers.parseNumber('nope', 9)).toBe(9);
    expect(helpers.parseNumberList('10, bad, 30', [1])).toEqual([10, 30]);
    expect(helpers.parseNumberList('', [1, 2])).toEqual([1, 2]);
    expect(helpers.parseModeList('bridge, agent, unknown', ['none'])).toEqual(['bridge', 'agent']);
    expect(helpers.parseModeList('unknown', ['bridge'])).toEqual(['bridge']);
    expect(helpers.parseStrategyList('adaptive_duel,paired_witness,witnessedor,cross,mirroredcross,triadwitness,garbage', ['adaptive'])).toEqual([
      'adaptive-duel',
      'paired-witness',
      'witnessed-or',
      'cross',
      'mirrored-cross',
      'triad-witness'
    ]);
    expect(helpers.parseSelectionPolicy('EXACT')).toBe('exact');
    expect(helpers.parseSelectionPolicy('invalid')).toBe('weighted');
  });

  test('parses JSON payloads even when benchmark output has log noise', () => {
    expect(helpers.parsePayload('noise before {"summary":{"bridge":{"rows":[]}}} noise after')).toEqual({
      summary: { bridge: { rows: [] } }
    });

    expect(() => helpers.parsePayload('no json here')).toThrow(
      'benchmark-majority-seeds output did not contain parseable JSON'
    );
  });

  test('flattens benchmark rows with weak-side and compose-share calibration fields', () => {
    const flattened = helpers.flattenRows(
      {
        summary: {
          bridge: {
            meanFitness: 0,
            medianFitness: 0,
            meanLowerBoundary: 0,
            meanUpperBoundary: 0,
            rows: [row({})]
          }
        }
      },
      60,
      'exact',
      'adaptive'
    );

    expect(flattened).toHaveLength(1);
    expect(flattened[0]).toMatchObject({
      generations: 60,
      strategyRequested: 'adaptive',
      strategyResolved: 'paired-witness',
      selectionRequested: 'exact',
      selectionResolved: 'exact',
      weakSide: 'upper',
      boundaryGap: 0.25,
      upperMinusLower: -0.25,
      composeShare: {
        cross: 0.2,
        mirroredCross: 0,
        triadWitness: 0,
        pairedWitness: 0.3,
        guardedCross: 0,
        guardedUpperRepair: 0.1,
        witnessedOr: 0.4
      },
      composeDecisionRate: 0.5,
      staticPairedWitnessComposeRate: 0,
      pairedWitnessIncumbentRate: 0.3,
      pairedWitnessVetoRate: 0.2
    });
  });

  test('keeps static paired-witness routing out of adaptive-duel counts', () => {
    const flattened = helpers.flattenRows(
      {
        summary: {
          bridge: {
            rows: [
              row({
                proofLane: {
                  ...row({}).proofLane,
                  composeRequested: 'paired-witness',
                  composeResolved: 'paired-witness',
                  upperCompose: 'paired-witness',
                  adaptiveDuelExperiment: false,
                  composeChooser: 'static',
                  composeCandidateSet: ['paired-witness'],
                  composeConditionalCandidateSet: [],
                  composeIncumbent: null,
                  composeVetoPolicy: null,
                  composeExperimentClass: 'single-candidate',
                  laneLabel: 'compose-paired-witness',
                  proofClass: 'compose-experiment'
                },
                composeDebug: {
                  thresholdComposeFocusCount: 10,
                  thresholdRoleComposeCount: 0,
                  thresholdUpperComposeOpportunityCount: 10,
                  thresholdAdaptiveDuelCount: 0,
                  thresholdAdaptiveCrossCount: 0,
                  thresholdAdaptiveMirroredCrossCount: 0,
                  thresholdAdaptiveTriadWitnessCount: 0,
                  thresholdAdaptivePairedWitnessCount: 0,
                  thresholdAdaptiveGuardedCrossCount: 0,
                  thresholdAdaptiveGuardedUpperRepairCount: 0,
                  thresholdAdaptiveWitnessedOrCount: 0,
                  thresholdAdaptiveCrossVetoCount: 0,
                  thresholdAdaptiveMirroredCrossVetoCount: 0,
                  thresholdAdaptiveTriadWitnessVetoCount: 0,
                  thresholdAdaptiveGuardedCrossVetoCount: 0,
                  thresholdAdaptiveGuardedUpperRepairVetoCount: 0,
                  thresholdAdaptiveWitnessedOrVetoCount: 0,
                  thresholdStaticPairedWitnessComposeCount: 10
                }
              })
            ]
          }
        }
      },
      20,
      'weighted',
      'paired-witness'
    );

    expect(flattened).toHaveLength(1);
    expect(flattened[0]).toMatchObject({
      strategyRequested: 'paired-witness',
      strategyResolved: 'paired-witness',
      composeDecisionRate: 0,
      staticPairedWitnessComposeRate: 1,
      composeShare: {
        cross: 0,
        mirroredCross: 0,
        triadWitness: 0,
        pairedWitness: 0,
        guardedCross: 0,
        guardedUpperRepair: 0,
        witnessedOr: 0
      }
    });

    const aggregate = helpers.buildAggregateSummary(flattened, helpers.buildWinnerRecords(flattened));
    expect(aggregate['paired-witness']).toMatchObject({
      meanComposeDecisionRate: 0,
      meanStaticPairedWitnessComposeRate: 1,
      composeDebugTotals: expect.objectContaining({
        thresholdAdaptiveDuelCount: 0,
        thresholdAdaptivePairedWitnessCount: 0,
        thresholdStaticPairedWitnessComposeCount: 10
      })
    });
  });

  test('builds deterministic winners and aggregate summaries across strategies', () => {
    const rows = [
      ...helpers.flattenRows({ summary: { bridge: { rows: [row({})] } } }, 20, 'exact', 'paired-witness'),
      ...helpers.flattenRows({
        summary: {
          bridge: {
            rows: [
              row({
                proofLane: { ...row({}).proofLane, composeResolved: 'cross' },
                fitness: 0.93,
                exactAccuracy: 0.9,
                lowerBoundary: 0.8,
                upperBoundary: 0.75,
                boundary: { total: 2, fp4: 1, fn5: 1 },
                composeDebug: {
                  thresholdComposeFocusCount: 4,
                  thresholdRoleComposeCount: 4,
                  thresholdUpperComposeOpportunityCount: 4,
                  thresholdAdaptiveDuelCount: 2,
                  thresholdAdaptiveCrossCount: 4,
                  thresholdAdaptiveMirroredCrossCount: 0,
                  thresholdAdaptiveTriadWitnessCount: 0,
                  thresholdAdaptivePairedWitnessCount: 0,
                  thresholdAdaptiveGuardedCrossCount: 0,
                  thresholdAdaptiveGuardedUpperRepairCount: 0,
                  thresholdAdaptiveWitnessedOrCount: 0,
                  thresholdAdaptivePairedWitnessIncumbentCount: 0,
                  thresholdAdaptivePairedWitnessVetoCount: 0,
                  thresholdAdaptiveCrossVetoCount: 0,
                  thresholdAdaptiveMirroredCrossVetoCount: 0,
                  thresholdAdaptiveTriadWitnessVetoCount: 0,
                  thresholdAdaptiveGuardedCrossVetoCount: 0,
                  thresholdAdaptiveGuardedUpperRepairVetoCount: 0,
                  thresholdAdaptiveWitnessedOrVetoCount: 0,
                  thresholdStaticPairedWitnessComposeCount: 0
                }
              })
            ]
          }
        }
      }, 20, 'exact', 'cross')
    ];

    const winners = helpers.buildWinnerRecords(rows);
    expect(winners).toEqual([
      expect.objectContaining({
        generations: 20,
        mode: 'bridge',
        seed: 11,
        exactWinners: ['cross'],
        boundaryWinners: ['cross'],
        fitnessWinners: ['cross'],
        fp4Winners: ['cross'],
        fn5Winners: ['cross', 'paired-witness'],
        balanceWinners: ['cross'],
        rows: [
          expect.objectContaining({ strategyRequested: 'cross', boundaryTotal: 2 }),
          expect.objectContaining({ strategyRequested: 'paired-witness', boundaryTotal: 4 })
        ]
      })
    ]);

    const aggregate = helpers.buildAggregateSummary(rows, winners);
    expect(aggregate.cross).toMatchObject({
      runs: 1,
      meanFitness: 0.93,
      meanBoundaryTotal: 2,
      meanBoundaryGap: 0.05,
      meanAdaptiveCrossShare: 1,
      meanComposeDecisionRate: 0.5,
      meanStaticPairedWitnessComposeRate: 0,
      meanAdaptivePairedWitnessIncumbentRate: 0,
      meanAdaptivePairedWitnessVetoRate: 0,
      exactWins: 1,
      boundaryWins: 1,
      fitnessWins: 1
    });
    expect(aggregate['paired-witness']).toMatchObject({
      runs: 1,
      meanAdaptivePairedWitnessShare: 0.3,
      composeDebugTotals: expect.objectContaining({
        thresholdComposeFocusCount: 10,
        thresholdUpperComposeOpportunityCount: 10,
        thresholdAdaptivePairedWitnessCount: 3,
        thresholdAdaptivePairedWitnessIncumbentCount: 3,
        thresholdAdaptivePairedWitnessVetoCount: 2,
        thresholdAdaptiveCrossVetoCount: 1,
        thresholdAdaptiveWitnessedOrVetoCount: 1,
        thresholdAdaptiveVetoLowerBoundaryRegressionCount: 1,
        thresholdAdaptiveVetoNoMaterialGainCount: 1
      }),
      exactWins: 0,
      boundaryWins: 0,
      fitnessWins: 0
    });
  });
});
