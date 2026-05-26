export type ThresholdSelectionPolicy = 'weighted' | 'hybrid' | 'exact';
export type ThresholdUpperCompose = 'adaptive' | 'coalition' | 'cross' | 'mirrored-cross' | 'triad-witness' | 'witnessed-or' | 'paired-witness';
export type ThresholdComposeChooser = 'none' | 'static' | 'duel';
export type ThresholdComposeExperimentClass = 'none' | 'single-candidate' | 'adaptive-routing' | 'adaptive-duel';

export type ThresholdProofLane = {
    selectionPolicy: ThresholdSelectionPolicy;
    selectionExperiment: boolean;
    composeExperiment: boolean;
    composeRequested: ThresholdUpperCompose | null;
    composeResolved: ThresholdUpperCompose | null;
    upperCompose: ThresholdUpperCompose | null;
    adaptiveDuelExperiment: boolean;
    composeChooser: ThresholdComposeChooser;
    composeCandidateSet: Array<'cross' | 'mirrored-cross' | 'triad-witness' | 'guarded-upper-repair' | 'witnessed-or' | 'paired-witness' | 'coalition'>;
    composeConditionalCandidateSet: Array<'guarded-upper-repair'>;
    composeIncumbent: 'paired-witness' | null;
    composeVetoPolicy: 'paired-witness-incumbent' | null;
    composeExperimentClass: ThresholdComposeExperimentClass;
    laneLabel: string;
    proofClass: 'honest' | 'selection-experiment' | 'compose-experiment' | 'compound-experiment';
};

function envFlagDefaultOn(name: string): boolean {
    const raw = (process.env[name] || '').trim().toLowerCase();
    return !(raw === '0' || raw === 'false' || raw === 'no' || raw === 'off');
}

export function resolveThresholdProofLane(): ThresholdProofLane {
    const rawSelectionPolicy = (process.env.GCL_THRESHOLD_SELECTION_POLICY || '').trim().toLowerCase();
    const selectionPolicy: ThresholdSelectionPolicy =
        rawSelectionPolicy === 'hybrid' ? 'hybrid'
        : rawSelectionPolicy === 'exact' ? 'exact'
        : 'weighted';
    const composeExperiment = envFlagDefaultOn('GCL_THRESHOLD_COMPOSE_EXPERIMENT');
    const adaptiveDuelExperiment = envFlagDefaultOn('GCL_THRESHOLD_ADAPTIVE_DUEL_EXPERIMENT');
    const rawUpperCompose = (process.env.GCL_THRESHOLD_UPPER_COMPOSE || '').trim().toLowerCase();
    const composeRequested =
        !composeExperiment ? null
        : rawUpperCompose === 'cross' ? 'cross'
        : rawUpperCompose === 'mirrored-cross' || rawUpperCompose === 'mirrored_cross' || rawUpperCompose === 'mirroredcross' ? 'mirrored-cross'
        : rawUpperCompose === 'triad-witness' || rawUpperCompose === 'triad_witness' || rawUpperCompose === 'triadwitness' ? 'triad-witness'
        : rawUpperCompose === 'coalition' ? 'coalition'
        : rawUpperCompose === 'paired-witness' || rawUpperCompose === 'paired_witness' || rawUpperCompose === 'pairedwitness' ? 'paired-witness'
        : rawUpperCompose === 'witnessed-or' || rawUpperCompose === 'witnessed_or' || rawUpperCompose === 'witnessedor' ? 'witnessed-or'
        : 'adaptive';
    const composeResolved =
        !composeExperiment ? null
        : composeRequested === 'adaptive'
            ? adaptiveDuelExperiment ? 'adaptive' : 'paired-witness'
            : composeRequested;
    const composeChooser: ThresholdComposeChooser =
        !composeExperiment ? 'none'
        : composeRequested === 'adaptive' && adaptiveDuelExperiment ? 'duel'
        : 'static';
    const composeCandidateSet: Array<'cross' | 'mirrored-cross' | 'triad-witness' | 'guarded-upper-repair' | 'witnessed-or' | 'paired-witness' | 'coalition'> =
        !composeExperiment ? []
        : composeRequested === 'adaptive'
            ? adaptiveDuelExperiment ? ['cross', 'mirrored-cross', 'triad-witness', 'witnessed-or'] : ['paired-witness']
            : composeRequested === 'coalition'
                ? ['coalition']
                : composeRequested === 'cross'
                    ? ['cross']
                    : composeRequested === 'mirrored-cross'
                        ? ['mirrored-cross']
                        : composeRequested === 'triad-witness'
                            ? ['triad-witness']
                            : composeRequested === 'paired-witness'
                                ? ['paired-witness']
                                : ['witnessed-or'];
    const composeConditionalCandidateSet: Array<'guarded-upper-repair'> =
        composeExperiment && composeRequested === 'adaptive' && adaptiveDuelExperiment
            ? ['guarded-upper-repair']
            : [];
    const composeIncumbent =
        composeExperiment && composeRequested === 'adaptive' && adaptiveDuelExperiment
            ? 'paired-witness' as const
            : null;
    const composeVetoPolicy =
        composeIncumbent === 'paired-witness'
            ? 'paired-witness-incumbent' as const
            : null;
    const composeExperimentClass: ThresholdComposeExperimentClass =
        !composeExperiment ? 'none'
        : composeRequested === 'adaptive'
            ? adaptiveDuelExperiment ? 'adaptive-duel' : 'adaptive-routing'
            : 'single-candidate';

    const selectionExperimental = selectionPolicy !== 'weighted';
    const proofClass =
        composeExperiment && selectionExperimental ? 'compound-experiment'
        : composeExperiment ? 'compose-experiment'
        : selectionExperimental ? 'selection-experiment'
        : 'honest';

    const composeLabel =
        !composeExperiment ? null
        : composeRequested === 'adaptive' && adaptiveDuelExperiment ? 'adaptive-duel'
        : composeResolved;
    const laneLabel =
        proofClass === 'honest' ? 'honest-weighted'
        : proofClass === 'selection-experiment' ? `selection-${selectionPolicy}`
        : proofClass === 'compose-experiment' ? `compose-${composeLabel}`
        : `selection-${selectionPolicy}+compose-${composeLabel}`;

    return {
        selectionPolicy,
        selectionExperiment: selectionExperimental,
        composeExperiment,
        composeRequested,
        composeResolved,
        upperCompose: composeResolved,
        adaptiveDuelExperiment,
        composeChooser,
        composeCandidateSet,
        composeConditionalCandidateSet,
        composeIncumbent,
        composeVetoPolicy,
        composeExperimentClass,
        laneLabel,
        proofClass
    };
}
