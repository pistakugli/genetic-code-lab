# GPT-2 counterfactual circuit transfer

- Variant: `pilot`
- Access mode: `instrumented-witness`
- Model/device: `GPT2LMHeadModel` on `cuda`
- Frozen source SHA-256: `f24428b69b9b1fb2d750dfbf2ea55a065fc1d6cc740c415d01ac6ccb94e3715b`
- Eligible factual pairs: 744
- Tested pairs: 10
- Control transplants per direction: 100

## Pair eligibility

| Relation family | Facts | Eligible unordered pairs |
|---|---:|---:|
| country-capital | 39 | 741 |
| gendered-parent-term | 2 | 1 |
| occupation-description-to-profession | 2 | 1 |
| planet-size-extreme | 2 | 1 |

Excluded facts: 5; rejected candidate pairs: 0. Exclusion reasons are listed in `report.json`.

## Acceptance evidence

| Criterion | Measured result |
|---|---|
| A_counterfactualPromptAlignment | `{"allTestedPairsPreciselyAligned": true, "eligiblePairCount": 744, "excludedFactCount": 5, "rejectedCandidatePairCount": 0, "testedPairCount": 10}` |
| B_successfulActivationTransplantation | `{"directionCount": 20, "successfulDirections": 20}` |
| C_directionalAToBTransfer | `{"directionCount": 10, "positiveDonorContrastDirections": 10}` |
| D_reverseBToATransfer | `{"directionCount": 10, "positiveDonorContrastDirections": 5}` |
| E_necessity | `{"factCount": 45, "factsWithPositiveAnswerLikelihoodDrop": 44, "factsWithStrictCorrectGreedyNecessity": 3}` |
| F_counterfactualTransferSufficiency | `{"directionCount": 20, "directionsWithPositiveDonorAnswerLikelihoodChange": 18, "interpretation": "directional transfer evidence only; not a claim of a complete sufficient circuit"}` |
| G_recovery | `{"directionCount": 20, "verifiedDirections": 20}` |
| H_specificityAgainstStructuralControls | `{"directionCount": 20, "directionsPassingStrictSpecificity": 0, "directionsWithValidMagnitudeControls": 0}` |
| I_sharedVsFactSpecificComputation | `{"familiesWithCorrectedPositivePathTransferEvidence": 1, "familiesWithCorrectedPositiveSpecificityEvidence": 0, "interpretation": "generic-or-shared-predictive-influence-consistent; fact-specific-transfer-not-established", "relationFamilyAnalyses": 1}` |

## Paired statistical analysis

Bootstrap unit: source factual endpoints; pair directions are not treated as independent. Correction: Benjamini-Hochberg over control-contrast families `[]` and path-transfer families `['country-capital']`.

| Relation family | Facts | Transfer directions | Mean path-transfer contrast delta | 95% cluster CI | transfer q | Valid-control directions | Mean path−control delta | 95% cluster CI | control q |
|---|---:|---:|---:|---|---:|---:|---:|---|---:|
| country-capital | 11 | 20 | 2.528685784339905 | [1.8926326297862188, 3.2162614254014827] | 0.0006224712107065049 | 0 | None | None | None |

## Per-pair explanations

## France ↔ Germany (country-capital)

**Alignment:** exact fixed-prefix/suffix token alignment; the variable fact slot is never patched. **A→B:** France → Germany: donor-vs-recipient answer mean-log-likelihood contrast changed by 3.349174; donor answer likelihood changed by 2.450187; empirical control percentile 100.0, magnitude-valid controls 0/100; recipient recovery error 0. **B→A:** Germany → France: donor-vs-recipient answer mean-log-likelihood contrast changed by 3.916336; donor answer likelihood changed by 2.663086; empirical control percentile 100.0, magnitude-valid controls 0/100; recipient recovery error 0. Transfer is measured as the change in donor-answer versus recipient-answer mean token log-likelihood. A positive effect is evidence of directional information transfer, not proof that the transplanted path is a complete sufficient circuit. If activation-magnitude control coverage is inadequate, specificity remains unresolved.

**Original-context path removal:** France: answer-sequence log-likelihood drop 0.562548; strict correct-greedy necessity=False. Germany: answer-sequence log-likelihood drop 0.221127; strict correct-greedy necessity=False. These necessity measurements are separate from counterfactual transfer.

## France ↔ Italy (country-capital)

**Alignment:** exact fixed-prefix/suffix token alignment; the variable fact slot is never patched. **A→B:** France → Italy: donor-vs-recipient answer mean-log-likelihood contrast changed by 3.609749; donor answer likelihood changed by 2.433270; empirical control percentile 100.0, magnitude-valid controls 0/100; recipient recovery error 0. **B→A:** Italy → France: donor-vs-recipient answer mean-log-likelihood contrast changed by 4.310592; donor answer likelihood changed by 2.820288; empirical control percentile 100.0, magnitude-valid controls 0/100; recipient recovery error 0. Transfer is measured as the change in donor-answer versus recipient-answer mean token log-likelihood. A positive effect is evidence of directional information transfer, not proof that the transplanted path is a complete sufficient circuit. If activation-magnitude control coverage is inadequate, specificity remains unresolved.

**Original-context path removal:** France: answer-sequence log-likelihood drop 0.562548; strict correct-greedy necessity=False. Italy: answer-sequence log-likelihood drop 0.554810; strict correct-greedy necessity=False. These necessity measurements are separate from counterfactual transfer.

## France ↔ Spain (country-capital)

**Alignment:** exact fixed-prefix/suffix token alignment; the variable fact slot is never patched. **A→B:** France → Spain: donor-vs-recipient answer mean-log-likelihood contrast changed by 3.146790; donor answer likelihood changed by 2.414665; empirical control percentile 100.0, magnitude-valid controls 0/100; recipient recovery error 0. **B→A:** Spain → France: donor-vs-recipient answer mean-log-likelihood contrast changed by -0.036896; donor answer likelihood changed by 0.017370; empirical control percentile 3.0, magnitude-valid controls 0/100; recipient recovery error 0. Transfer is measured as the change in donor-answer versus recipient-answer mean token log-likelihood. A positive effect is evidence of directional information transfer, not proof that the transplanted path is a complete sufficient circuit. If activation-magnitude control coverage is inadequate, specificity remains unresolved.

**Original-context path removal:** France: answer-sequence log-likelihood drop 0.562548; strict correct-greedy necessity=False. Spain: answer-sequence log-likelihood drop 0.782004; strict correct-greedy necessity=True. These necessity measurements are separate from counterfactual transfer.

## France ↔ Portugal (country-capital)

**Alignment:** exact fixed-prefix/suffix token alignment; the variable fact slot is never patched. **A→B:** France → Portugal: donor-vs-recipient answer mean-log-likelihood contrast changed by 3.164642; donor answer likelihood changed by 2.176991; empirical control percentile 100.0, magnitude-valid controls 0/100; recipient recovery error 0. **B→A:** Portugal → France: donor-vs-recipient answer mean-log-likelihood contrast changed by -0.002091; donor answer likelihood changed by 0.010835; empirical control percentile 1.0, magnitude-valid controls 0/100; recipient recovery error 0. Transfer is measured as the change in donor-answer versus recipient-answer mean token log-likelihood. A positive effect is evidence of directional information transfer, not proof that the transplanted path is a complete sufficient circuit. If activation-magnitude control coverage is inadequate, specificity remains unresolved.

**Original-context path removal:** France: answer-sequence log-likelihood drop 0.562548; strict correct-greedy necessity=False. Portugal: answer-sequence log-likelihood drop 1.075219; strict correct-greedy necessity=False. These necessity measurements are separate from counterfactual transfer.

## France ↔ Greece (country-capital)

**Alignment:** exact fixed-prefix/suffix token alignment; the variable fact slot is never patched. **A→B:** France → Greece: donor-vs-recipient answer mean-log-likelihood contrast changed by 3.378586; donor answer likelihood changed by 2.777013; empirical control percentile 100.0, magnitude-valid controls 0/100; recipient recovery error 0. **B→A:** Greece → France: donor-vs-recipient answer mean-log-likelihood contrast changed by 0.008713; donor answer likelihood changed by 0.023945; empirical control percentile 0.0, magnitude-valid controls 0/100; recipient recovery error 0. Transfer is measured as the change in donor-answer versus recipient-answer mean token log-likelihood. A positive effect is evidence of directional information transfer, not proof that the transplanted path is a complete sufficient circuit. If activation-magnitude control coverage is inadequate, specificity remains unresolved.

**Original-context path removal:** France: answer-sequence log-likelihood drop 0.562548; strict correct-greedy necessity=False. Greece: answer-sequence log-likelihood drop 0.963473; strict correct-greedy necessity=True. These necessity measurements are separate from counterfactual transfer.

## France ↔ Egypt (country-capital)

**Alignment:** exact fixed-prefix/suffix token alignment; the variable fact slot is never patched. **A→B:** France → Egypt: donor-vs-recipient answer mean-log-likelihood contrast changed by 3.466073; donor answer likelihood changed by 3.697471; empirical control percentile 100.0, magnitude-valid controls 0/100; recipient recovery error 0. **B→A:** Egypt → France: donor-vs-recipient answer mean-log-likelihood contrast changed by -0.036934; donor answer likelihood changed by 0.051332; empirical control percentile 3.5, magnitude-valid controls 0/100; recipient recovery error 0. Transfer is measured as the change in donor-answer versus recipient-answer mean token log-likelihood. A positive effect is evidence of directional information transfer, not proof that the transplanted path is a complete sufficient circuit. If activation-magnitude control coverage is inadequate, specificity remains unresolved.

**Original-context path removal:** France: answer-sequence log-likelihood drop 0.562548; strict correct-greedy necessity=False. Egypt: answer-sequence log-likelihood drop 0.944108; strict correct-greedy necessity=False. These necessity measurements are separate from counterfactual transfer.

## France ↔ Japan (country-capital)

**Alignment:** exact fixed-prefix/suffix token alignment; the variable fact slot is never patched. **A→B:** France → Japan: donor-vs-recipient answer mean-log-likelihood contrast changed by 5.071602; donor answer likelihood changed by 4.139894; empirical control percentile 100.0, magnitude-valid controls 0/100; recipient recovery error 0. **B→A:** Japan → France: donor-vs-recipient answer mean-log-likelihood contrast changed by -0.002373; donor answer likelihood changed by 0.008545; empirical control percentile 0.0, magnitude-valid controls 0/100; recipient recovery error 0. Transfer is measured as the change in donor-answer versus recipient-answer mean token log-likelihood. A positive effect is evidence of directional information transfer, not proof that the transplanted path is a complete sufficient circuit. If activation-magnitude control coverage is inadequate, specificity remains unresolved.

**Original-context path removal:** France: answer-sequence log-likelihood drop 0.562548; strict correct-greedy necessity=False. Japan: answer-sequence log-likelihood drop 0.920667; strict correct-greedy necessity=False. These necessity measurements are separate from counterfactual transfer.

## France ↔ China (country-capital)

**Alignment:** exact fixed-prefix/suffix token alignment; the variable fact slot is never patched. **A→B:** France → China: donor-vs-recipient answer mean-log-likelihood contrast changed by 4.920097; donor answer likelihood changed by 4.298441; empirical control percentile 100.0, magnitude-valid controls 0/100; recipient recovery error 0. **B→A:** China → France: donor-vs-recipient answer mean-log-likelihood contrast changed by 0.007164; donor answer likelihood changed by -0.003545; empirical control percentile 1.0, magnitude-valid controls 0/100; recipient recovery error 0. Transfer is measured as the change in donor-answer versus recipient-answer mean token log-likelihood. A positive effect is evidence of directional information transfer, not proof that the transplanted path is a complete sufficient circuit. If activation-magnitude control coverage is inadequate, specificity remains unresolved.

**Original-context path removal:** France: answer-sequence log-likelihood drop 0.562548; strict correct-greedy necessity=False. China: answer-sequence log-likelihood drop 0.585075; strict correct-greedy necessity=False. These necessity measurements are separate from counterfactual transfer.

## France ↔ Russia (country-capital)

**Alignment:** exact fixed-prefix/suffix token alignment; the variable fact slot is never patched. **A→B:** France → Russia: donor-vs-recipient answer mean-log-likelihood contrast changed by 3.453911; donor answer likelihood changed by 3.453274; empirical control percentile 100.0, magnitude-valid controls 0/100; recipient recovery error 0. **B→A:** Russia → France: donor-vs-recipient answer mean-log-likelihood contrast changed by -0.137848; donor answer likelihood changed by -0.230044; empirical control percentile 52.0, magnitude-valid controls 11/100; recipient recovery error 0. Transfer is measured as the change in donor-answer versus recipient-answer mean token log-likelihood. A positive effect is evidence of directional information transfer, not proof that the transplanted path is a complete sufficient circuit. If activation-magnitude control coverage is inadequate, specificity remains unresolved.

**Original-context path removal:** France: answer-sequence log-likelihood drop 0.562548; strict correct-greedy necessity=False. Russia: answer-sequence log-likelihood drop 0.917113; strict correct-greedy necessity=False. These necessity measurements are separate from counterfactual transfer.

## France ↔ Canada (country-capital)

**Alignment:** exact fixed-prefix/suffix token alignment; the variable fact slot is never patched. **A→B:** France → Canada: donor-vs-recipient answer mean-log-likelihood contrast changed by 3.831696; donor answer likelihood changed by 2.849205; empirical control percentile 100.0, magnitude-valid controls 0/100; recipient recovery error 0. **B→A:** Canada → France: donor-vs-recipient answer mean-log-likelihood contrast changed by 5.154732; donor answer likelihood changed by 4.166144; empirical control percentile 100.0, magnitude-valid controls 0/100; recipient recovery error 0. Transfer is measured as the change in donor-answer versus recipient-answer mean token log-likelihood. A positive effect is evidence of directional information transfer, not proof that the transplanted path is a complete sufficient circuit. If activation-magnitude control coverage is inadequate, specificity remains unresolved.

**Original-context path removal:** France: answer-sequence log-likelihood drop 0.562548; strict correct-greedy necessity=False. Canada: answer-sequence log-likelihood drop 1.312102; strict correct-greedy necessity=False. These necessity measurements are separate from counterfactual transfer.

## Original factual states and necessity

| Fact | Answer | Greedy next token | Answer mean log-likelihood | Group ablation answer LL drop | Strict necessity |
|---|---|---|---:|---:|---|
| France | Paris |  the | -3.434415 | 0.562548 | False |
| Germany | Berlin |  the | -2.942636 | 0.221127 | False |
| Italy | Rome |  Rome | -1.849299 | 0.554810 | False |
| Spain | Madrid |  Madrid | -2.253682 | 0.782004 | True |
| Portugal | Lisbon |  the | -2.796748 | 1.075219 | False |
| Greece | Athens |  Athens | -1.726530 | 0.963473 | True |
| Egypt | Cairo |  the | -2.900995 | 0.944108 | False |
| Japan | Tokyo |  the | -2.699018 | 0.920667 | False |
| China | Beijing |  the | -4.090648 | 0.585075 | False |
| Russia | Moscow |  the | -3.652952 | 0.917113 | False |
| Canada | Ottawa |  the | -5.250278 | 1.312102 | False |
| Australia | Canberra |  the | -4.793439 | 0.747784 | False |
| Sweden | Stockholm |  the | -4.413470 | 1.079988 | False |
| Norway | Oslo |  the | -5.016771 | 0.444069 | False |
| Denmark | Copenhagen |  the | -3.845566 | 0.471722 | False |
| Finland | Helsinki |  the | -3.455549 | 0.283007 | False |
| Poland | Warsaw |  Warsaw | -2.271559 | 0.532275 | True |
| Hungary | Budapest |  the | -2.450612 | 0.963804 | False |
| Austria | Vienna |  the | -4.502680 | 1.129602 | False |
| Belgium | Brussels |  the | -4.246530 | 0.916746 | False |
| Netherlands | Amsterdam |  the | -3.095956 | 0.201600 | False |
| Ireland | Dublin |  the | -2.951297 | 0.402483 | False |
| Turkey | Ankara |  the | -4.145680 | 1.076075 | False |
| Iran | Tehran |  the | -3.261160 | 0.916864 | False |
| Iraq | Baghdad |  the | -3.827909 | 0.898439 | False |
| Syria | Damascus |  the | -4.603391 | -0.108896 | False |
| Israel | Jerusalem |  the | -3.720451 | 1.028895 | False |
| Saudi Arabia | Riyadh |  the | -4.046493 | 0.962008 | False |
| Pakistan | Islamabad |  the | -3.252666 | 0.091530 | False |
| Thailand | Bangkok |  the | -3.720597 | 0.866596 | False |
| Indonesia | Jakarta |  the | -4.138652 | 0.169962 | False |
| Philippines | Manila |  the | -3.010049 | 0.802108 | False |
| South Korea | Seoul |  the | -3.645093 | 1.052007 | False |
| Czech Republic | Prague |  the | -3.529799 | 1.118632 | False |
| Tunisia | Tunis |  the | -5.327062 | 0.703096 | False |
| Chile | Santiago |  the | -4.257068 | 0.986990 | False |
| Peru | Lima |  the | -6.427203 | 0.929176 | False |
| Cuba | Havana |  the | -4.708624 | 0.957783 | False |
| Jamaica | Kingston |  the | -6.018969 | 0.969536 | False |
| largest planet | Jupiter |  about | -3.505697 | 0.654620 | False |
| smallest planet | Mercury |  about | -5.648396 | 0.642257 | False |
| female parent | mother |  " | -3.389569 | 0.343021 | False |
| male parent | father |  " | -2.995496 | 0.361721 | False |
| medical profession | doctor |  person | -4.902764 | 0.859830 | False |
| school profession | teacher |  teacher | -2.161960 | 0.666335 | False |

## Computation interpretation and limitations

- Interpretation: `generic-or-shared-predictive-influence-consistent; fact-specific-transfer-not-established`
- Transfer: `{'evidence': 'path transfer changes donor-vs-recipient answer likelihood contrast relative to the recipient baseline and matched structural controls', 'positiveDirectionalTransfers': 15, 'directionCount': 20, 'strictSpecificityPassingDirections': 0, 'validMagnitudeAndStructuralControlDirections': 0, 'claim': 'not established unless a relation-family effect survives the cluster-aware CI and multiple-comparison procedure with valid magnitude controls'}`
- Shared relation: `{'evidenceAvailable': 'repeated prompt families and component/group interventions', 'interpretation': 'shared processing is not semantically identified by unit overlap or activation correlation alone', 'claim': 'not distinguished from generic prediction without selective transfer against valid controls'}`
- General influence: `{'factsWithPositiveAnswerLikelihoodDropUnderPathRemoval': 44, 'factsWithStrictCorrectGreedyPredictionNecessity': 3, 'factCount': 45, 'claim': 'ablation influence is separate from necessity and transfer sufficiency'}`
- Distributed effects: `{'evidence': 'per-component removal, group removal, and group-minus-sum-of-individual answer-log-likelihood effects', 'perFactInteractionValuesRecorded': 45, 'claim': 'interaction and redundancy are not equivalent; inspect raw component and group effects'}`

This hook-based local-weight experiment is not black-box decompilation. Transfer evidence is not complete circuit sufficiency. Invalid or amplitude-unmatched controls cannot establish specificity. Relation semantics are not assigned to individual neurons.

Raw activations, selected transfer logits, and control effect arrays are stored in the adjacent NPZ archive.
