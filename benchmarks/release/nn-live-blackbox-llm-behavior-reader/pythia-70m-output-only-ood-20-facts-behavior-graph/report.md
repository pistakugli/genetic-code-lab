# Output-only Black-box LLM Behavior Reader

- generatedAt: `2026-11-19T16:54:16.240727Z`
- model: `EleutherAI/pythia-70m-deduped`
- pass: `false`
- validCases: `18`
- passedCases: `7`
- passRate: `0.3888888888888889`
- queryCount: `360`

## Access Audit

- solverAccess: `prompt-to-next-token-logits-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

## Aggregate Metrics

- medianTargetRank: `95.5`
- maxTargetRank: `730`
- medianSubjectSpecificityLift: `2.0`
- minSubjectSpecificityLift: `0.35000000000000053`
- medianAnswerSpecificityMargin: `0.5`
- minAnswerSpecificityMargin: `-1.0`
- medianRelationTemplateMargin: `0.0`
- minRelationTemplateMargin: `-1.5`
- medianParaphraseRetentionRate: `0.41666666666666663`

- paraphraseProfile: `ood`

## Paraphrase family results

Answer-blind OOD paraphrase families. A family item is retained when the
target answer keeps a strictly positive answer-pool margin under the new
phrasing. Every family prompt of every valid case is listed; there is no
selection, so the table restates the whole recorded population.

| family | count | retained | retention rate | mean margin | min margin | median target rank | max target rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| context-prefixed-question | 18 | 2 | 0.111 | -1.222222 | -3.000000 | 1466.0 | 6091 |
| embedded-question | 18 | 4 | 0.222 | -1.111111 | -3.000000 | 4518.0 | 15038 |
| imperative | 18 | 9 | 0.500 | 0.500000 | -2.000000 | 452.0 | 2443 |
| possessive-question | 18 | 17 | 0.944 | 1.777778 | -1.000000 | 21.0 | 86 |
| sentence-completion | 18 | 11 | 0.611 | 0.888889 | -1.000000 | 14.0 | 67 |
| service-question | 18 | 3 | 0.167 | -0.166667 | -3.000000 | 883.5 | 7542 |

## Case Summary

| subject | answer | pass | rank | subjectLift | answerMargin | relationTemplateMargin | paraphraseRetention |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| France | Paris | true | 30 | 2.000000 | 2.000000 | 2.000000 | 0.833 |
| Germany | Berlin | true | 28 | 1.350000 | 1.000000 | 0.500000 | 0.500 |
| Italy | Rome | false | 68 | 0.350000 | 0.000000 | 0.000000 | 0.333 |
| Spain | Madrid | true | 92 | 1.350000 | 1.000000 | 0.500000 | 0.500 |
| Portugal | Lisbon | false | 730 | 2.000000 | -1.000000 | -1.000000 | 0.167 |
| Greece | Athens | false | 213 | 2.000000 | 0.000000 | 0.000000 | 0.333 |
| Egypt | Cairo | false | 187 | 4.000000 | 1.000000 | 0.000000 | 0.500 |
| Japan | Tokyo | true | 25 | 3.000000 | 2.000000 | 2.500000 | 1.000 |
| China | Beijing | true | 68 | 3.000000 | 1.000000 | 1.500000 | 0.667 |
| Russia | Moscow | true | 56 | 3.000000 | 2.000000 | 1.000000 | 0.667 |
| Canada | Ottawa | false | 260 | 5.000000 | 1.000000 | 0.000000 | 0.167 |
| Australia | Canberra | skip | - | - | - | - |
| Brazil | Brasilia | skip | - | - | - | - |
| Sweden | Stockholm | true | 99 | 3.000000 | 1.000000 | 0.500000 | 0.500 |
| Norway | Oslo | false | 161 | 3.000000 | 0.000000 | -1.000000 | 0.167 |
| Denmark | Copenhagen | false | 400 | 3.000000 | 0.000000 | -1.000000 | 0.167 |
| Finland | Helsinki | false | 233 | 2.000000 | 0.000000 | 0.500000 | 0.500 |
| Poland | Warsaw | false | 343 | 2.000000 | -1.000000 | -1.000000 | 0.333 |
| Austria | Vienna | false | 66 | 2.000000 | 0.000000 | -1.500000 | 0.333 |
| Belgium | Brussels | false | 92 | 0.350000 | 0.000000 | -1.000000 | 0.000 |

## Claim Boundary

This gate is output/logit-only. It builds a behavioral counterfactual graph
from prompt interventions and next-token logits. It does not claim original
neuron identity, hidden activation access, hook-based circuit recovery, or
full LLM decompilation.
