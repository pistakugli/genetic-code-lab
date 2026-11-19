# Output-only Black-box LLM Behavior Reader

- generatedAt: `2026-11-19T17:30:03.397471Z`
- model: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- pass: `true`
- validCases: `19`
- passedCases: `19`
- passRate: `1.0`
- queryCount: `380`

## Access Audit

- solverAccess: `prompt-to-next-token-logits-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

## Aggregate Metrics

- medianTargetRank: `1.0`
- maxTargetRank: `1`
- medianSubjectSpecificityLift: `17.0765625`
- minSubjectSpecificityLift: `12.584375`
- medianAnswerSpecificityMargin: `7.125`
- minAnswerSpecificityMargin: `5.4375`
- medianRelationTemplateMargin: `7.90625`
- minRelationTemplateMargin: `5.46875`
- medianParaphraseRetentionRate: `1.0`

- paraphraseProfile: `ood`

## Paraphrase family results

Answer-blind OOD paraphrase families. A family item is retained when the
target answer keeps a strictly positive answer-pool margin under the new
phrasing. Every family prompt of every valid case is listed; there is no
selection, so the table restates the whole recorded population.

| family | count | retained | retention rate | mean margin | min margin | median target rank | max target rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| context-prefixed-question | 19 | 19 | 1.000 | 6.889803 | 5.250000 | 1.0 | 2 |
| embedded-question | 19 | 19 | 1.000 | 9.208882 | 6.437500 | 1.0 | 3 |
| imperative | 19 | 19 | 1.000 | 7.728618 | 5.375000 | 1.0 | 3 |
| possessive-question | 19 | 19 | 1.000 | 7.868421 | 5.125000 | 1.0 | 1 |
| sentence-completion | 19 | 19 | 1.000 | 7.169408 | 5.562500 | 3.0 | 13 |
| service-question | 19 | 19 | 1.000 | 8.819079 | 6.125000 | 1.0 | 2 |

## Case Summary

| subject | answer | pass | rank | subjectLift | answerMargin | relationTemplateMargin | paraphraseRetention |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| France | Paris | true | 1 | 12.584375 | 5.437500 | 7.593750 | 1.000 |
| Germany | Berlin | true | 1 | 13.946875 | 7.875000 | 8.687500 | 1.000 |
| Italy | Rome | true | 1 | 14.962500 | 7.125000 | 8.812500 | 1.000 |
| Spain | Madrid | true | 1 | 16.398438 | 6.812500 | 7.218750 | 1.000 |
| Portugal | Lisbon | true | 1 | 17.203125 | 9.625000 | 9.500000 | 1.000 |
| Greece | Athens | true | 1 | 18.475000 | 9.750000 | 9.625000 | 1.000 |
| Egypt | Cairo | true | 1 | 19.985937 | 8.468750 | 8.218750 | 1.000 |
| Japan | Tokyo | true | 1 | 16.679688 | 7.812500 | 9.062500 | 1.000 |
| China | Beijing | true | 1 | 17.367188 | 8.375000 | 8.250000 | 1.000 |
| Russia | Moscow | true | 1 | 17.146875 | 8.562500 | 8.156250 | 1.000 |
| Canada | Ottawa | true | 1 | 18.973438 | 8.312500 | 8.000000 | 1.000 |
| Australia | Canberra | true | 1 | 17.057813 | 6.187500 | 7.171875 | 1.000 |
| Brazil | Brasilia | skip | - | - | - | - |
| Sweden | Stockholm | true | 1 | 17.076563 | 7.187500 | 7.437500 | 1.000 |
| Norway | Oslo | true | 1 | 17.212500 | 5.562500 | 5.468750 | 1.000 |
| Denmark | Copenhagen | true | 1 | 17.385938 | 7.062500 | 6.906250 | 1.000 |
| Finland | Helsinki | true | 1 | 17.695312 | 6.562500 | 7.140625 | 1.000 |
| Poland | Warsaw | true | 1 | 15.792188 | 5.500000 | 7.125000 | 1.000 |
| Austria | Vienna | true | 1 | 14.696875 | 5.687500 | 7.906250 | 1.000 |
| Belgium | Brussels | true | 1 | 13.793750 | 6.000000 | 7.781250 | 1.000 |

## Claim Boundary

This gate is output/logit-only. It builds a behavioral counterfactual graph
from prompt interventions and next-token logits. It does not claim original
neuron identity, hidden activation access, hook-based circuit recovery, or
full LLM decompilation.
