# Output-only Black-box LLM Behavior Reader

- generatedAt: `2026-11-19T17:31:51.260212Z`
- model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- pass: `true`
- validCases: `9`
- passedCases: `9`
- passRate: `1.0`
- queryCount: `180`

## Access Audit

- solverAccess: `prompt-to-next-token-logits-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

## Aggregate Metrics

- medianTargetRank: `1.0`
- maxTargetRank: `1`
- medianSubjectSpecificityLift: `9.900390625`
- minSubjectSpecificityLift: `7.707812500000001`
- medianAnswerSpecificityMargin: `8.375`
- minAnswerSpecificityMargin: `7.4375`
- medianRelationTemplateMargin: `8.21875`
- minRelationTemplateMargin: `6.0625`
- medianParaphraseRetentionRate: `1.0`

- paraphraseProfile: `ood`

## Paraphrase family results

Answer-blind OOD paraphrase families. A family item is retained when the
target answer keeps a strictly positive answer-pool margin under the new
phrasing. Every family prompt of every valid case is listed; there is no
selection, so the table restates the whole recorded population.

| family | count | retained | retention rate | mean margin | min margin | median target rank | max target rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| context-prefixed-question | 9 | 9 | 1.000 | 7.555556 | 5.375000 | 2.0 | 2 |
| embedded-question | 9 | 9 | 1.000 | 8.550347 | 6.187500 | 3.0 | 3 |
| imperative | 9 | 9 | 1.000 | 9.487847 | 7.187500 | 2.0 | 3 |
| possessive-question | 9 | 9 | 1.000 | 7.718750 | 5.687500 | 2.0 | 2 |
| sentence-completion | 9 | 9 | 1.000 | 6.611111 | 5.250000 | 3.0 | 6 |
| service-question | 9 | 9 | 1.000 | 8.673611 | 5.937500 | 2.0 | 2 |

## Case Summary

| subject | answer | pass | rank | subjectLift | answerMargin | relationTemplateMargin | paraphraseRetention |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| France | Paris | true | 1 | 7.775000 | 8.187500 | 8.593750 | 1.000 |
| Germany | Berlin | true | 1 | 8.916406 | 8.531250 | 8.218750 | 1.000 |
| Italy | Rome | true | 1 | 10.146875 | 8.718750 | 8.851562 | 1.000 |
| Spain | Madrid | true | 1 | 7.707813 | 7.437500 | 7.875000 | 1.000 |
| Portugal | Lisbon | skip | - | - | - | - |
| Greece | Athens | skip | - | - | - | - |
| Egypt | Cairo | skip | - | - | - | - |
| Japan | Tokyo | true | 1 | 11.487891 | 9.500000 | 9.390625 | 1.000 |
| China | Beijing | true | 1 | 10.863672 | 8.187500 | 7.937500 | 1.000 |
| Russia | Moscow | true | 1 | 9.823828 | 8.375000 | 7.875000 | 1.000 |
| Canada | Ottawa | skip | - | - | - | - |
| Australia | Canberra | skip | - | - | - | - |
| Brazil | Brasilia | skip | - | - | - | - |
| Sweden | Stockholm | true | 1 | 12.610547 | 10.046875 | 9.171875 | 1.000 |
| Norway | Oslo | skip | - | - | - | - |
| Denmark | Copenhagen | skip | - | - | - | - |
| Finland | Helsinki | skip | - | - | - | - |
| Poland | Warsaw | skip | - | - | - | - |
| Austria | Vienna | true | 1 | 9.900391 | 8.187500 | 6.062500 | 1.000 |
| Belgium | Brussels | skip | - | - | - | - |

## Claim Boundary

This gate is output/logit-only. It builds a behavioral counterfactual graph
from prompt interventions and next-token logits. It does not claim original
neuron identity, hidden activation access, hook-based circuit recovery, or
full LLM decompilation.
