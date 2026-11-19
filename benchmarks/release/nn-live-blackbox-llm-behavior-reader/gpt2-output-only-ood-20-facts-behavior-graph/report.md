# Output-only Black-box LLM Behavior Reader

- generatedAt: `2026-11-19T16:53:57.254238Z`
- model: `gpt2`
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

- medianTargetRank: `7.0`
- maxTargetRank: `23`
- medianSubjectSpecificityLift: `7.648689270019532`
- minSubjectSpecificityLift: `4.280092620849611`
- medianAnswerSpecificityMargin: `2.58935546875`
- minAnswerSpecificityMargin: `0.27651214599609375`
- medianRelationTemplateMargin: `2.3717803955078125`
- minRelationTemplateMargin: `0.67193603515625`
- medianParaphraseRetentionRate: `1.0`

- paraphraseProfile: `ood`

## Paraphrase family results

Answer-blind OOD paraphrase families. A family item is retained when the
target answer keeps a strictly positive answer-pool margin under the new
phrasing. Every family prompt of every valid case is listed; there is no
selection, so the table restates the whole recorded population.

| family | count | retained | retention rate | mean margin | min margin | median target rank | max target rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| context-prefixed-question | 19 | 19 | 1.000 | 2.511201 | 0.914978 | 12.0 | 49 |
| embedded-question | 19 | 19 | 1.000 | 2.080285 | 0.521103 | 16.0 | 38 |
| imperative | 19 | 19 | 1.000 | 1.717128 | 0.433388 | 108.0 | 201 |
| possessive-question | 19 | 19 | 1.000 | 2.738974 | 1.053741 | 7.0 | 35 |
| sentence-completion | 19 | 18 | 0.947 | 4.318451 | -0.468254 | 2.0 | 43 |
| service-question | 19 | 18 | 0.947 | 1.864439 | -0.395500 | 31.0 | 62 |

## Case Summary

| subject | answer | pass | rank | subjectLift | answerMargin | relationTemplateMargin | paraphraseRetention |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| France | Paris | true | 5 | 6.319559 | 2.456993 | 2.371780 | 1.000 |
| Germany | Berlin | true | 2 | 6.161688 | 2.502846 | 2.999924 | 1.000 |
| Italy | Rome | true | 1 | 6.404760 | 3.790810 | 2.931190 | 1.000 |
| Spain | Madrid | true | 1 | 5.259928 | 2.905441 | 2.649479 | 1.000 |
| Portugal | Lisbon | true | 2 | 6.377966 | 2.009872 | 1.380760 | 1.000 |
| Greece | Athens | true | 1 | 7.296106 | 4.728027 | 3.093796 | 1.000 |
| Egypt | Cairo | true | 2 | 9.193467 | 4.059410 | 4.047340 | 1.000 |
| Japan | Tokyo | true | 2 | 8.259701 | 4.423889 | 2.925587 | 1.000 |
| China | Beijing | true | 10 | 8.135340 | 3.207779 | 3.515724 | 1.000 |
| Russia | Moscow | true | 7 | 8.629742 | 3.831207 | 3.198143 | 1.000 |
| Canada | Ottawa | true | 23 | 9.565548 | 2.589355 | 2.375172 | 1.000 |
| Australia | Canberra | true | 17 | 10.570255 | 2.978752 | 2.106064 | 1.000 |
| Brazil | Brasilia | skip | - | - | - | - |
| Sweden | Stockholm | true | 13 | 7.698993 | 2.065872 | 2.128117 | 1.000 |
| Norway | Oslo | true | 19 | 7.648689 | 0.512939 | 2.294716 | 1.000 |
| Denmark | Copenhagen | true | 11 | 7.483702 | 1.873543 | 1.031181 | 1.000 |
| Finland | Helsinki | true | 7 | 9.261189 | 2.519188 | 2.121651 | 1.000 |
| Poland | Warsaw | true | 1 | 8.449841 | 4.636353 | 1.806934 | 1.000 |
| Austria | Vienna | true | 13 | 4.280093 | 0.276512 | 0.671936 | 0.667 |
| Belgium | Brussels | true | 11 | 4.463548 | 1.118301 | 1.227222 | 1.000 |

## Claim Boundary

This gate is output/logit-only. It builds a behavioral counterfactual graph
from prompt interventions and next-token logits. It does not claim original
neuron identity, hidden activation access, hook-based circuit recovery, or
full LLM decompilation.
