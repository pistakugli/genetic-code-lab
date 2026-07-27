# Output-only Black-box LLM Behavior Reader

- generatedAt: `2026-07-09T19:23:04.705211Z`
- model: `gpt2`
- pass: `true`
- validCases: `10`
- passedCases: `10`
- passRate: `1.0`
- queryCount: `240`

## Access Audit

- solverAccess: `prompt-to-next-token-logits-only`
- weightsUsedBySolver: `false`
- hooksUsedBySolver: `false`
- postHocWitnessOnly: `true`

## Aggregate Metrics

- medianTargetRank: `2.0`
- maxTargetRank: `10`
- medianSubjectSpecificityLift: `6.850424766540527`
- minSubjectSpecificityLift: `5.2599224090576175`
- medianAnswerSpecificityMargin: `3.499279022216797`
- minAnswerSpecificityMargin: `2.0098876953125`
- medianRelationTemplateMargin: `3.4460906982421875`
- minRelationTemplateMargin: `1.2060317993164062`
- medianParaphraseRetentionRate: `1.0`

## Case Summary

| subject | answer | pass | rank | subjectLift | answerMargin | relationTemplateMargin | paraphraseRetention |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| France | Paris | true | 5 | 6.319567 | 2.457001 | 2.406136 | 1.000 |
| Germany | Berlin | true | 2 | 6.161694 | 2.502861 | 2.738525 | 1.000 |
| Italy | Rome | true | 1 | 6.404729 | 3.790779 | 3.188179 | 1.000 |
| Spain | Madrid | true | 1 | 5.259922 | 2.905426 | 2.530426 | 1.000 |
| Portugal | Lisbon | true | 2 | 6.377979 | 2.009888 | 1.206032 | 1.000 |
| Greece | Athens | true | 1 | 7.296121 | 4.728027 | 4.772469 | 1.000 |
| Egypt | Cairo | true | 2 | 9.193455 | 4.059395 | 3.712822 | 1.000 |
| Japan | Tokyo | true | 2 | 8.259727 | 4.423904 | 4.517128 | 1.000 |
| China | Beijing | true | 10 | 8.135350 | 3.207779 | 3.704002 | 1.000 |
| Russia | Moscow | true | 7 | 8.629738 | 3.831200 | 4.574677 | 1.000 |

## Claim Boundary

This gate is output/logit-only. It builds a behavioral counterfactual graph
from prompt interventions and next-token logits. It does not claim original
neuron identity, hidden activation access, hook-based circuit recovery, or
full LLM decompilation.
