# GPT-2 autoregressive causal computation trace

- Protocol: `autoregressive`
- Access mode: `instrumented-witness`
- Model: `gpt2` (`GPT2LMHeadModel`), device `cuda:0`
- Source artifact SHA-256: `f24428b69b9b1fb2d750dfbf2ea55a065fc1d6cc740c415d01ac6ccb94e3715b`
- Passed source cases: 5
- Generated prediction steps: 60
- Deterministic matched controls per prediction: 100
- Maximum generation length: 12

The source paths and prompts are read-only inputs from the existing ordered-path artifact. 
Each causal test is a one-step intervention on the unchanged baseline-generated context; 
intervened tokens are not fed into later baseline contexts. Teacher-forced source-token scores 
are reported separately from free-running predictions.

## Per-case acceptance criteria

| Subject | A trace | B greedy reconstruction | C influence | D matched specificity | E interaction measured | F actual greedy explained | G provenance | Sufficiency | Greedy tokens |
|---|---|---|---|---|---|---|---|---|---:|
| France | pass | pass | pass | fail | pass | pass | pass | not-tested | the  capital  of  the  French  Republic ,  and  the  capital  of  the |

## France: The capital of France is

Source fact token: ` Paris`; generated 12 greedy tokens. All causal effects below use the unchanged baseline-generated context for that step.

### Teacher-forced score (not free-running generation)

The source artifact's manually selected fact token had rank 5 on the original prompt; the model's actual first greedy token was ` the`. Selected-path ablation changed the teacher-forced target logit by 1.818840.

### Token 1: ` the` (id 262)

Context token IDs: `[464, 3139, 286, 4881, 318]`. The actual greedy token had logit -100.249771, probability 0.084592521; top-five competitors: ` the` (0.084593), ` now` (0.047946), ` a` (0.04616), ` France` (0.032377), ` Paris` (0.032244).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L11 unit 1369 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.37e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 1.256737 and its probability by -3.452599e-05; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 99.0 (absolute-effect percentile 99.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.003761 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 2: ` capital` (id 3139)

Context token IDs: `[464, 3139, 286, 4881, 318, 262]`. The actual greedy token had logit -97.601990, probability 0.16968498; top-five competitors: ` capital` (0.16968), ` city` (0.03353), ` French` (0.023679), ` most` (0.022424), ` largest` (0.022121).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L11 unit 1369 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.29e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by -0.089455 and its probability by 0.0048312098; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 0.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.000000 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 3: ` of` (id 286)

Context token IDs: `[464, 3139, 286, 4881, 318, 262, 3139]`. The actual greedy token had logit -49.064934, probability 0.92708766; top-five competitors: ` of` (0.92709), `,` (0.019319), ` city` (0.014007), ` and` (0.0062949), `.` (0.0055876).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L11 unit 1369 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 7.96e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by -0.659668 and its probability by -0.00010228157; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 0.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.000000 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 4: ` the` (id 262)

Context token IDs: `[464, 3139, 286, 4881, 318, 262, 3139, 286]`. The actual greedy token had logit -73.677658, probability 0.28278133; top-five competitors: ` the` (0.28278), ` France` (0.20279), ` a` (0.057721), ` Europe` (0.014461), ` French` (0.012675).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L11 unit 1369 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 8.11e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.005341 and its probability by -0.0009469986; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 0.0 (absolute-effect percentile 0.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.000000 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 5: ` French` (id 4141)

Context token IDs: `[464, 3139, 286, 4881, 318, 262, 3139, 286, 262]`. The actual greedy token had logit -88.884140, probability 0.13667817; top-five competitors: ` French` (0.13668), ` Republic` (0.052175), ` European` (0.035652), ` United` (0.035225), ` world` (0.032136).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L11 unit 1369 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.1e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.009766 and its probability by -0.00074294209; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 0.0 (absolute-effect percentile 0.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.000000 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 6: ` Republic` (id 2066)

Context token IDs: `[464, 3139, 286, 4881, 318, 262, 3139, 286, 262, 4141]`. The actual greedy token had logit -96.783524, probability 0.30062601; top-five competitors: ` Republic` (0.30063), ` Empire` (0.080694), ` Revolution` (0.054633), ` republic` (0.04983), ` people` (0.033652).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L11 unit 1369 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 7.27e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.002701 and its probability by -0.0012111366; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 0.0 (absolute-effect percentile 0.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.000000 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 7: `,` (id 11)

Context token IDs: `[464, 3139, 286, 4881, 318, 262, 3139, 286, 262, 4141, 2066]`. The actual greedy token had logit -57.253181, probability 0.33913189; top-five competitors: `,` (0.33913), `.` (0.33672), ` and` (0.13086), ` (` (0.017511), `;` (0.017149).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L11 unit 1369 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.14e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by -0.172890 and its probability by -0.0018704832; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 0.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.000000 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token changed; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 8: ` and` (id 290)

Context token IDs: `[464, 3139, 286, 4881, 318, 262, 3139, 286, 262, 4141, 2066, 11]`. The actual greedy token had logit -102.214394, probability 0.28098443; top-five competitors: ` and` (0.28098), ` which` (0.098807), ` the` (0.083649), ` but` (0.047611), ` France` (0.029248).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L11 unit 1369 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.34e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by -0.057587 and its probability by 0.00067785382; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 0.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.000000 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 9: ` the` (id 262)

Context token IDs: `[464, 3139, 286, 4881, 318, 262, 3139, 286, 262, 4141, 2066, 11, 290]`. The actual greedy token had logit -112.741531, probability 0.1633528; top-five competitors: ` the` (0.16335), ` its` (0.079573), ` it` (0.075469), ` is` (0.060774), ` in` (0.030469).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L11 unit 1369 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.53e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by -0.113342 and its probability by -0.00077958405; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 0.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.000000 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 10: ` capital` (id 3139)

Context token IDs: `[464, 3139, 286, 4881, 318, 262, 3139, 286, 262, 4141, 2066, 11, 290, 262]`. The actual greedy token had logit -99.431747, probability 0.10518589; top-five competitors: ` capital` (0.10519), ` French` (0.059371), ` country` (0.033508), ` city` (0.018096), ` Republic` (0.01439).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L11 unit 1369 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.41e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by -0.058464 and its probability by 0.00099418312; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 0.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.000000 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 11: ` of` (id 286)

Context token IDs: `[464, 3139, 286, 4881, 318, 262, 3139, 286, 262, 4141, 2066, 11, 290, 262, 3139]`. The actual greedy token had logit -24.733545, probability 0.94119895; top-five competitors: ` of` (0.9412), ` is` (0.015447), ` and` (0.0038546), ` city` (0.003478), `,` (0.0033311).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L11 unit 1369 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.35e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by -0.037334 and its probability by -5.0902367e-05; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 0.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.000000 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 12: ` the` (id 262)

Context token IDs: `[464, 3139, 286, 4881, 318, 262, 3139, 286, 262, 4141, 2066, 11, 290, 262, 3139, 286]`. The actual greedy token had logit -82.143906, probability 0.39687112; top-five competitors: ` the` (0.39687), ` France` (0.23412), ` Europe` (0.041196), ` a` (0.024137), ` its` (0.023626).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L11 unit 1369 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by -0.098778 and its probability by -0.00076496601; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 0.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.000000 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.


| Germany | pass | pass | pass | fail | pass | pass | pass | not-tested | the  capital  of  the  German  state ,  and  the  capital  of  the |

## Germany: The capital of Germany is

Source fact token: ` Berlin`; generated 12 greedy tokens. All causal effects below use the unchanged baseline-generated context for that step.

### Teacher-forced score (not free-running generation)

The source artifact's manually selected fact token had rank 2 on the original prompt; the model's actual first greedy token was ` the`. Selected-path ablation changed the teacher-forced target logit by 1.388084.

### Token 1: ` the` (id 262)

Context token IDs: `[464, 3139, 286, 4486, 318]`. The actual greedy token had logit -97.949272, probability 0.089100756; top-five competitors: ` the` (0.089101), ` Berlin` (0.052727), ` now` (0.051712), ` a` (0.043828), ` in` (0.032007).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 687 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.48e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 1.164284 and its probability by 0.00023633242; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.048340 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 2: ` capital` (id 3139)

Context token IDs: `[464, 3139, 286, 4486, 318, 262]`. The actual greedy token had logit -96.386452, probability 0.12579028; top-five competitors: ` capital` (0.12579), ` city` (0.041452), ` largest` (0.033634), ` most` (0.026248), ` second` (0.019833).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 687 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 7.63e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.606552 and its probability by 0.0059125423; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.017822 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 3: ` of` (id 286)

Context token IDs: `[464, 3139, 286, 4486, 318, 262, 3139]`. The actual greedy token had logit -49.150902, probability 0.91522968; top-five competitors: ` of` (0.91523), ` city` (0.023535), `,` (0.014417), ` Berlin` (0.0078979), ` and` (0.0056655).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 687 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.26e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.640869 and its probability by -0.0016127229; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.010029 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 4: ` the` (id 262)

Context token IDs: `[464, 3139, 286, 4486, 318, 262, 3139, 286]`. The actual greedy token had logit -72.425644, probability 0.37917605; top-five competitors: ` the` (0.37918), ` Germany` (0.10293), ` a` (0.047638), ` Europe` (0.020042), ` Austria` (0.015611).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 687 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.34e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.401596 and its probability by -0.0042802989; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.017990 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 5: ` German` (id 2679)

Context token IDs: `[464, 3139, 286, 4486, 318, 262, 3139, 286, 262]`. The actual greedy token had logit -88.575745, probability 0.06776204; top-five competitors: ` German` (0.067762), ` European` (0.040266), ` world` (0.035782), ` United` (0.034561), ` Reich` (0.027653).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 687 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 6.68e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.259560 and its probability by 0.00041820109; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.007942 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 6: ` state` (id 1181)

Context token IDs: `[464, 3139, 286, 4486, 318, 262, 3139, 286, 262, 2679]`. The actual greedy token had logit -87.256950, probability 0.17990452; top-five competitors: ` state` (0.1799), ` Reich` (0.12235), ` people` (0.073025), ` People` (0.067263), ` Empire` (0.060159).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 687 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 7.63e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.137520 and its probability by -0.0030854493; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.008110 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 7: `,` (id 11)

Context token IDs: `[464, 3139, 286, 4486, 318, 262, 3139, 286, 262, 2679, 1181]`. The actual greedy token had logit -61.534294, probability 0.33534864; top-five competitors: `,` (0.33535), `.` (0.28996), ` and` (0.1428), ` of` (0.035777), `,"` (0.027322).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 687 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.1e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.341095 and its probability by -0.00050875545; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.003700 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 8: ` and` (id 290)

Context token IDs: `[464, 3139, 286, 4486, 318, 262, 3139, 286, 262, 2679, 1181, 11]`. The actual greedy token had logit -100.316910, probability 0.27985218; top-five competitors: ` and` (0.27985), ` the` (0.09383), ` but` (0.057259), ` which` (0.054524), ` with` (0.031469).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 687 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 7.39e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.210587 and its probability by 0.0006197989; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.006317 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 9: ` the` (id 262)

Context token IDs: `[464, 3139, 286, 4486, 318, 262, 3139, 286, 262, 2679, 1181, 11, 290]`. The actual greedy token had logit -111.876160, probability 0.17174208; top-five competitors: ` the` (0.17174), ` its` (0.074734), ` it` (0.073821), ` is` (0.050755), ` in` (0.034113).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 687 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.48e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.221420 and its probability by -0.0002873987; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.006844 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 10: ` capital` (id 3139)

Context token IDs: `[464, 3139, 286, 4486, 318, 262, 3139, 286, 262, 2679, 1181, 11, 290, 262]`. The actual greedy token had logit -99.720642, probability 0.097339526; top-five competitors: ` capital` (0.09734), ` state` (0.053391), ` German` (0.052958), ` city` (0.026973), ` country` (0.023683).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 687 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.14e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.091431 and its probability by 0.0022472292; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 83.0 (absolute-effect percentile 83.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.003113 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 11: ` of` (id 286)

Context token IDs: `[464, 3139, 286, 4486, 318, 262, 3139, 286, 262, 2679, 1181, 11, 290, 262, 3139]`. The actual greedy token had logit -30.915760, probability 0.92700046; top-five competitors: ` of` (0.927), ` is` (0.019997), ` city` (0.0045575), ` and` (0.0039584), ` in` (0.0031303).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 687 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 7.51e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.285561 and its probability by 0.00013494492; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.002930 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 12: ` the` (id 262)

Context token IDs: `[464, 3139, 286, 4486, 318, 262, 3139, 286, 262, 2679, 1181, 11, 290, 262, 3139, 286]`. The actual greedy token had logit -77.975983, probability 0.52808785; top-five competitors: ` the` (0.52809), ` Germany` (0.10943), ` a` (0.033377), ` Europe` (0.030473), ` its` (0.027564).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 687 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 8.58e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.024818 and its probability by -0.0018041134; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 67.0 (absolute-effect percentile 64.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.009323 (subadditive-overlap-consistent).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.


| Italy | pass | pass | pass | fail | pass | pass | pass | not-tested | Rome ,  and  the  capital  of  the  world  is  Rome .  The |

## Italy: The capital of Italy is

Source fact token: ` Rome`; generated 12 greedy tokens. All causal effects below use the unchanged baseline-generated context for that step.

### Teacher-forced score (not free-running generation)

The source artifact's manually selected fact token had rank 1 on the original prompt; the model's actual first greedy token was ` Rome`. Selected-path ablation changed the teacher-forced target logit by 3.218582.

### Token 1: ` Rome` (id 10598)

Context token IDs: `[464, 3139, 286, 8031, 318]`. The actual greedy token had logit -95.563095, probability 0.15734816; top-five competitors: ` Rome` (0.15735), ` the` (0.073164), ` Milan` (0.046878), ` a` (0.034498), ` now` (0.032003).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 2746 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.34e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 3.218582 and its probability by -0.067001462; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.044197 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 2: `,` (id 11)

Context token IDs: `[464, 3139, 286, 8031, 318, 10598]`. The actual greedy token had logit -60.646454, probability 0.49245346; top-five competitors: `,` (0.49245), `.` (0.2332), ` and` (0.13514), ` but` (0.018874), `;` (0.011739).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 2746 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.38e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.970695 and its probability by -0.0098958313; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.028584 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 3: ` and` (id 290)

Context token IDs: `[464, 3139, 286, 8031, 318, 10598, 11]`. The actual greedy token had logit -91.908569, probability 0.25230825; top-five competitors: ` and` (0.25231), ` but` (0.088658), ` the` (0.081755), ` where` (0.077153), ` which` (0.066343).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 2746 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.1e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.483658 and its probability by -0.0022783577; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.013290 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 4: ` the` (id 262)

Context token IDs: `[464, 3139, 286, 8031, 318, 10598, 11, 290]`. The actual greedy token had logit -106.403793, probability 0.18481545; top-five competitors: ` the` (0.18482), ` it` (0.079267), ` its` (0.066835), ` Rome` (0.033705), ` is` (0.031545).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 2746 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.24e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.134239 and its probability by -0.0010309219; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 59.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.004883 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 5: ` capital` (id 3139)

Context token IDs: `[464, 3139, 286, 8031, 318, 10598, 11, 290, 262]`. The actual greedy token had logit -94.504654, probability 0.1587359; top-five competitors: ` capital` (0.15874), ` city` (0.1478), ` country` (0.024233), ` Italian` (0.011015), ` largest` (0.0098167).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 2746 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 7.63e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.254044 and its probability by 0.00069627166; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.000320 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 6: ` of` (id 286)

Context token IDs: `[464, 3139, 286, 8031, 318, 10598, 11, 290, 262, 3139]`. The actual greedy token had logit -54.821133, probability 0.72220737; top-five competitors: ` of` (0.72221), ` is` (0.08713), ` city` (0.03259), ` has` (0.016814), `'s` (0.0090196).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 2746 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.41e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.363884 and its probability by -0.00013566017; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.002792 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 7: ` the` (id 262)

Context token IDs: `[464, 3139, 286, 8031, 318, 10598, 11, 290, 262, 3139, 286]`. The actual greedy token had logit -84.579330, probability 0.26245072; top-five competitors: ` the` (0.26245), ` Italy` (0.13403), ` Rome` (0.081121), ` Spain` (0.033436), ` Greece` (0.032272).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 2746 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.05e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.095352 and its probability by -0.0011238158; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 21.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.002197 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 8: ` world` (id 995)

Context token IDs: `[464, 3139, 286, 8031, 318, 10598, 11, 290, 262, 3139, 286, 262]`. The actual greedy token had logit -95.813690, probability 0.072345801; top-five competitors: ` world` (0.072346), ` Roman` (0.048114), ` Republic` (0.041022), ` United` (0.038238), ` Italian` (0.032329).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 2746 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.24e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.119415 and its probability by -0.00010160357; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 65.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.002342 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 9: ` is` (id 318)

Context token IDs: `[464, 3139, 286, 8031, 318, 10598, 11, 290, 262, 3139, 286, 262, 995]`. The actual greedy token had logit -72.087891, probability 0.63585478; top-five competitors: ` is` (0.63585), `.` (0.097262), `,` (0.070368), `'s` (0.028525), ` has` (0.014396).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 2746 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 6.2e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.341248 and its probability by 0.0042221546; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.011917 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 10: ` Rome` (id 10598)

Context token IDs: `[464, 3139, 286, 8031, 318, 10598, 11, 290, 262, 3139, 286, 262, 995, 318]`. The actual greedy token had logit -103.036331, probability 0.23906949; top-five competitors: ` Rome` (0.23907), ` London` (0.076056), ` New` (0.050106), ` the` (0.045359), ` Berlin` (0.038851).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 2746 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 9.54e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.207764 and its probability by -0.009722665; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.001213 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 11: `.` (id 13)

Context token IDs: `[464, 3139, 286, 8031, 318, 10598, 11, 290, 262, 3139, 286, 262, 995, 318, 10598]`. The actual greedy token had logit -57.423222, probability 0.62421477; top-five competitors: `.` (0.62421), `,` (0.17011), `."` (0.035826), `,"` (0.029117), ` itself` (0.022158).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 2746 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 6.2e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.451466 and its probability by -0.00067955256; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.008591 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 12: ` The` (id 383)

Context token IDs: `[464, 3139, 286, 8031, 318, 10598, 11, 290, 262, 3139, 286, 262, 995, 318, 10598, 13]`. The actual greedy token had logit -152.702560, probability 0.12493528; top-five competitors: ` The` (0.12494), `\n` (0.1186), ` It` (0.062436), ` And` (0.050071), ` Rome` (0.046604).

**WHAT:** The unchanged source path includes attention L9 unit 8 at positions [4], mlp L9 unit 2746 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.53e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by -0.097595 and its probability by 5.5655837e-05; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 0.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.001266 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.


| Portugal | pass | pass | pass | fail | pass | pass | pass | not-tested | the  capital  of  the  Portuguese  Republic ,  and  the  capital  of  the |

## Portugal: The capital of Portugal is

Source fact token: ` Lisbon`; generated 12 greedy tokens. All causal effects below use the unchanged baseline-generated context for that step.

### Teacher-forced score (not free-running generation)

The source artifact's manually selected fact token had rank 2 on the original prompt; the model's actual first greedy token was ` the`. Selected-path ablation changed the teacher-forced target logit by 1.984413.

### Token 1: ` the` (id 262)

Context token IDs: `[464, 3139, 286, 19101, 318]`. The actual greedy token had logit -95.174965, probability 0.091947228; top-five competitors: ` the` (0.091947), ` Lisbon` (0.061008), ` a` (0.052165), ` now` (0.041694), ` also` (0.027128).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L6 unit 2941 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 7.63e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.978088 and its probability by -0.0061233565; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.053696 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 2: ` capital` (id 3139)

Context token IDs: `[464, 3139, 286, 19101, 318, 262]`. The actual greedy token had logit -92.672272, probability 0.16191801; top-five competitors: ` capital` (0.16192), ` city` (0.060824), ` largest` (0.02649), ` port` (0.022592), ` country` (0.021442).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L6 unit 2941 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.24e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by -0.224457 and its probability by -0.02874437; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 0.0 (absolute-effect percentile 8.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.037422 (subadditive-overlap-consistent).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 3: ` of` (id 286)

Context token IDs: `[464, 3139, 286, 19101, 318, 262, 3139]`. The actual greedy token had logit -49.694969, probability 0.8499679; top-five competitors: ` of` (0.84997), ` city` (0.041989), `,` (0.032229), ` Lisbon` (0.018689), ` and` (0.0081513).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L6 unit 2941 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 8.58e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by -1.489006 and its probability by 0.019824922; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 0.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.035236 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 4: ` the` (id 262)

Context token IDs: `[464, 3139, 286, 19101, 318, 262, 3139, 286]`. The actual greedy token had logit -70.184471, probability 0.26063591; top-five competitors: ` the` (0.26064), ` Portugal` (0.17804), ` a` (0.039186), ` Lisbon` (0.019264), ` Spain` (0.01312).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L6 unit 2941 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 7.63e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by -0.147209 and its probability by 0.0069176853; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 0.0 (absolute-effect percentile 90.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.002266 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 5: ` Portuguese` (id 21813)

Context token IDs: `[464, 3139, 286, 19101, 318, 262, 3139, 286, 262]`. The actual greedy token had logit -80.123878, probability 0.088896096; top-five competitors: ` Portuguese` (0.088896), ` European` (0.028789), ` Republic` (0.028144), ` United` (0.025427), ` Spanish` (0.020827).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L6 unit 2941 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.24e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.029732 and its probability by -0.00053633749; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 0.0 (absolute-effect percentile 0.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.007515 (subadditive-overlap-consistent).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 6: ` Republic` (id 2066)

Context token IDs: `[464, 3139, 286, 19101, 318, 262, 3139, 286, 262, 21813]`. The actual greedy token had logit -91.672043, probability 0.121517; top-five competitors: ` Republic` (0.12152), ` state` (0.048928), ` republic` (0.042233), ` Empire` (0.037916), ` people` (0.036113).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L6 unit 2941 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 7.15e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.077896 and its probability by -0.0030106902; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 99.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.007446 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 7: `,` (id 11)

Context token IDs: `[464, 3139, 286, 19101, 318, 262, 3139, 286, 262, 21813, 2066]`. The actual greedy token had logit -53.042488, probability 0.34254187; top-five competitors: `,` (0.34254), `.` (0.31118), ` and` (0.14826), ` of` (0.043265), ` (` (0.02498).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L6 unit 2941 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 9.54e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.123108 and its probability by 0.00063246489; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 7.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.011097 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 8: ` and` (id 290)

Context token IDs: `[464, 3139, 286, 19101, 318, 262, 3139, 286, 262, 21813, 2066, 11]`. The actual greedy token had logit -101.393982, probability 0.24866593; top-five competitors: ` and` (0.24867), ` which` (0.13383), ` the` (0.072109), ` but` (0.041076), ` with` (0.039147).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L6 unit 2941 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.1e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.134567 and its probability by 0.00061504543; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 0.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.009521 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 9: ` the` (id 262)

Context token IDs: `[464, 3139, 286, 19101, 318, 262, 3139, 286, 262, 21813, 2066, 11, 290]`. The actual greedy token had logit -108.192673, probability 0.15472586; top-five competitors: ` the` (0.15473), ` is` (0.090086), ` it` (0.078529), ` its` (0.076904), ` has` (0.035853).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L6 unit 2941 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.38e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.192749 and its probability by 0.00021164119; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.012459 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 10: ` capital` (id 3139)

Context token IDs: `[464, 3139, 286, 19101, 318, 262, 3139, 286, 262, 21813, 2066, 11, 290, 262]`. The actual greedy token had logit -95.854187, probability 0.18954132; top-five competitors: ` capital` (0.18954), ` Portuguese` (0.069692), ` country` (0.040937), ` city` (0.037067), ` largest` (0.012885).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L6 unit 2941 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 8.82e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by -0.069138 and its probability by -0.0038588047; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 0.0 (absolute-effect percentile 6.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.004700 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 11: ` of` (id 286)

Context token IDs: `[464, 3139, 286, 19101, 318, 262, 3139, 286, 262, 21813, 2066, 11, 290, 262, 3139]`. The actual greedy token had logit -28.368416, probability 0.92777538; top-five competitors: ` of` (0.92778), ` is` (0.01762), ` city` (0.013436), ` and` (0.0036969), `,` (0.0027499).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L6 unit 2941 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.12e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by -0.080105 and its probability by -0.00077265501; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 67.0 (absolute-effect percentile 32.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.003532 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 12: ` the` (id 262)

Context token IDs: `[464, 3139, 286, 19101, 318, 262, 3139, 286, 262, 21813, 2066, 11, 290, 262, 3139, 286]`. The actual greedy token had logit -74.183105, probability 0.3432273; top-five competitors: ` the` (0.34323), ` Portugal` (0.29896), ` Spain` (0.034964), ` Lisbon` (0.020999), ` its` (0.016238).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L6 unit 2941 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.24e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.117386 and its probability by 0.0014595091; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 78.0 (absolute-effect percentile 78.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.002068 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 0/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.


| Greece | pass | pass | pass | fail | pass | pass | pass | not-tested | Athens ,  and  the  capital  of  the  country  is  Athens .  The |

## Greece: The capital of Greece is

Source fact token: ` Athens`; generated 12 greedy tokens. All causal effects below use the unchanged baseline-generated context for that step.

### Teacher-forced score (not free-running generation)

The source artifact's manually selected fact token had rank 1 on the original prompt; the model's actual first greedy token was ` Athens`. Selected-path ablation changed the teacher-forced target logit by 2.556099.

### Token 1: ` Athens` (id 21891)

Context token IDs: `[464, 3139, 286, 10315, 318]`. The actual greedy token had logit -99.606049, probability 0.17790157; top-five competitors: ` Athens` (0.1779), ` the` (0.077141), ` now` (0.046067), ` a` (0.044569), ` in` (0.03031).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L9 unit 1993 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 2.556099 and its probability by -0.11001961; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.019325 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token changed; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 24/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 2: `,` (id 11)

Context token IDs: `[464, 3139, 286, 10315, 318, 21891]`. The actual greedy token had logit -62.576019, probability 0.49518219; top-five competitors: `,` (0.49518), `.` (0.23472), ` and` (0.12747), ` but` (0.012841), ` (` (0.010409).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L9 unit 1993 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 5.72e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.638718 and its probability by 0.0012257397; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 50.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.003689 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 23/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 3: ` and` (id 290)

Context token IDs: `[464, 3139, 286, 10315, 318, 21891, 11]`. The actual greedy token had logit -93.869873, probability 0.16088492; top-five competitors: ` and` (0.16088), ` the` (0.14349), ` a` (0.082469), ` which` (0.081626), ` home` (0.078342).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L9 unit 1993 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.14e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.340424 and its probability by -0.0030233264; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 72.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.003738 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 24/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 4: ` the` (id 262)

Context token IDs: `[464, 3139, 286, 10315, 318, 21891, 11, 290]`. The actual greedy token had logit -108.071594, probability 0.18079783; top-five competitors: ` the` (0.1808), ` its` (0.10704), ` it` (0.070964), ` is` (0.037803), ` in` (0.026857).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L9 unit 1993 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.1e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.220001 and its probability by 0.0020956695; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.001770 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 23/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 5: ` capital` (id 3139)

Context token IDs: `[464, 3139, 286, 10315, 318, 21891, 11, 290, 262]`. The actual greedy token had logit -100.746010, probability 0.20710632; top-five competitors: ` capital` (0.20711), ` city` (0.087999), ` country` (0.069108), ` Greek` (0.047101), ` island` (0.019046).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L9 unit 1993 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.1e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.195808 and its probability by -0.010083959; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 88.0 (absolute-effect percentile 88.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.001434 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 24/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 6: ` of` (id 286)

Context token IDs: `[464, 3139, 286, 10315, 318, 21891, 11, 290, 262, 3139]`. The actual greedy token had logit -60.993698, probability 0.62920475; top-five competitors: ` of` (0.6292), ` is` (0.10357), ` city` (0.057043), ` has` (0.033696), `'s` (0.015397).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L9 unit 1993 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 5.84e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by -0.390781 and its probability by 0.010876775; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 50.0 (absolute-effect percentile 50.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.007408 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 23/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 7: ` the` (id 262)

Context token IDs: `[464, 3139, 286, 10315, 318, 21891, 11, 290, 262, 3139, 286]`. The actual greedy token had logit -84.441833, probability 0.37286612; top-five competitors: ` the` (0.37287), ` Greece` (0.18536), ` Europe` (0.039022), ` its` (0.026892), ` Cyprus` (0.018807).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L9 unit 1993 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 8.11e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.231346 and its probability by 0.00082954764; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.004356 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 23/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 8: ` country` (id 1499)

Context token IDs: `[464, 3139, 286, 10315, 318, 21891, 11, 290, 262, 3139, 286, 262]`. The actual greedy token had logit -101.419563, probability 0.090038933; top-five competitors: ` country` (0.090039), ` Republic` (0.059308), ` Greek` (0.054294), ` European` (0.051484), ` euro` (0.040412).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L9 unit 1993 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 9.54e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.342674 and its probability by -0.00081002712; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 100.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.002831 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 24/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 9: ` is` (id 318)

Context token IDs: `[464, 3139, 286, 10315, 318, 21891, 11, 290, 262, 3139, 286, 262, 1499]`. The actual greedy token had logit -89.407387, probability 0.60641223; top-five competitors: ` is` (0.60641), `'s` (0.068555), ` has` (0.047402), `,` (0.039464), ` was` (0.017926).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L9 unit 1993 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 9.89e-06. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by -0.524368 and its probability by 0.0075679421; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 37.0 (absolute-effect percentile 63.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.007767 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 24/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 10: ` Athens` (id 21891)

Context token IDs: `[464, 3139, 286, 10315, 318, 21891, 11, 290, 262, 3139, 286, 262, 1499, 318]`. The actual greedy token had logit -101.066238, probability 0.54830867; top-five competitors: ` Athens` (0.54831), ` the` (0.096658), ` its` (0.018563), ` also` (0.013551), ` called` (0.012973).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L9 unit 1993 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.51e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by 0.101990 and its probability by -0.0020903945; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 100.0 (absolute-effect percentile 91.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was -0.005959 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 24/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 11: `.` (id 13)

Context token IDs: `[464, 3139, 286, 10315, 318, 21891, 11, 290, 262, 3139, 286, 262, 1499, 318, 21891]`. The actual greedy token had logit -64.956367, probability 0.50670457; top-five competitors: `.` (0.5067), `,` (0.23325), `-` (0.043367), ` (` (0.022568), ` and` (0.020453).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L9 unit 1993 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.14e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by -0.119743 and its probability by -2.5629997e-06; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 62.0 (absolute-effect percentile 38.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.005989 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 24/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.

### Token 12: ` The` (id 383)

Context token IDs: `[464, 3139, 286, 10315, 318, 21891, 11, 290, 262, 3139, 286, 262, 1499, 318, 21891, 13]`. The actual greedy token had logit -145.802063, probability 0.16347751; top-five competitors: ` The` (0.16348), `\n` (0.14849), ` It` (0.065924), ` In` (0.050319), ` And` (0.034176).

**WHAT:** The unchanged source path includes attention L6 unit 1 at positions [4], mlp L9 unit 1993 at positions [4]. Every layer records residual/attention/MLP values; the largest first-order contributors are stored per layer and in the raw tensor archive.

**HOW:** The computation was replayed on the exact growing context. Layerwise residual-add maximum error was 1.23e-05. The archive stores the read-position residual inputs/outputs, each attention output and per-head projected writes, MLP inputs/preactivations/activations/outputs, per-neuron first-order target contributions, final normalization, and full vocabulary logits.

**WHY:** Ablating the discovered group changed the actual next-token logit by -0.044846 and its probability by 0.0011293292; restoration maximum full-logit error was 0. Its signed target-logit drop was at percentile 51.0 (absolute-effect percentile 0.0) among 100 deterministic structurally matched-control draws (100 unique groups). The interaction term (group drop minus summed individual drops) was 0.001801 (approximately-additive).

**WHAT ELSE:** After path ablation the greedy token did not change; the altered top five and full-vocabulary TV/JS changes are recorded. Matched controls had 24/100 groups within the preregistered activation-magnitude tolerance.

**LIMITATIONS:** This is a local-weight, hook-instrumented measurement. Ablations estimate causal influence on one next-token distribution conditional on the baseline context; when an intervention changes the greedy token, later free-running contexts are not regenerated from that altered output and are not interpreted as isolated downstream effects. Path sufficiency is not tested, and first-order attributions do not assign semantic meanings to units.


## Protocol and claim limits

This instrumented hook-based experiment is not black-box decompilation or a complete semantic circuit. Causal influence is distinct from matched-control specificity; path sufficiency is not tested. If an intervention changes a token, later steps remain conditioned on the unmodified baseline-generated context. No semantic meanings are assigned to individual units.

## Artifacts

- `report.json`: every case, step, intervention, control draw, and explanation.
- `raw-tensors.npz`: read-position transformer traces and full-vocabulary baseline/selected-path/individual logits.
- `manifest.json`: wrapper provenance and generated variant metadata.
