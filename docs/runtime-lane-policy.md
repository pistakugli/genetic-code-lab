# Runtime Lane Policy

## Purpose

`Genetic-Code-Lab` ima dve različite vrste runtime lane-ova:
- `performance lane`
- `control lane`

Ove dve stvari ne smeju da se mešaju.

## Performance Lane

Performance lane je svaki entrypoint koji služi da pokaže:
- stvarni benchmark rezultat
- capability proof
- gauntlet rezultat
- realan throughput / resource usage

### Rules

- Ne sme tiho da padne na `parallelEvaluators = 1`.
- Mora eksplicitno da resolve-uje `parallelEvaluators` iz:
  - `GCL_PARALLEL_EVALUATORS`
  - ili `os.availableParallelism()`
- Mora da prosledi `parallelEvaluators` u `EvolutionEngine` ili `SuperEngine`.
- U distributed režimu mora jasno da razdvoji:
  - `distributedIslands`
  - `parallelEvaluators` / `islandParallelEvaluators`
- Nije dozvoljeno da isti knob istovremeno znači i “broj islands” i “broj evaluatora” bez eksplicitne compatibility logike.
- Ako koristi checked-in JS twin, taj twin mora biti usklađen sa TS source-om ili fail-fast označen kao stale.

### Examples

- `tests/65-0-majority-family-gauntlet.ts`
- `tests/65-1-skeleton-honesty-audit.ts`
- `tests/64-2-multiclass-gauntlet.ts`
- `scripts/diagnose-threshold.ts`
- `scripts/benchmark-majority-seeds.ts`

## Control Lane

Control lane je entrypoint koji služi za:
- deterministic repro
- forensic audit
- stability fallback
- minimal bug isolation

### Rules

- Sme namerno da koristi `parallelEvaluators: 1` ili `0`.
- Mora eksplicitno da kaže razlog u komentaru pored konfiguracije.
- Nije validan dokaz da softver “koristi celu mašinu”.

### Allowed Reasons

- `deterministic control lane`
- `forensic repro lane`
- `stability fallback`

## Failure Condition

Za ovaj projekat, sledeće je bug:

- glavni benchmark / gauntlet / proof lane koristi engine default `parallelEvaluators = 1`
- a fajl to ne kaže eksplicitno i ne radi to namerno

Najkraće:
- `performance lane` bez eksplicitnog parallelism resolver-a je bug
- `control lane` bez eksplicitnog objašnjenja je loš audit dizajn

## Current Project Constraint

Auto lane koji pokušava da koristi sve raspoložive workere nije automatski validan production path.

Ako puni lane ruši Node/V8 worker bootstrap, onda:
- bounded-worker lane je trenutno stabilni proof path
- full-machine lane ostaje otvoren infrastrukturni problem dok se ne stabilizuje

## Distributed Runtime Rule

- `DistributedManager` mora da tretira:
  - broj islands kao topološki parametar
  - intra-island evaluator count kao zaseban runtime parametar
- `SuperEngine` sme da zadrži legacy compatibility samo ako:
  - novi call-site ne prosledi `distributedIslands`
  - i tada je eksplicitno jasno da je `parallelEvaluators` legacy island-count behavior
