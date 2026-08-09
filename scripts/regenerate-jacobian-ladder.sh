#!/bin/bash
# Regenerate the six Jacobian ladder artifacts, and refuse to produce ones that are stale the
# moment they are written.
#
# Three times in one session I started a regeneration, edited the solver while it ran, and
# ended up with artifacts whose recorded generator hash no longer matched the file - the exact
# state the stale-artifact check exists to catch, manufactured by hand. The loop takes about
# fifteen minutes and an edit in that window is easy to make and invisible until the verifier
# runs much later.
#
# So the hash is taken before and after. A change in between means every artifact written by
# this run is already wrong, and saying so immediately is cheaper than finding out from a gate.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTORCH_NVML_BASED_CUDA_CHECK=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

SOLVER=tools/llm_reader/jacobian_boundary_recovery.py
ORACLE=tools/llm_reader/blind_swiglu_oracle_decompiler.py
before=$(sha256sum "$SOLVER" "$ORACLE" | sha256sum | cut -d' ' -f1)

if [ -n "$(git status --porcelain "$SOLVER" "$ORACLE")" ]; then
  echo "refusing: solver or oracle has uncommitted changes, so the artifacts would record a hash nobody can check out" >&2
  exit 2
fi

failed=()
for variant in 16 32 64 128 256 512; do
  if [ "$variant" = "64" ]; then
    script="benchmark:nn-live-jacobian-boundary-recovery"
  else
    script="benchmark:nn-live-jacobian-boundary-recovery:$variant"
  fi
  if nice -n 10 npm run -s "$script" > "/tmp/ladder-$variant.out" 2>&1; then
    echo "$variant ok $(date +%T)"
  else
    echo "$variant FAILED $(date +%T), see /tmp/ladder-$variant.out" >&2
    failed+=("$variant")
  fi
done

after=$(sha256sum "$SOLVER" "$ORACLE" | sha256sum | cut -d' ' -f1)
if [ "$before" != "$after" ]; then
  echo "the solver changed while this ran: every artifact just written is already stale" >&2
  exit 3
fi

if [ ${#failed[@]} -gt 0 ]; then
  echo "failed: ${failed[*]}" >&2
  exit 1
fi
echo "all six regenerated against an unchanged solver"
