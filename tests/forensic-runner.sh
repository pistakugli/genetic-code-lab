#!/bin/bash
# FORENSIC REGRESSION RUNNER (PHASE 62.6)
# Standardized Gatekeeper for GP Identity and Survival Proofs.
# Exit 0 if all tests PASS.

echo "--- STARTING FORENSIC REGRESSION SUITE ---"

# 1. Round-trip Identity (AST-Binary-AST)
echo "Running Round-trip Identity Proof..."
npx ts-node tests/discovery-agent-roundtrip.ts
if [ $? -ne 0 ]; then
    echo "ERROR: Round-trip identity failed!"
    exit 1
fi

# 2. Agent Isolation (Skeleton + Refinement to 1.0)
echo "Running Discovery Agent Isolation Proof..."
npx ts-node tests/discovery-agent-isolation.ts
if [ $? -ne 0 ]; then
    echo "ERROR: Agent isolation failed!"
    exit 1
fi

# 3. Injection Survival (5-generation structural persistence)
echo "Running Injection Survival Proof (Tournament)..."
npx ts-node tests/injection-survival-proof.ts
if [ $? -ne 0 ]; then
    echo "ERROR: Injection survival failed!"
    exit 1
fi

echo "--- FORENSIC REGRESSION SUITE: ALL PASS ---"
exit 0
