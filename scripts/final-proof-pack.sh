#!/bin/bash
set -e

echo "=== STARTING FINAL REPRODUCIBLE PROOF PACK ==="

mkdir -p proof_pack

# 1. Environment Info
echo "--- ENVIRONMENT ---" > proof_pack/environment.txt
git rev-parse HEAD >> proof_pack/environment.txt || echo "No git repo" >> proof_pack/environment.txt
git status --short >> proof_pack/environment.txt || true
echo "Node: $(node -v)" >> proof_pack/environment.txt
echo "NPM: $(npm -v)" >> proof_pack/environment.txt
echo "TS-Node: $(npx ts-node -v)" >> proof_pack/environment.txt
echo "Date: $(date)" >> proof_pack/environment.txt
echo "PWD: $(pwd)" >> proof_pack/environment.txt

# 2. Run Tests
echo "Running Forensic Test (if exists)..."
npm run test:forensic > proof_pack/forensic.txt 2>&1 || echo "npm run test:forensic failed or not found" >> proof_pack/forensic.txt

echo "Running 72-1 Quotient Validity Audit..."
npx ts-node tests/72-1-quotient-validity-audit.ts > proof_pack/72_1_validity.txt 2>&1 || true

echo "Running 73 Rational Family Generalization Audit..."
npx ts-node tests/73-rational-family-generalization.ts > proof_pack/73_family.txt 2>&1 || true

echo "Running 74-1 A3 Stabilization Audit..."
NODE_OPTIONS=--max-old-space-size=8192 npx ts-node tests/74-1-a3-stabilization.ts > proof_pack/74_1_a3.txt 2>&1 || true

# 3. Extract Final Formulas
echo "Extracting Final Formulas..."
> proof_pack/final_formulas.txt
grep " | P" proof_pack/73_family.txt | awk -F'\`' '{print $2}' >> proof_pack/final_formulas.txt || true
grep " | A3 " proof_pack/74_1_a3.txt | awk -F'\`' '{print $2}' >> proof_pack/final_formulas.txt || true

# 4. Pointwise Proof
echo "Running Pointwise Proof..."
npx ts-node tests/final-pointwise-proof.ts > proof_pack/pointwise_proof.txt 2>&1 || true

# 5. README
cat << 'EOF' > proof_pack/README.txt
FINAL TAMPER RESISTANT PROOF PACK
---------------------------------

REGENERATION COMMAND:
bash scripts/final-proof-pack.sh

REQUIRED TESTS:
- tests/72-1-quotient-validity-audit.ts
- tests/73-rational-family-generalization.ts
- tests/74-1-a3-stabilization.ts
- tests/final-pointwise-proof.ts

ACCEPTANCE CRITERIA:
1. All extracted formulas are RAW VALID (no regularization hacks).
2. Pointwise differences on adversarial points must be exactly 0.000000.
3. No 'NaN', 'Infinity', or Removable Artifacts.
4. Git status must ideally be clean (if committed).
EOF

# 6. Hash and Tar
echo "Hashing Artifacts..."
sha256sum proof_pack/* > proof_pack/SHA256SUMS.txt
tar -czf proof_pack.tar.gz proof_pack

echo "=== PROOF PACK GENERATED SUCCESSFULLY: proof_pack.tar.gz ==="
