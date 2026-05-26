#!/bin/bash
set -e

# LOCAL CLEAN PROOF CLOSURE SCRIPT
# Goal: Regenerate proof from a clean local copy.

ORIG_DIR="/home/kugli/Downloads/Genetic-Code-Lab/genetic-code-lab"
CLEAN_DIR="/home/kugli/Downloads/Genetic-Code-Lab/genetic-code-lab-proof"

echo "=== 1. PREPARING CLEAN PROOF FOLDER ==="
rm -rf "$CLEAN_DIR"
mkdir -p "$CLEAN_DIR"

echo "=== 2. COPYING PROJECT ==="
# Copy everything excluding node_modules (will copy separately to save time if needed, or just copy all)
# User said "clean local copy", usually implies fresh npm install, but they also said "copy project".
# To be safe and fast, I'll copy and then run npm install if needed, but the user didn't specify npm install.
# The previous script copied node_modules. I'll do the same to ensure it works.
rsync -a --exclude='.git' --exclude='proof_pack*' --exclude='node_modules' "$ORIG_DIR/" "$CLEAN_DIR/"
if [ -d "$ORIG_DIR/node_modules" ]; then
    echo "Copying node_modules from original to save time..."
    cp -r "$ORIG_DIR/node_modules" "$CLEAN_DIR/"
fi

cd "$CLEAN_DIR"
mkdir -p proof_pack

echo "=== 3. SAVING ENVIRONMENT INFO ==="
{
    echo "DATE: $(date)"
    echo "PWD: $(pwd)"
    echo "ORIGINAL WORKING COPY: $ORIG_DIR"
    echo "CLEAN PROOF COPY: $CLEAN_DIR"
    echo "NODE: $(node -v)"
    echo "NPM: $(npm -v)"
    echo "TS-NODE: $(npx ts-node -v 2>/dev/null || echo 'not installed')"
    echo "UNAME: $(uname -a)"
    echo "CPU CORES: $(nproc)"
    echo "TOTAL RAM: $(free -m | grep Mem | awk '{print $2}') MB"
} > proof_pack/environment.txt

echo "=== 4. RUNNING FORENSIC TESTS ==="
npm run test:forensic > proof_pack/forensic.txt 2>&1 || echo "test:forensic failed or not found" >> proof_pack/forensic.txt

echo "=== 5. RUNNING AUDIT TESTS ==="
echo "Running 72-1..."
npx ts-node tests/72-1-quotient-validity-audit.ts > proof_pack/72_1_validity.txt 2>&1 || echo "72-1 failed" >> proof_pack/72_1_validity.txt
echo "Running 73..."
npx ts-node tests/73-rational-family-generalization.ts > proof_pack/73_family.txt 2>&1 || echo "73 failed" >> proof_pack/73_family.txt
echo "Running 74-1..."
NODE_OPTIONS=--max-old-space-size=8192 npx ts-node tests/74-1-a3-stabilization.ts > proof_pack/74_1_a3.txt 2>&1 || echo "74-1 failed" >> proof_pack/74_1_a3.txt

echo "=== 6. EXTRACTING FULL FORMULAS ==="
{
    echo "=== P1-P5 FORMULAS ==="
    grep " | P" proof_pack/73_family.txt | awk -F'\`' '{print $2}' || true
    echo ""
    echo "=== A3 FORMULAS ==="
    grep " | A3 " proof_pack/74_1_a3.txt | awk -F'\`' '{print $2}' || true
} > proof_pack/final_formulas.txt

echo "=== 7. RUNNING POINTWISE PROOF ==="
npx ts-node tests/final-pointwise-proof.ts > proof_pack/pointwise_proof.txt 2>&1 || echo "pointwise proof failed" >> proof_pack/pointwise_proof.txt

echo "=== 8. GENERATING LOCAL DIFF REPORT ==="
# User asked for: find oba foldera, uporedi fajlove, sačuvaj listu razlika
echo "Comparing $ORIG_DIR and $CLEAN_DIR" > proof_pack/local_diff_report.txt
diff -rq --exclude=node_modules --exclude=proof_pack --exclude=.git --exclude='*.log' "$ORIG_DIR" "$CLEAN_DIR" >> proof_pack/local_diff_report.txt || true

echo "=== 9. HASHING ARTIFACTS ==="
cd proof_pack
sha256sum * > SHA256SUMS.txt
cd ..

echo "=== 10. PACKAGING ==="
tar -czf proof_pack_local.tar.gz proof_pack
cp proof_pack_local.tar.gz "$ORIG_DIR/"

echo "=== DONE ==="
echo "Proof pack generated at: $ORIG_DIR/proof_pack_local.tar.gz"
