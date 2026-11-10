#!/bin/bash
set -e

cd "$(dirname "$0")/.."

OUT_DIR=".cache/y0/dafny/id_line5_extracted_py"

echo "=== Building Dafny ID Line 5 Extracted Runtime ==="

# Step 1: Verify Dafny module
echo "Step 1: Verifying src/dafny/id_line5_extracted.dfy..."
dafny verify src/dafny/id_line5_extracted.dfy --verification-time-limit:30 || {
    echo "ERROR: Dafny verification failed"
    exit 1
}

# Step 2: Translate to Python
echo "Step 2: Translating to Python..."
rm -rf src/dafny/id_line5_extracted-py* 2>/dev/null || true
dafny translate py src/dafny/id_line5_extracted.dfy --include-runtime --output "src/dafny/id_line5_extracted-py" || {
    echo "ERROR: Dafny translation failed"
    exit 1
}

TEMP_DIR="$(ls -d src/dafny/id_line5_extracted-py* 2>/dev/null | head -n 1)"
if [ -z "$TEMP_DIR" ]; then
    echo "ERROR: Could not find translated output directory"
    exit 1
fi

# Step 3: Move to cache
echo "Step 3: Moving to cache directory..."
mkdir -p "$(dirname "$OUT_DIR")"
rm -rf "$OUT_DIR"
mv "$TEMP_DIR" "$OUT_DIR"

# Step 4: Cleanup
echo "Step 4: Cleaning up temporary directories..."
rm -rf "src/dafny/id_line5_extracted-py" 2>/dev/null || true

echo "Done. Extracted module directory: $OUT_DIR"