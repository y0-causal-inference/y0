#!/bin/bash
set -e

cd "$(dirname "$0")/.."

OUT_DIR=".cache/y0/dafny/id_tian2002_extracted_py"

echo "=== Building Dafny Tian 2002 Extracted Runtime ==="

echo "Step 1: Verifying src/dafny/id_tian2002_extracted.dfy..."
dafny verify src/dafny/id_tian2002_extracted.dfy --verification-time-limit:30 || {
    echo "ERROR: Dafny verification failed"
    exit 1
}

echo "Step 2: Translating to Python..."
rm -rf src/dafny/id_tian2002_extracted-py* 2>/dev/null || true
dafny translate py src/dafny/id_tian2002_extracted.dfy --include-runtime --output "src/dafny/id_tian2002_extracted-py" || {
    echo "ERROR: Dafny translation failed"
    exit 1
}

TEMP_DIR="$(ls -d src/dafny/id_tian2002_extracted-py* 2>/dev/null | head -n 1)"
if [ -z "$TEMP_DIR" ]; then
    echo "ERROR: Could not find translated output directory"
    exit 1
fi

echo "Step 3: Moving to cache directory..."
mkdir -p "$(dirname "$OUT_DIR")"
rm -rf "$OUT_DIR"
mv "$TEMP_DIR" "$OUT_DIR"

echo "Step 4: Cleaning up temporary directories..."
rm -rf "src/dafny/id_tian2002_extracted-py" 2>/dev/null || true

REPORT_PATH=".cache/y0/dafny/tian2002_tex_case_comparison.json"

echo "Step 5: Running tex-case comparison report..."
python scripts/compare_dafny_tian2002_tex_cases.py --output "$REPORT_PATH" || {
    echo "ERROR: Tex-case comparison failed"
    exit 1
}

echo "Done. Extracted module directory: $OUT_DIR"
echo "Comparison report: $REPORT_PATH"
