#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DAFNY_FILE="$ROOT_DIR/src/dafny/id_line1_extracted.dfy"
OUT_DIR="$ROOT_DIR/.cache/y0/dafny/id_line1_extracted_py"
TRANSLATED_DIR="$ROOT_DIR/src/dafny/id_line1_extracted-py"

echo "[1/3] Verifying $DAFNY_FILE"
dafny verify "$DAFNY_FILE" --verification-time-limit:30 --isolate-assertions --progress:Batch

echo "[2/3] Translating to Python"
dafny translate py "$DAFNY_FILE" --include-runtime --verbose

if [[ ! -d "$TRANSLATED_DIR" ]]; then
  echo "Expected translated directory not found: $TRANSLATED_DIR" >&2
  exit 1
fi

echo "[3/3] Installing extracted artifacts into $OUT_DIR"
rm -rf "$OUT_DIR"
mkdir -p "$(dirname "$OUT_DIR")"
cp -R "$TRANSLATED_DIR" "$OUT_DIR"
rm -rf "$TRANSLATED_DIR"

echo "Done. Extracted module directory: $OUT_DIR"
