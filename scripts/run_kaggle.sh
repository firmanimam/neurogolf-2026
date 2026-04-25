#!/usr/bin/env bash
# Push notebook to Kaggle GPU, wait for it to finish, pull ONNX output back.
# Run from the neurogolf-2026/ directory: bash scripts/run_kaggle.sh

set -e

KERNEL_ID="piririp/neurogolf-2026-baseline"
NOTEBOOKS_DIR="$(dirname "$0")/../notebooks"
OUTPUT_DIR="$(dirname "$0")/../output/onnx"

mkdir -p "$OUTPUT_DIR"

# ── 1. Push ──────────────────────────────────────────────────────────────────
echo "==> Pushing notebook to Kaggle..."
kaggle kernels push -p "$NOTEBOOKS_DIR"
echo "    Kernel submitted: $KERNEL_ID"
echo "    View live logs: https://www.kaggle.com/code/$KERNEL_ID"

# ── 2. Poll until done ───────────────────────────────────────────────────────
echo ""
echo "==> Waiting for GPU training to complete (checks every 60s)..."
echo "    This usually takes 20–60 min for 400 tasks."
echo ""

while true; do
    STATUS=$(kaggle kernels status "$KERNEL_ID" 2>/dev/null \
        | tail -1 | awk '{print $NF}')

    TIMESTAMP=$(date '+%H:%M:%S')
    echo "    [$TIMESTAMP] status: $STATUS"

    case "$STATUS" in
        complete)
            echo ""
            echo "==> Kernel finished successfully!"
            break
            ;;
        error|failed|cancelAcknowledged|cancel*)
            echo ""
            echo "ERROR: Kernel ended with status: $STATUS"
            echo "Check logs at: https://www.kaggle.com/code/$KERNEL_ID"
            exit 1
            ;;
        running|queued|*)
            sleep 60
            ;;
    esac
done

# ── 3. Pull output ───────────────────────────────────────────────────────────
echo "==> Downloading ONNX output files..."
kaggle kernels output "$KERNEL_ID" -p "$OUTPUT_DIR"

# The output comes as a zip — unzip if present
if ls "$OUTPUT_DIR"/*.zip 1>/dev/null 2>&1; then
    echo "==> Unzipping output..."
    for zipfile in "$OUTPUT_DIR"/*.zip; do
        unzip -o "$zipfile" -d "$OUTPUT_DIR"
        rm "$zipfile"
    done
fi

echo ""
echo "==> Done! ONNX files saved to: $OUTPUT_DIR"
echo "    Files:"
ls "$OUTPUT_DIR"/*.onnx 2>/dev/null | wc -l | xargs -I{} echo "    {} .onnx files"
