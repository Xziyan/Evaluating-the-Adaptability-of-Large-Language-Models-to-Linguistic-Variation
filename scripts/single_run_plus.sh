#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./single_run_plus.sh PRED_XML_DIR GOLD_ANN_DIR OUT_DIR [--fuzzy 0.5] [--genre-map /path/map.csv]
#
# 1) Run the post_processing_pipeline.py to produce ANN in $OUT_DIR/final_ann
# 2) Runs eval_single_folder_plus.py and writes:
#    $OUT_DIR/eval_<mode>/{per_document.csv, per_genre.csv, overall.csv, matches/...}
#
# All stdout/stderr is tee’d to $OUT_DIR/run.log

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 PRED_XML_DIR GOLD_ANN_DIR OUT_DIR [--fuzzy 0.5] [--genre-map /path/map.csv]"
  exit 1
fi

PRED_XML_DIR="$1"; shift
GOLD_DIR="$1"; shift
OUT_DIR="$1"; shift

export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run.log"

echo "=== single_run_plus.sh ===" | tee "$LOG"
echo "PRED_XML_DIR: $PRED_XML_DIR" | tee -a "$LOG"
echo "GOLD_DIR     : $GOLD_DIR"     | tee -a "$LOG"
echo "OUT_DIR      : $OUT_DIR"      | tee -a "$LOG"
echo "FLAGS        : $*"            | tee -a "$LOG"

echo "[1/3] Post-processing XML -> ANN ..." | tee -a "$LOG"
python post_processing_pipeline.py \
  --input_xml_dir "$PRED_XML_DIR" \
  --text_dir "$GOLD_DIR" \
  --output_root "$OUT_DIR" \
  --do_alignement True 2>&1 | tee -a "$LOG"

FINAL_ANN_DIR="$OUT_DIR/final_ann"
if [[ ! -d "$FINAL_ANN_DIR" ]]; then
  echo "ERROR: final ANN folder not found: $FINAL_ANN_DIR" | tee -a "$LOG"
  exit 2
fi

# Count ann files
PRED_COUNT=$(ls "$FINAL_ANN_DIR"/*.ann 2>/dev/null | wc -l | tr -d ' ')
GOLD_COUNT=$(ls "$GOLD_DIR"/*.ann 2>/dev/null | wc -l | tr -d ' ')
echo "final_ann *.ann: $PRED_COUNT" | tee -a "$LOG"
echo "gold_dir *.ann : $GOLD_COUNT" | tee -a "$LOG"

if [[ "$PRED_COUNT" == "0" ]]; then
  echo "ERROR: No .ann files in $FINAL_ANN_DIR — evaluator would be empty." | tee -a "$LOG"
  exit 3
fi
if [[ "$GOLD_COUNT" == "0" ]]; then
  echo "ERROR: No .ann files in $GOLD_DIR — pass the GOLD_ANN_DIR (not the raw text dir)." | tee -a "$LOG"
  exit 4
fi

# Determine eval subfolder label
MODE_LABEL="strict"
if printf '%s\0' "$@" | grep -q -- "--fuzzy"; then
  FUZZY_THR=""
  ARGS=("$@")
  for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ "${ARGS[$i]}" == "--fuzzy" && $((i+1)) -lt ${#ARGS[@]} ]]; then
      FUZZY_THR="${ARGS[$i+1]}"
      break
    fi
  done
  MODE_LABEL=${FUZZY_THR:+fuzzy_${FUZZY_THR}}
  MODE_LABEL=${MODE_LABEL:-fuzzy}
fi

EVAL_DIR="$OUT_DIR/eval_${MODE_LABEL}"
mkdir -p "$EVAL_DIR"

echo "[2/3] Evaluating ($MODE_LABEL) -> $EVAL_DIR ..." | tee -a "$LOG"
python eval_single_folder_plus.py \
  --pred "$FINAL_ANN_DIR" \
  --gold "$GOLD_DIR" \
  --out  "$EVAL_DIR" \
  --write-matches \
  "$@" 2>&1 | tee -a "$LOG"

# Verify outputs exist
echo "[3/3] Verifying evaluation outputs ..." | tee -a "$LOG"
if [[ ! -s "$EVAL_DIR/per_document.csv" ]]; then
  echo "ERROR: per_document.csv not created. Check $LOG for 'No .ann files found' or Python errors." | tee -a "$LOG"
  exit 5
fi
if [[ ! -d "$EVAL_DIR/matches" ]]; then
  echo "WARN: no matches directory found (no files evaluated?), check $LOG." | tee -a "$LOG"
else
  # List one example doc folder if any
  ONE_DOC=$(ls -1 "$EVAL_DIR/matches" 2>/dev/null | head -n1 || true)
  if [[ -n "$ONE_DOC" ]]; then
    echo "Example matches: $EVAL_DIR/matches/$ONE_DOC/{tp.csv,fp.csv,fn.csv,diff.tsv}" | tee -a "$LOG"
  fi
fi

echo "DONE. See:" | tee -a "$LOG"
echo " - Predictions (ANN): $FINAL_ANN_DIR" | tee -a "$LOG"
echo " - Eval outputs     : $EVAL_DIR"      | tee -a "$LOG"
echo " - Log              : $LOG"           | tee -a "$LOG"
