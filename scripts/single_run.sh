#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./single_run.sh /path/to/pred_xml_dir /path/to/gold_ann_dir /path/to/out_dir [--fuzzy 0.5] [--genre-map /path/map.csv]
# /Users/ziyanxu/AGLAGLA/llm_apis/single_run.sh \                                                   
# /Users/ziyanxu/AGLAGLA/output_gml_xml/xml_output_FewshotADOPTED_balise_DeepSeek_r1 \
# /Users/ziyanxu/AGLAGLA/dataog \
# /Users/ziyanxu/AGLAGLA/output_gml_xml/OUT_DIR ### --fuzzy 0.5 ### with this fuzzy 0.5 mode, it does fuzzy matching
# Steps:
# 1) Post-process XML -> aligned XML -> final ANN (under OUT_DIR/final_ann)
# 2) Evaluate final ANN vs gold ANN (strict or fuzzy)

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 PRED_XML_DIR GOLD_ANN_DIR OUT_DIR [--fuzzy 0.5] [--genre-map /path/map.csv]"
  exit 1
fi

PRED_XML_DIR="$1"; shift
GOLD_DIR="$1"; shift
OUT_DIR="$1"; shift

mkdir -p "$OUT_DIR"

echo "[1/2] Post-processing XML -> ANN under $OUT_DIR"
python post_processing_pipeline.py \
  --input_xml_dir "$PRED_XML_DIR" \
  --text_dir "$GOLD_DIR" \
  --output_root "$OUT_DIR" \
  --do_alignement True
 
FINAL_ANN_DIR="$OUT_DIR/final_ann"
if [[ ! -d "$FINAL_ANN_DIR" ]]; then
  echo "ERROR: final ANN folder not found: $FINAL_ANN_DIR"
  exit 2
fi

echo "[2/2] Evaluation"
# Forward remaining flags (e.g., --fuzzy 0.5 --genre-map map.csv)
python eval_single_folder.py \
  --pred "$FINAL_ANN_DIR" \
  --gold "$GOLD_DIR" \
  --out  "$OUT_DIR" \
  "$@"

echo "Done. See CSVs in $OUT_DIR:"
ls -1 "$OUT_DIR"/per_*.csv "$OUT_DIR"/overall.csv 2>/dev/null || true
