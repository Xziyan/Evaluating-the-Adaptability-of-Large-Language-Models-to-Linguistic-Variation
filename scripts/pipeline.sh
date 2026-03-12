#!/bin/bash
# This script is used to run the LLM APIs pipeline
# It includes steps for aligning XML spaces, post-processing, and evaluating results
# Ensure the script is executable
# chmod +x llm_apis/pipeline.sh


cd /Users/ziyanxu/AGLAGLA/llm_apis

# Step 1 ; align XML spaces

python post_processing_pipeline.py --input_xml_dir //Users/ziyanxu/AGLAGLA/output_gml_xml/OUT_COT_DeepSeekR1 --text_dir /Users/ziyanxu/AGLAGLA/dataog --output_root /Users/ziyanxu/AGLAGLA/OUT_DIR_INTERMEDIATE --do_alignement True 


# Step 2 ; eval results
/Users/ziyanxu/AGLAGLA/.venv/bin/python /Users/ziyanxu/AGLAGLA/llm_apis/eva_xml_results.py --fuzzy

# Step 3 ; generate heatmap
/Users/ziyanxu/AGLAGLA/.venv/bin/python /Users/ziyanxu/AGLAGLA/llm_apis/heatmap_eval.py