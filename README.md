
## Adaptability of Large Language Models (LLMs) to Attested Linguistic Genres 
This repository accompanies a research project aimed at evaluating the extent to which large language models (LLMs) adapt to linguistic variations in named entity recognition (NER) tasks for french.
It is the official repository for the paper **Adaptability of Large Language Models (LLMs) to Attested Linguistic Genres** (LREC 2026) 

## Main features

1. **Scripts for prompting LLM on a NER corpus with customized prompts**
2. **Scripts for processing the LLM output and evaluating with fuzzy matching**
3. **Scripts for analysing the result and study of the impact of variations in wording and domain on LLM performance**

## Repository structure
- `data/`: NEM.fr source texts and annotation files; should be downloaded from the official [NEM.fr repository](https://github.com/ayusekyo111/NEM.fr)
- `scripts/`: Main scripts 

## How to :
### Prompt LLMs
Run `scripts/adopted_in_domain_runner.py`. Documentation and usage of the script :
```python
# Adopted per-genre prompting runner.
# - Keeps base prompt ZEROSHOT_PROMPT_BALISE unchanged.
# - For poetry/prose/encyclopedia/information/spoken files, appends the matching example from example_prompt.yaml.
# - Skips files whose stem starts with 'multi'.
# - Infers genre from filename like 'prose01-Voltaire.txt' -> 'prose'.
# - Writes predictions to /Users/ziyanxu/AGLAGLA/output_gml_xml/output_adopted_dir/<stem>.xml
#
# USAGE
# -----
# python adopted_in_domain_runner.py \
#   --input /path/to/text_or_folder \
#   --examples /path/to/example_prompt.yaml \
#   --prompts-module /path/to/prompts.py \
#   --model-func-module /path/to/model_api.py \
#   --model-func-name call_model
# -----
# Where:
# - prompts.py must define ZEROSHOT_PROMPT_BALISE (str)
# - model_api.py must define a function with signature:
#     def call_model(prompt: str, text: str) -> str:
#         # returns XML as string (already <root>...</root>)
```

### Post process results :
Run `post_processing_pipeline.py`:

```bash
post_processing_pipeline.py [-h] --input_xml_dir INPUT_XML_DIR --text_dir TEXT_DIR --output_root OUTPUT_ROOT [--do_alignement DO_ALIGNEMENT]

options:
  -h, --help            show this help message and exit
  --input_xml_dir INPUT_XML_DIR
                        Root of raw LLM XML outputs (possibly nested)
  --text_dir TEXT_DIR   Flat gold folder containing .txt/.ann pairs
  --output_root OUTPUT_ROOT
                        Where to place all intermediate and final outputs
  --do_alignement DO_ALIGNEMENT
                        Whether to perform alignment step
```

This performs xml to ann conversion, and possibly alignment (if --do_alignement is set) to produce aligned .ann files in the output_root directory, ready for evaluation.

### Post process + Analyse results :

Run `single_run_plus.sh PRED_XML_DIR GOLD_ANN_DIR OUT_DIR [--fuzzy 0.5] [--genre-map /path/map.csv]`
where PRED_XML_DIR_GOLD is the prediction (xml format) data path, GOLD_ANN_DIR is the gold data path (ann format), OUT_DIR is the output directory for all intermediate and final outputs, --fuzzy is the fuzzy matching threshold (default 0.5), and --genre-map is an optional csv file mapping file stems to genres for per-genre analysis.
