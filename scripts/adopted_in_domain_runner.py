
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
#python adopted_in_domain_runner.py \                                                      
#  --input /Users/ziyanxu/AGLAGLA/dataog \
#  --examples /Users/ziyanxu/AGLAGLA/llm_apis/example_prompt.yaml \
#  --prompts-module /Users/ziyanxu/AGLAGLA/llm_apis/utils.py \
#  --model-func-module /Users/ziyanxu/AGLAGLA/llm_apis/model_api.py \
#  --model-func-name call_model
# Where:
# - prompts.py must define ZEROSHOT_PROMPT_BALISE (str)
# - model_api.py must define a function with signature:
#     def call_model(prompt: str, text: str) -> str:
#         # returns XML as string (already <root>...</root>)

import argparse
import importlib.util
import sys
import yaml
import re
import os
from pathlib import Path

OUTPUT_DIR = "/Users/ziyanxu/AGLAGLA/output_gml_xml/output_deepseek_r1_adopted_dir"

SUPPORTED_GENRES = {"poetry","prose","encyclopedia","information","spoken"}  # multi excluded

def infer_genre_from_filename(name: str) -> str:
    stem = Path(name).stem.lower()
    # prefix until first non-letter
    m = re.match(r"^([a-zA-Z]+)", stem)
    if m:
        return m.group(1)
    return stem.split("-")[0].split("_")[0]

def load_module_from_path(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)  # type: ignore
    return module

def load_examples(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def select_example_block(examples_dict, genre: str) -> str:
    arr = examples_dict.get(genre, [])
    if not arr:
        return ""
    # pick the first example
    ex = arr[0]
    lines = []
    lines.append("Here is one in-domain example:")
    lines.append("Input:")
    lines.append(f"\"{ex['input']}\"")
    lines.append("Output:")
    lines.append(ex["output"].strip())
    return "\n".join(lines)

def compose_prompt(base_prompt: str, example_block: str) -> str:
    if not example_block:
        return base_prompt
    # 在“Here is the input text:”之前插入示例块
    #这里改句子了
    marker = "Now process the following input. Think step by step, but at the very end return ONLY the valid XML:"
    if marker in base_prompt:
        return base_prompt.replace(marker, f"{example_block}\n\n{marker}")
    # 兜底：直接附加在末尾
    return f"{base_prompt}\n\n{example_block}\n"


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def main():
    ap = argparse.ArgumentParser(description="Adopted per-genre prompting runner (base prompt + in-domain example).")
    ap.add_argument("--input", required=True, help="Path to a .txt file or a folder of .txt files")
    ap.add_argument("--examples", required=True, help="Path to example_prompt.yaml")
    ap.add_argument("--prompts-module", required=True, help="Path to Python module that defines ZEROSHOT_PROMPT_BALISE")
    ap.add_argument("--model-func-module", required=True, help="Path to Python module that defines the model call function")
    ap.add_argument("--model-func-name", default="call_model", help="Function name to call inside model-func-module")
    args = ap.parse_args()

    prompts_mod = load_module_from_path(args.prompts_module, "prompts_mod")
    #base_prompt = getattr(prompts_mod, "ZEROSHOT_PROMPT_BALISE")
    base_prompt = getattr(prompts_mod, "COT_ZEROSHOT") #the cot version
    model_mod = load_module_from_path(args.model_func_module, "model_mod")
    model_func = getattr(model_mod, args.model_func_name)

    examples_dict = load_examples(args.examples)
    ensure_outdir(OUTPUT_DIR)

    input_path = Path(args.input)
    if input_path.is_dir():
        txt_files = sorted(input_path.glob("*.txt"))
    else:
        txt_files = [input_path]

    for txt in txt_files:
        stem = txt.stem
        genre = infer_genre_from_filename(txt.name)
        if genre == "multi" or stem.startswith("multi"):
            print(f"[SKIP] {txt.name} (multi genre)")
            continue
        if genre not in SUPPORTED_GENRES:
            print(f"[WARN] {txt.name}: inferred genre '{genre}' not in {SUPPORTED_GENRES}; using base prompt only.")
            example_block = ""
        else:
            example_block = select_example_block(examples_dict, genre)

        final_prompt = compose_prompt(base_prompt, example_block)
        text_content = txt.read_text(encoding="utf-8", errors="ignore")

        # Call user's model function
        try:
            xml_out = model_func(final_prompt, text_content)
        except TypeError:
            # fallback for kwargs variants
            xml_out = model_func(final_prompt, text=text_content)

        out_path = Path(OUTPUT_DIR) / f"{stem}.xml"
        out_path.write_text(xml_out, encoding="utf-8")
        print(f"[OK] {txt.name} -> {out_path}")

if __name__ == "__main__":
    main()
