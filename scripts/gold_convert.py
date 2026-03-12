import os
import json
from collections import defaultdict

# Constants
TOP_LEVEL_TAGS = ["PERS", "LOC", "PROD", "ORG", "EVENT", "TIME"]

def normalize_tag(tag):
    """Extract top-level category (e.g., loc.adm.town → LOC)"""
    prefix = tag.split('.')[0].upper()
    return prefix if prefix in TOP_LEVEL_TAGS else None

def convert_ann_file(ann_path):
    """Convert one .ann file into LLM-style JSON dict"""
    entities_by_type = {tag: [] for tag in TOP_LEVEL_TAGS}

    with open(ann_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.startswith('T'):
                continue
            try:
                parts = line.strip().split('\t')
                tag_info = parts[1]
                tag_parts = tag_info.split()
                tag = tag_parts[0]
                start = int(tag_parts[1])
                end = int(tag_parts[2])
                text = parts[2]

                top_level_tag = normalize_tag(tag)
                if top_level_tag:
                    entities_by_type[top_level_tag].append({
                        "entity": text,
                        "start": start,
                        "end": end
                    })
            except Exception as e:
                print(f"Error in {ann_path}: {e}")
                continue

    return entities_by_type

def batch_convert_ann_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".ann"):
            ann_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace(".ann", ".json"))

            result = convert_ann_file(ann_path)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"✅ Converted {filename} → {os.path.basename(output_path)}")

if __name__ == "__main__":
    input_dir = "/Users/ziyanxu/AGLAGLA/dataog"
    output_dir = "/Users/ziyanxu/AGLAGLA/converted_gold_formatjson"
    batch_convert_ann_folder(input_dir, output_dir)
