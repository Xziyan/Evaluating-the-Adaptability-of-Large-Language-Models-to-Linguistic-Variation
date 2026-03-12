import os
import sys
import re

def is_xml_candidate(fname: str) -> bool:
    low = fname.lower()
    return low.endswith(".xml") and ".raw" not in low and "only_root" not in low

def fix_tokenization_artifacts(text: str) -> str:
    # L 'aér -> L'aér
    text = re.sub(r"\s+'\s*", "'", text)
    text = re.sub(r"\s+'", "'", text)
    text = re.sub(r"'\s+", "'", text)
    # collapse letter splits like U k r a i n e -> Ukraine (保守：只有 2+ 单字母间空格连续)
    text = re.sub(r'\b([A-Za-z])\s+([A-Za-z])\b', lambda m: m.group(1) + m.group(2), text)
    # 标点前删空格
    text = re.sub(r'\s+([,;:?!.])', r'\1', text)
    # 多空格合并
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text

def process_file(gold_dir, in_path, out_path):
    try:
        with open(in_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"[ERROR] reading {in_path}: {e}")
        return

    # 只在标签外做替换：用简单状态机跳过 <...>
    result_chars = []
    in_tag = False
    buffer = []
    for c in content:
        if c == '<':
            if buffer:
                segment = ''.join(buffer)
                result_chars.append(fix_tokenization_artifacts(segment))
                buffer = []
            in_tag = True
            result_chars.append(c)
        elif c == '>':
            in_tag = False
            result_chars.append(c)
        else:
            if in_tag:
                result_chars.append(c)
            else:
                buffer.append(c)
    if buffer:
        result_chars.append(fix_tokenization_artifacts(''.join(buffer)))

    fixed = ''.join(result_chars)
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(fixed)
        print(f"[ALIGNED_SIMPLE] {in_path} -> {out_path}")
    except Exception as e:
        print(f"[ERROR] writing aligned simple xml {out_path}: {e}")

def batch_align(gold_txt_folder, xml_root_dir, output_root):
    for root, dirs, files in os.walk(xml_root_dir):
        rel = os.path.relpath(root, xml_root_dir)
        target_dir = os.path.join(output_root, rel)
        os.makedirs(target_dir, exist_ok=True)
        for file in files:
            if not is_xml_candidate(file):
                continue
            in_path = os.path.join(root, file)
            out_path = os.path.join(target_dir, file)
            process_file(gold_txt_folder, in_path, out_path)

def usage():
    print("Usage: python align_xml_spaces_simple.py <gold_txt_folder_unused> <xml_root_dir> <output_root>")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        usage()
        sys.exit(1)
    _, gold_folder, xml_root_dir, output_root = sys.argv
    batch_align(gold_folder, xml_root_dir, output_root)
