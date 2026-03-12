import os
import re
import shutil
import subprocess
import html
import sys
from lxml import etree


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def is_xml_candidate(filename: str) -> bool:
    low = filename.lower()
    return low.endswith(".xml") and ".raw" not in low and "only_root" not in low

def replace_root(path, old_root, new_root):
    rel = os.path.relpath(path, old_root)
    return os.path.join(new_root, rel)

# --------- cleaning functions (conservative) ----------
def normalize_text_node(t: str) -> str:
    if t is None:
        return t
    t = html.unescape(t)  # 把 &amp; 变成 &
    t = re.sub(r"[ \t]{2,}", " ", t)  # 多空格合并
    t = re.sub(r"\s+([,;:?!.])", r"\1", t)  # 标点前不要空格
    t = re.sub(r"\s+'\s*", "'", t)  # L 'a -> L'
    t = re.sub(r"\s+'", "'", t)
    t = re.sub(r"'\s+", "'", t)
    return t

def clean_xml_preserve_word_structure(raw_xml_str: str) -> str:
    raw_xml_str = raw_xml_str.replace('<sup>','').replace('</sup>', '')  # 移除 <sup> 标签
    raw_xml_str = raw_xml_str.replace('&', 'K')  
    return raw_xml_str
    parser = etree.XMLParser(recover=True, encoding='utf-8')
    try:
        root = etree.fromstring(raw_xml_str.encode('utf-8'), parser=parser)
    except Exception:
        # recover=True 通常不会抛出，但以防万一 fallback
        try:
            root = etree.fromstring(raw_xml_str.encode('utf-8'), parser=parser)
        except Exception:
            # 无法解析为 XML，退回最小化处理：只做 text-level 规范化，不破坏内容
            t = normalize_text_node(raw_xml_str)
            return t

    for elem in root.iter():
        if elem.text:
            elem.text = normalize_text_node(elem.text)
        if elem.tail:
            elem.tail = normalize_text_node(elem.tail)

    # flatten <sup> tags safely
    for sup in root.xpath("//sup"):
        parent = sup.getparent()
        if sup.text and parent is not None:
            insertion = sup.text
            tail = sup.tail or ""
            idx = parent.index(sup)
            parent.remove(sup)
            if idx == 0:
                parent.text = (parent.text or "") + insertion
            else:
                prev = parent[idx - 1]
                prev.tail = (prev.tail or "") + insertion
            if tail:
                if idx == 0:
                    parent.text += tail
                else:
                    prev.tail += tail

    # 序列化（lxml 会自动对 & 进行正确转义成 &amp; 保持结构合法）
    try:
        cleaned = etree.tostring(root, encoding='utf-8', xml_declaration=False).decode('utf-8')
    except Exception:
        # fallback to raw normalized text if serialization fails
        cleaned = raw_xml_str
    return cleaned

# --------- pipeline steps ----------
def step1_clean_xml(input_xml_root, safe_xml_root):
    print("Step 1: Conservative cleaning of original XML -> safe_xml/")
    for root, _, files in os.walk(input_xml_root):
        for file in files:
            if not is_xml_candidate(file):
                continue
            in_path = os.path.join(root, file)
            try:
                with open(in_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw = f.read()
            except Exception as e:
                print(f"[ERROR] reading {in_path}: {e}")
                continue
            cleaned = clean_xml_preserve_word_structure(raw)
            out_path = replace_root(in_path, input_xml_root, safe_xml_root)
            ensure_dir(os.path.dirname(out_path))
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            print(f"[CLEANED] {in_path} -> {out_path}")

def step2_extract_initial_ann(safe_xml_root, ann_output_raw_root):
    print("Step 2: Generating initial .ann from cleaned XML (for debug/comparison)")
    for root, _, files in os.walk(safe_xml_root):
        for file in files:
            if not is_xml_candidate(file):
                continue
            xml_path = os.path.join(root, file)
            rel = os.path.relpath(xml_path, safe_xml_root)
            target_dir = os.path.join(ann_output_raw_root, os.path.dirname(rel))
            ensure_dir(target_dir)
            try:
                subprocess.run(
                    ['python', 'ann_xmlconvert_new.py', xml_path, 'ann', '--out-dir', target_dir],
                    check=True,
                    capture_output=False,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] xmlconvert initial ann failed for {xml_path}: {e}")

def step3_align_spaces(safe_xml_root, gold_txt_folder, aligned_xml_root, do_alignement=True):
    print("Step 3: Aligning cleaned XML to gold .txt using smart align")
    ensure_dir(aligned_xml_root)
    try:
        subprocess.run(
            ['python', 'align_xml_spaces_final_version_hopefully.py', gold_txt_folder, safe_xml_root, aligned_xml_root, str(do_alignement)],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] align_xml_spaces_final_version_hopefully.py failed: {e}")
        sys.exit(1)

def step4_pre_final_clean(aligned_xml_root, pre_final_clean_root):
    print("Step 4: Cleaning aligned XML (preserve structure) before final conversion")
    for root, _, files in os.walk(aligned_xml_root):
        for file in files:
            if not is_xml_candidate(file):
                continue
            in_path = os.path.join(root, file)
            try:
                with open(in_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw = f.read()
            except Exception as e:
                print(f"[ERROR] reading aligned xml {in_path}: {e}")
                continue
            cleaned = clean_xml_preserve_word_structure(raw)
            out_path = replace_root(in_path, aligned_xml_root, pre_final_clean_root)
            ensure_dir(os.path.dirname(out_path))
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            print(f"[PRE-CLEANED] {in_path} -> {out_path}")

def step5_final_ann(pre_final_clean_root, final_ann_root):
    print("Step 5: Generating final .ann from pre-cleaned aligned XML")
    for root, _, files in os.walk(pre_final_clean_root):
        for file in files:
            if not is_xml_candidate(file):
                continue
            xml_path = os.path.join(root, file)
            rel = os.path.relpath(xml_path, pre_final_clean_root)
            target_dir = os.path.join(final_ann_root, os.path.dirname(rel))
            ensure_dir(target_dir)
            try:
                subprocess.run(
                    ['python', 'ann_xmlconvert_new.py', xml_path, 'ann', '--out-dir', target_dir],
                    check=True,
                    capture_output=True,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] xmlconvert final ann failed for {xml_path}: {e}")

def run_pipeline(input_xml_dir, gold_txt_dir, output_root, do_alignement):
    safe_xml = os.path.join(output_root, 'safe_xml')
    ann_output_raw = os.path.join(output_root, 'ann_output_raw')
    aligned_xml = os.path.join(output_root, 'aligned_xml')
    pre_final_clean = os.path.join(output_root, 'aligned_xml_pre_final_clean')
    final_ann = os.path.join(output_root, 'final_ann')

    #step1_clean_xml(input_xml_dir, safe_xml)
    #step2_extract_initial_ann(safe_xml, ann_output_raw)
    step3_align_spaces(input_xml_dir, gold_txt_dir, aligned_xml, do_alignement=do_alignement)

    #step4_pre_final_clean(input_xml_dir, pre_final_clean)
    step5_final_ann(aligned_xml, final_ann)

    print("✅ Full pipeline complete.")
    print(f"Outputs under: {output_root}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Full post-processing pipeline: clean -> initial ann -> align -> final ann")
    parser.add_argument('--input_xml_dir', required=True, help='Root of raw LLM XML outputs (possibly nested)')
    parser.add_argument('--text_dir', required=True, help='Flat gold folder containing .txt/.ann pairs')
    parser.add_argument('--output_root', required=True, help='Where to place all intermediate and final outputs')
    parser.add_argument('--do_alignement', help='Whether to perform alignment step', default=True)
    args = parser.parse_args()

    run_pipeline(args.input_xml_dir, args.text_dir, args.output_root, args.do_alignement)
