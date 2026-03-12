#!/usr/bin/env python3
import os
import sys
import re
import argparse
#坏坏脚本，开了窗口做了fuzzy match

def load_ann(ann_path):
    """
    读取 .ann 文件，支持 tab 或空格分隔格式，跳过格式不对的行。
    返回实体列表：每项包含 ent_id, type, start, end, text。
    """
    entries = []
    # 正则：可选 T#, 必须 type start end text
    pattern = re.compile(r"^(T\d+)\s+([A-Z]+)\s+(\d+)\s+(\d+)\s+(.+)$")
    pattern2 = re.compile(r"^([A-Z]+)\s+(\d+)\s+(\d+)\s+(.+)$")
    with open(ann_path, encoding='utf-8') as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip()
            if not line:
                continue
            m = pattern.match(line)
            if m:
                ent_id, ent_type, start, end, text = m.groups()
            else:
                m2 = pattern2.match(line)
                if m2:
                    ent_id = None
                    ent_type, start, end, text = m2.groups()
                else:
                    print(f"[WARN] Skipping malformed line {lineno} in {ann_path}: '{line}'")
                    continue
            try:
                start = int(start)
                end = int(end)
            except ValueError:
                print(f"[WARN] Invalid offsets on line {lineno} in {ann_path}: '{line}'")
                continue
            entries.append({
                'ent_id': ent_id,
                'type': ent_type,
                'start': start,
                'end': end,
                'text': text
            })
    return entries


def find_all_positions(text, sub):
    """返回 text 中所有能匹配 sub 的起始位置列表"""
    return [m.start() for m in re.finditer(re.escape(sub), text)]


def fix_offsets(entries, gold_text):
    """
    对于每个实体，找到 gold_text 中所有匹配的位置，
    选取与原始 start 最近的位置来修正。
    """
    fixed = []
    for ent in entries:
        sub = ent['text']
        orig_start = ent['start']
        positions = find_all_positions(gold_text, sub)
        if not positions:
            print(f"[WARN] '{sub}' not found in gold, keeping {orig_start}")
            ent['end'] = orig_start + len(sub)
        else:
            best = min(positions, key=lambda x: abs(x - orig_start))
            if best != orig_start:
                print(f"[FIX] '{sub}': {orig_start} -> {best}")
            ent['start'] = best
            ent['end'] = best + len(sub)
        fixed.append(ent)
    return fixed


def write_ann(entries, out_path):
    """将修正后的实体写入 .ann 文件，保持目录结构"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for ent in entries:
            if ent['ent_id']:
                line = f"{ent['ent_id']}\t{ent['type']} {ent['start']} {ent['end']}\t{ent['text']}"
            else:
                line = f"{ent['type']} {ent['start']} {ent['end']}\t{ent['text']}"
            f.write(line + "\n")
    print(f"Written fixed .ann to {out_path}")


def process_single(ann_path, gold_txt, out_path):
    entries = load_ann(ann_path)
    if not entries:
        print(f"[SKIP] No valid entries in {ann_path}")
        return
    gold_text = open(gold_txt, 'r', encoding='utf-8', errors='ignore').read()
    fixed = fix_offsets(entries, gold_text)
    write_ann(fixed, out_path)


def batch_process(ann_dir, gold_dir, out_dir):
    """
    递归遍历 ann_dir 所有子目录，将每个 .ann 文件修正后输出到 out_dir 相同子目录结构。
    """
    for root, dirs, files in os.walk(ann_dir):
        rel = os.path.relpath(root, ann_dir)
        target_sub = os.path.join(out_dir, rel)
        for fname in files:
            if not fname.lower().endswith('.ann'):
                continue
            ann_path = os.path.join(root, fname)
            base = os.path.splitext(fname)[0]
            gold_txt = os.path.join(gold_dir, base + '.txt')
            if not os.path.exists(gold_txt):
                print(f"[SKIP] No gold .txt for {base}")
                continue
            out_path = os.path.join(target_sub, fname)
            process_single(ann_path, gold_txt, out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Batch fix .ann offsets by nearest match in gold text"
    )
    parser.add_argument("--ann-dir",  required=True,
                        help="Root directory of .ann files (recursive)")
    parser.add_argument("--gold-dir", required=True,
                        help="Directory of corresponding .txt files")
    parser.add_argument("--out-dir",  required=True,
                        help="Where to write fixed .ann files")
    args = parser.parse_args()

    if not os.path.isdir(args.ann_dir) or not os.path.isdir(args.gold_dir):
        print("Error: --ann-dir or --gold-dir is invalid.")
        sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)
    batch_process(args.ann_dir, args.gold_dir, args.out_dir)

if __name__ == '__main__':
    main()
