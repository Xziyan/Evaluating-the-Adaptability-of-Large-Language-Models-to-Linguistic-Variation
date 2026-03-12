import os
import sys
import re
import Levenshtein as lev


def is_xml_candidate(fname: str) -> bool:
    low = fname.lower()
    return low.endswith(".xml") and ".raw" not in low and "only_root" not in low


def process_file(gold_txt_file, in_path, out_path):
    try:
        with open(in_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        with open(gold_txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            content_gold = f.read()
    except Exception as e:
        print(f"[ERROR] reading {in_path}: {e}")
        return

    def align(content, content_gold):
        content_free = re.sub(r'<[^>]+>', '', content)
        ops = lev.editops(content_free, content_gold)
        # Maitain a dictionary of characters in content
        # With the keys being the positio
        # And the values being a two way list between
        # The character at this index
        # And the xml tag that start at this index
        # We skip the xml tags for the indexes
        content_chars = {}
        i_content = 0
        real_index = 0
        while real_index < len(content):
            char = content[real_index]
            if char == '<':
                xml_tag = char
                i = real_index
                while i < len(content) and content[i] != '>':
                    i += 1
                    xml_tag += content[i]
                real_index = i+1
                content_chars[i_content] = [None, xml_tag]
            else:
                if i_content not in content_chars:
                    content_chars[i_content] = [char, None]
                else:
                    content_chars[i_content][0] = char
                i_content += 1
                real_index += 1
        # Now to list
        l = [None] * len(content_chars)
        for k, v in content_chars.items():
            l[k] = v
        # Now we repercute the operations on l
        for operation in ops:
            op_type, place, place_gold = operation
            if op_type == 'delete':
                print(f"[DELETE] {content_free[place]} at position {place}")
                xml_place = l[place][1]
                l[place][0] = None
            if op_type == 'replace':
                l[place][0] = content_gold[place_gold]
                print(f"[REPLACE] {content_free[place]} with {content_gold[place_gold]} at position {place}")
            if op_type == 'insert':
                print(f"[INSERT] {content_gold[place_gold]
                                  } at position {place}")
                l[place-1][0] += content_gold[place_gold]
        fixed_text = ""
        for chars, xml_tag in l:
            if xml_tag is not None:
                fixed_text += xml_tag
            if chars is not None:
                fixed_text += chars
        return fixed_text

    fixed = align(content, content_gold)
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
            name_file = in_path.split("/")[-1].replace('.xml', '.txt')
            gold_txt_file = os.path.join(gold_txt_folder, name_file)
            process_file(gold_txt_file, in_path, out_path)


def usage():
    print("Usage: python align_xml_spaces_simple.py <gold_txt_folder_unused> <xml_root_dir> <output_root>")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        usage()
        sys.exit(1)
    _, gold_folder, xml_root_dir, output_root = sys.argv
    batch_align(gold_folder, xml_root_dir, output_root)
