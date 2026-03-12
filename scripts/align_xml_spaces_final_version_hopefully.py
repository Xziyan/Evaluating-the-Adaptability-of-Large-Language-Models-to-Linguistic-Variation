import os
import sys
import re
import Levenshtein as lev


def is_xml_candidate(fname: str) -> bool:
    low = fname.lower()
    return low.endswith(".xml") and ".raw" not in low and "only_root" not in low


def process_file(gold_txt_file, in_path, out_path, do_alignement=True):
    try:
        with open(in_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        with open(gold_txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            content_gold = f.read()
    except Exception as e:
        print(f"[ERROR] reading {in_path}: {e}")
        return

    def align(content, content_gold, verbose=False):
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
                if i_content not in content_chars:
                    content_chars[i_content] = [None, xml_tag]
                else:
                    content_chars[i_content] = [content_chars[i_content][0],content_chars[i_content][1]+xml_tag]
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
                if verbose:
                    print(f"[DELETE] {content_free[place]} at position {place}")
                l[place][0] = None
            if op_type == 'replace':
                l[place][0] = content_gold[place_gold]
                if verbose:
                    print(f"[REPLACE] {content_free[place]} with {content_gold[place_gold]} at position {place}")
            if op_type == 'insert':
                if verbose:
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

    if do_alignement:
        fixed = align(content, content_gold)
    else:
        fixed = content
    # Replace <entity type=...></entity> with ''
    # We need to do this recursively
    different = True
    old_fixed = ""
    while different:
        fixed = re.sub(r'<entity[^>]*></entity>', '', fixed)
        different = old_fixed != fixed
        old_fixed = fixed
    fixed_new = ""
    for char in fixed:
        # Check if char is a valid xml token (not &, etc...)
        if char in ['&']:
            fixed_new += 'K'
        else:
            fixed_new += char
    # Post-process to add </entity> tags if there is a mismatch
    # Count <entity and </entity>
    # using fixed_new.findall('<entity')
    entity_open_count = fixed_new.count('<entity')
    entity_close_count = fixed_new.count('</entity>')
    if entity_open_count != entity_close_count:
        # At the end of each line, add </entity> enough times, behind the last </entity>
        print(f"[WARNING] Mismatch in entity tags: {entity_open_count} <entity, {entity_close_count} </entity")
        lines = fixed_new.split('\n')
        for i in range(len(lines)):
            if lines[i].count('</entity>') < lines[i].count('<entity'):
                # Add </entity> at the end of the line
                to_add = '</entity>' * (lines[i].count('<entity') - lines[i].count('</entity>'))
                # Find the last </entity> in the line
                last_entity_index = lines[i].rfind('</entity>')
                if last_entity_index != -1:
                    lines[i] = lines[i][:last_entity_index + len('</entity>')] + to_add + lines[i][last_entity_index + len('</entity>'):]
                else:
                    lines[i] += to_add
        fixed_new = '\n'.join(lines)
        entity_open_count = fixed_new.count('<entity')
        entity_close_count = fixed_new.count('</entity>')
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(fixed_new)
        print(f"[ALIGNED_SIMPLE] {in_path} -> {out_path}")
    except Exception as e:
        print(f"[ERROR] writing aligned simple xml {out_path}: {e}")


def batch_align(gold_txt_folder, xml_root_dir, output_root, do_alignement=True):
    for root, dirs, files in os.walk(xml_root_dir):
        rel = os.path.relpath(root, xml_root_dir)
        target_dir = os.path.join(output_root, rel)
        os.makedirs(target_dir, exist_ok=True)
        for file in files:
            if not is_xml_candidate(file):
                continue
            print (file)
            in_path = os.path.join(root, file)
            out_path = os.path.join(target_dir, file)
            name_file = in_path.split("/")[-1].replace('.xml', '.txt')
            gold_txt_file = os.path.join(gold_txt_folder, name_file)
            process_file(gold_txt_file, in_path, out_path, do_alignement)


def usage():
    print("Usage: python align_xml_spaces_final_version_hopefully.py <gold_txt_folder_unused> <xml_root_dir> <output_root> <do_alignement>")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        usage()
        sys.exit(1)
    _, gold_folder, xml_root_dir, output_root, do_alignement = sys.argv
    batch_align(gold_folder, xml_root_dir, output_root, do_alignement == 'True')
