import sys #版本1
#This script is capable to deal with the unaligned spaces due to llm deletion or correction of spaces, but can't deal with other modification made by llm. 
def map_xml_offsets(xml_content: str) -> dict:
    """
    Map character offsets in an XML string to their real content offsets, ignoring tags.
    Returns a dict: {xml_offset: (real_offset, character)}
    """
    in_tag = False
    real_offset = 0
    offset_map = {}

    for i, char in enumerate(xml_content):
        if char == '<':
            in_tag = True
        elif char == '>':
            in_tag = False
            continue
        elif not in_tag:
            offset_map[i] = (real_offset, char)
            real_offset += 1

    return offset_map

def align(offset_mapping_gold, offset_mapping_1, xml_content_1):
    """
    Aligns xml_content_1 to match the gold offset map by inserting/removing spaces.
    Returns the new xml content and updated offset map.
    """
    sorted_gold = sorted(offset_mapping_gold.items())
    sorted_1 = sorted(offset_mapping_1.items())

    xml_chars = list(xml_content_1)
    i = -1
    for (k_gold, (_, char_gold)), (k_1, (_, char_1)) in zip(sorted_gold, sorted_1):
        i += 1
        if char_gold != char_1 :
            if char_gold == ' 'and sorted_gold[i-1][1] == sorted_gold[i-1][1]:
                xml_chars.insert(k_1, ' ')
                updated_mapping = {}
                for k, (roff, c) in offset_mapping_1.items():
                    if k >= k_1:
                        updated_mapping[k + 1] = (roff + 1 if c != ' ' else roff, c)
                    else:
                        updated_mapping[k] = (roff, c)
                offset_mapping_1 = updated_mapping
                #print(f"Inserted space at position {k_1}")
                break
            elif char_1 == '\n' and sorted_gold[i-1][1] == sorted_gold[i-1][1]:
                del xml_chars[k_1]
                updated_mapping = {}
                for k, (roff, c) in offset_mapping_1.items():
                    if k == k_1:
                        continue
                    elif k > k_1:
                        updated_mapping[k - 1] = (roff - 1 if c != ' ' else roff, c)
                    else:
                        updated_mapping[k] = (roff, c)
                offset_mapping_1 = updated_mapping
                #print(f"Removed linebreak at position {k_1}")
                break
    new_content = ''.join(xml_chars)
    return new_content, offset_mapping_1

def align_until_done(xml_gold, xml_1):
    """
    Repeatedly call align until xml_1 matches the content of xml_gold (ignoring tags).
    """
    gold_mapping = map_xml_offsets(xml_gold)
    while True:
        mapping_1 = map_xml_offsets(xml_1)

        gold_chars = [char for _, char in sorted(gold_mapping.values())]
        chars_1 = [char for _, char in sorted(mapping_1.values())]

        if gold_chars == chars_1:
            print("Alignment complete.")
            return xml_1

        old_xml_1 = xml_1
        xml_1, mapping_1 = align(gold_mapping, mapping_1, xml_1)
        if xml_1 == old_xml_1:
            print("Failed")
            return xml_1


import os

def main():
    if len(sys.argv) != 4:
        print("Usage: python align_spaces_new.py gold_dir xml_in_dir xml_out_dir")
        sys.exit(1)

    gold_dir = sys.argv[1]
    xml_in_dir = sys.argv[2]
    xml_out_dir = sys.argv[3]

    os.makedirs(xml_out_dir, exist_ok=True)

    for filename in os.listdir(xml_in_dir):
        xml_in_path = os.path.join(xml_in_dir, filename)
        gold_path = os.path.join(gold_dir, filename)
        xml_out_path = os.path.join(xml_out_dir, filename)

        if not os.path.exists(gold_path):
            print(f"Skipping {filename}, no corresponding gold file.")
            continue

        with open(gold_path, 'r', encoding='utf-8') as f:
            xml_gold = f.read()
        with open(xml_in_path, 'r', encoding='utf-8') as f:
            xml_1 = f.read()

        aligned_xml = align_until_done(xml_gold, xml_1)

        with open(xml_out_path, 'w', encoding='utf-8') as f:
            f.write(aligned_xml)
        print(f"Aligned XML written to: {xml_out_path}")

if __name__ == "__main__":
    main()
