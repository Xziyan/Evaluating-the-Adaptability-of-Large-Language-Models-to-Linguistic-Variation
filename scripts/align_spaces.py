import sys

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
        elif not in_tag and char != '\n':
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

    for (k_gold, (_, char_gold)), (k_1, (_, char_1)) in zip(sorted_gold, sorted_1):
        if char_gold != char_1:
            if char_gold == ' ':
                # Insert space
                xml_chars.insert(k_1, ' ')
                updated_mapping = {}
                for k, (roff, c) in offset_mapping_1.items():
                    if k >= k_1:
                        updated_mapping[k + 1] = (roff + 1 if c != ' ' else roff, c)
                    else:
                        updated_mapping[k] = (roff, c)
                offset_mapping_1 = updated_mapping
                print(f"Inserted space at position {k_1}")
                break
            else:
                # Remove the mismatching character
                print("".join(xml_chars[k_1:k_1+30]))
                del xml_chars[k_1]
                print("".join(xml_chars[k_1:k_1+30]))
                updated_mapping = {}
                for k, (roff, c) in offset_mapping_1.items():
                    if k == k_1:
                        continue
                    elif k > k_1:
                        updated_mapping[k - 1] = (roff - 1 if c != ' ' else roff, c)
                    else:
                        updated_mapping[k] = (roff, c)
                offset_mapping_1 = updated_mapping
                print(f"Removed space at position {k_1}")
                break

    new_content = ''.join(xml_chars)
    return new_content, offset_mapping_1

def align_until_done(xml_gold, xml_1):
    """
    Repeatedly call align until xml_1 matches the content of xml_gold (ignoring tags).
    Stops if no further changes can be made.
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


def main():
    if len(sys.argv) != 4:
        print("Usage: python align_s.py gold.txt xml_in.xml xml_out.xml")
        sys.exit(1)

    gold_path = sys.argv[1]
    xml_in_path = sys.argv[2]
    xml_out_path = sys.argv[3]
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
