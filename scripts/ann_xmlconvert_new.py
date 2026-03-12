import copy
import sys
import os
import xml.etree.ElementTree as ET
import re
import argparse

COLOR_MAP = {
    "PERS": "red",
    "LOC": "green",
    "PROD": "blue",
    "ORG": "orange",
    "EVENT": "brown",
    "TIME": "purple",
    # add more mappings as needed
}


def parse_xml(xml_content):
    """Parses XML string into root element."""
    return ET.fromstring(xml_content)


def normalize_entity_type(entity_type):
    """Normalizes entity type to uppercase, remove whati s adfter the first dot."""
    return entity_type.split('.')[0].upper()

def convert_to_html(xml_content, filename):
    """Converts XML entities to HTML <span> elements with color."""
    def replacer(match):
        entity_type = normalize_entity_type(match.group(1))
        content = match.group(2)
        color = COLOR_MAP.get(entity_type, "gray")
        return f'<span style="color:{color}">{content}</span>'
    legend = '<div style="margin-bottom: 10px;">'
    for entity, color in COLOR_MAP.items():
        legend += f'<span style="color:{color}">{entity}</span> '
    legend += f'<span style="color:gray">Other</span>'
    legend += '</div>'
    html_body = re.sub(r'<entity type="(.*?)">(.*?)</entity>',
                       replacer, xml_content)
    title = os.path.basename(filename)
    html_output = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
</head>
<body>
{legend}
{html_body}
</body>
</html>"""
    return html_output


def convert_to_ann(xml_content, short=True):
    output = []
    xml_content = xml_content.replace ("<root>","")
    xml_content = xml_content.replace ("</root>","")
    current_entity = extract_first_entity_block(xml_content)
    current_xml_content = xml_content
    counter = 1
    last = None
    while current_entity:
        offset = str.find (current_xml_content,current_entity[0])
        entity_cleaned = remove_tags (current_entity[0])
        length = len(entity_cleaned)
        if short:
            output.append (f"{normalize_entity_type(current_entity[1])}\t{offset} {length+offset}\t{entity_cleaned}")
        else:
            output.append (f"T{counter}\t{normalize_entity_type(current_entity[1])} {offset} {length+offset}\t{entity_cleaned}")
        last = current_entity[0]
        current_xml_content = current_xml_content.replace(current_entity[0], strip_outer_entity_tags(current_entity[0]))
        current_entity = extract_first_entity_block(current_xml_content)
        if current_entity and current_entity[0] == last:
            print(current_entity[0])
            sys.exit("Error: Entity not cleaned properly, please check the XML content.")
        counter += 1
    return "\n".join(output)
def remove_tags (xml_content):
    text = re.sub(r'<entity type="(.*?)">|</entity>', "", xml_content)
    return text

def strip_outer_entity_tags(entity_xml):
    return re.sub(r'^<entity[^>]*>(.*)</entity>$', r'\1', entity_xml, flags=re.DOTALL)


def extract_first_entity_block(xml_content):
    xml_content = f"<root>{xml_content}</root>"
    root = ET.fromstring(xml_content)


    def find_first_entity(element):
        if element.tag == 'entity':
            return element
        for child in element:
            result = find_first_entity(child)
            if result is not None:
                return result
        return None

    first_entity = find_first_entity(root)

    if first_entity is not None:
        # Deep copy to avoid modifying original tree
        clean_entity = copy.deepcopy(first_entity)

        # Remove tail text from all children recursively
        def strip_tails(elem):
            elem.tail = None
            for child in elem:
                strip_tails(child)

        def strip_tail (xml):
            match = re.match(r'^(.*</entity>)(?:.*)$', xml, flags=re.DOTALL)
            return match.group(1) if match else xml
        return strip_tail (ET.tostring(clean_entity, encoding='unicode')), clean_entity.attrib["type"]
    else:
        return None

    first_entity = find_first_entity(root)

    if first_entity is not None:
        return ET.tostring(first_entity, encoding='unicode')
    else:
        return None


def convert_file(xml_file, mode, out_dir):
    with open(xml_file, 'r', encoding='utf-8', errors='ignore') as f:
        raw_xml = f.read()
    if mode == 'html':
        html_content = convert_to_html(raw_xml, xml_file)
        out_file = os.path.join(out_dir, os.path.splitext(os.path.basename(xml_file))[0] + '.html')
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"HTML output written to {out_file}")
    elif mode in ['ann', 'short_ann']:
        ann_content = convert_to_ann(raw_xml, short=(mode == 'short_ann'))
        out_file = os.path.join(out_dir, os.path.splitext(os.path.basename(xml_file))[0] + '.ann')
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(ann_content)
        print(f"ANN output written to {out_file}")
    


def main():
    p = argparse.ArgumentParser(description="Parser-based XML→.ann converter")
    p.add_argument('filename')
    p.add_argument('mode', choices=['ann','short_ann','html'])
    p.add_argument('--out-dir', default=None)
    args = p.parse_args()

    if args.mode == 'html':
        # 简单 html 输出
        with open(args.filename, 'r', encoding='utf-8', errors='ignore') as f:
            xml_str = f.read()
        safe = html.escape(xml_str)
        html_out = f"<html><body><pre>{safe}</pre></body></html>"
        out_dir = args.out_dir or os.path.dirname(args.filename)
        os.makedirs(out_dir, exist_ok=True)
        fn = os.path.join(out_dir, os.path.splitext(os.path.basename(args.filename))[0] + '.html')
        with open(fn, 'w', encoding='utf-8') as f:
            f.write(html_out)
        print(f"HTML output written to {fn}")
    else:
        out_dir = args.out_dir or os.path.dirname(args.filename)
        print("converting", args.filename)
        convert_file(args.filename, args.mode, out_dir)

if __name__ == '__main__':
    main()

