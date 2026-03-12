import sys
import os
import re
import argparse
import xml.etree.ElementTree as ET
import html

def parse_xml_safe(xml_str):
    """
    尝试用 ElementTree 解析 XML；若失败，包装成根元素后再试。
    返回根节点。
    """
    try:
        return ET.fromstring(xml_str)
    except ET.ParseError:
        wrapped = f"<root>{xml_str}</root>"
        return ET.fromstring(wrapped)


def flatten_text(root):
    """
    获取 XML 树中所有文本节点的扁平化字符串并进行 HTML 实体解码。
    """
    text = ''.join(root.itertext())
    return html.unescape(text) #call html.unescape(...) on the flattened text, so by the time we compute offsets, it’s the real & in the plaintext


def extract_entities(root):
    """
    按文档顺序提取所有 <entity> 元素（包括嵌套），返回列表 of (type, text).
    """
    out = []
    for elem in root.findall('.//entity'):
        etype = elem.get('type', '').strip().upper()
        # 拼接所有内部文本内容
        raw = ''.join(elem.itertext())
        clean = html.unescape(raw)
        out.append((etype, clean))
    return out


def match_spans(plain, entities):
    """
    依据 entities 列表中顺序，尝试在 plain 文本中找到非重叠的最早匹配位置。
    返回列表 of (type, start, end, text).
    """
    spans = []  # 已占用 span 列表
    out = []
    for etype, etext in entities:
        found = None
        # 使用字面匹配
        for m in re.finditer(re.escape(etext), plain):
            s,e = m.start(), m.end()
            # 检查是否与已有 span 重叠
            if all(e <= us or s >= ue for us,ue,_,_ in spans):
                found = (s,e)
                break
        if not found:
            print(f"[WARNING] 无法定位实体 '{etext}' 于纯文本中，跳过。")
            continue
        s,e = found
        spans.append((s,e,etype,etext))
        out.append((etype, s, e, etext))
    return out


def write_ann(spans, out_dir, base_name, short):
    """
    根据 spans 写入 .ann 文件。
    """
    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, base_name + '.ann')
    with open(fn, 'w', encoding='utf-8') as f:
        for i, (etype,s,e,etext) in enumerate(spans, start=1):
            if short:
                line = f"{etype} {s} {e}\t{etext}"
            else:
                line = f"T{i}\t{etype} {s} {e}\t{etext}"
            f.write(line + '\n')
    print(f"Annotation output written to {fn}")


def convert_file(xml_file, mode, out_dir):
    with open(xml_file, 'r', encoding='utf-8', errors='ignore') as f:
        raw_xml = f.read()
    root = parse_xml_safe(raw_xml)
    plain = flatten_text(root)
    entities = extract_entities(root)
    spans = match_spans(plain, entities)
    write_ann(spans, out_dir, os.path.splitext(os.path.basename(xml_file))[0], short=(mode == 'short_ann'))


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
        convert_file(args.filename, args.mode, out_dir)

if __name__ == '__main__':
    main()
