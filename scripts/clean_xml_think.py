# extract only the last <root>...</root> block from XML files, or wrap content in <root> if none found
import sys, re, html
from pathlib import Path

def extract_last_root_block(text: str) -> str:
    blocks = [m.group(0) for m in re.finditer(r"<root\b[^>]*>.*?</root>", text, flags=re.DOTALL | re.IGNORECASE)]
    if blocks: return blocks[-1].strip()
    unescaped = html.unescape(text)
    if unescaped != text:
        blocks2 = [m.group(0) for m in re.finditer(r"<root\b[^>]*>.*?</root>", unescaped, flags=re.DOTALL | re.IGNORECASE)]
        if blocks2: return blocks2[-1].strip()
    m = list(re.finditer(r"<root\b[^>]*>", text, flags=re.IGNORECASE))
    if m:
        start = m[-1].start()
        candidate = text[start:].strip()
        if "</root>" not in candidate.lower():
            candidate += "\n</root>"
        return candidate
    return f"<root>\n{text.strip()}\n</root>"

def main(folder):
    folder = Path(folder)
    for p in sorted(folder.glob("*.xml")):
        raw = p.read_text(encoding="utf-8", errors="ignore")
        clean = extract_last_root_block(raw)
        if clean != raw:
            p.write_text(clean, encoding="utf-8")
            print(f"[CLEANED] {p.name}")
        else:
            print(f"[OK] {p.name}")

if __name__ == "__main__":
    main(sys.argv[1])
