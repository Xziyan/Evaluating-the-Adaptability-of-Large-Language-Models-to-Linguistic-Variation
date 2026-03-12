
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Single-folder evaluator for NER .ann files (LLM outputs vs gold), with micro F1 per document and per genre.
#
# - Strict matching: tag and exact [start,end) match.
# - Fuzzy matching: tag matches and Jaccard over character spans ≥ threshold (default 0.5).
#
# Assumptions:
# - Predictions and gold are in separate folders, filenames match (e.g., foo.ann vs foo.ann).
# - .ann format lines like: T1<TAB>TYPE start end<TAB>text
# - Genres are inferred from filename prefixes (before first '_' or '-' or digit).
#   You can override/normalize via --genre-map FILE (CSV with columns: prefix,genre).
#
# Outputs:
# - Prints summary to stdout.
# - Writes CSVs:
#   - per_document.csv (doc, genre, tp, fp, fn, precision, recall, f1)
#   - per_genre.csv (genre, tp, fp, fn, precision, recall, f1)
#   - overall.csv (single row aggregate micro scores)
#
# Usage:
#     python eval_single_folder.py --pred PRED_DIR --gold GOLD_DIR [--fuzzy 0.5]

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

TARGET_TAGS = {"PERS", "LOC", "EVENT", "TIME", "PROD", "ORG"}

def normalize_tag(tag: str) -> str:
    return tag.split(".")[0].upper()

def parse_ann_file(path: Path):
    '''
    Parse a .ann file; return list of (TAG, start, end, text) filtered to TARGET_TAGS.
    '''
    ents = []
    if not path.exists():
        return ents
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Only entity lines start with 'T' in BRAT; skip attributes/relations (A,R,E,*)
        if not line.startswith('T'):
            continue
        parts = line.split("\t")
        # Expected minimal: [TID, "TYPE start end", "text"]
        if len(parts) < 2:
            continue
        span_info = parts[1]
        text = parts[2] if len(parts) >= 3 else ""
        # span_info like: "TYPE start end" or "TYPE start1 end1;start2 end2"
        span_tokens = span_info.split()
        if not span_tokens:
            continue
        tag = normalize_tag(span_tokens[0])
        if tag not in TARGET_TAGS:
            continue
        # Extract first start/end pair (handle discontinuous spans by taking the first segment)
        try:
            # Join the rest and split on ';' to isolate first segment
            rest = " ".join(span_tokens[1:])
            first_seg = rest.split(';', 1)[0].strip()
            se = first_seg.split()
            if len(se) < 2:
                continue
            start_i, end_i = int(se[0]), int(se[1])
        except Exception:
            continue
        ents.append((tag, start_i, end_i, text))
    return ents

def iou_char(a, b) -> float:
    a0,a1 = a; b0,b1 = b
    inter = max(0, min(a1,b1) - max(a0,b0))
    union = max(a1-a0,0) + max(b1-b0,0) - inter
    return (inter/union) if union>0 else 0.0

def match_strict(pred, gold):
    used_g = set()
    tp = 0
    fp = 0
    for p in pred:
        found = False
        for j,g in enumerate(gold):
            if j in used_g:
                continue
            if p[0]==g[0] and p[1]==g[1] and p[2]==g[2]:
                used_g.add(j)
                tp += 1
                found = True
                break
        if not found:
            fp += 1
    fn = len(gold) - len(used_g)
    return tp, fp, fn

def match_fuzzy(pred, gold, thr: float=0.5):
    used_g = set()
    tp = 0
    fp = 0
    for p in pred:
        best_j = -1
        best_iou = 0.0
        for j,g in enumerate(gold):
            if j in used_g or p[0] != g[0]:
                continue
            iou = iou_char((p[1],p[2]), (g[1],g[2]))
            if iou >= thr and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            used_g.add(best_j)
            tp += 1
        else:
            fp += 1
    fn = len(gold) - len(used_g)
    return tp, fp, fn

def safe_div(a: int, b: int) -> float:
    return a/b if b else 0.0

def prf(tp: int, fp: int, fn: int):
    prec = safe_div(tp, tp+fp)
    rec = safe_div(tp, tp+fn)
    f1 = safe_div(2*prec*rec, (prec+rec)) if (prec+rec)>0 else 0.0
    return prec, rec, f1

GENRE_PREFIX_RE = re.compile(r'^([A-Za-zÀ-ÖØ-öø-ÿ]+)')

def infer_genre(filename: str, genre_map):
    '''
    Infer genre from filename stem prefix (before '_' or '-') or via map.
    '''
    stem = Path(filename).stem
    # try maps on full stem and on prefix tokens
    for key in (stem, stem.split('_')[0], stem.split('-')[0]):
        if key in genre_map:
            return genre_map[key]
    m = GENRE_PREFIX_RE.match(stem)
    if m:
        pref = m.group(1)
        if pref in genre_map:
            return genre_map[pref]
        return pref
    return 'unknown'

def load_genre_map(path: Path):
    mapping = {}
    if path and path.exists():
        import csv
        with path.open(encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping[row['prefix']] = row['genre']
    return mapping

def main():
    ap = argparse.ArgumentParser(description='Evaluate NER .ann predictions vs gold for a single folder; report micro F1 per document and per genre.')
    ap.add_argument('--pred', required=True, help='Folder with predicted .ann files')
    ap.add_argument('--gold', required=True, help='Folder with gold .ann files')
    ap.add_argument('--fuzzy', type=float, default=None, help='If set, use fuzzy IoU threshold (e.g., 0.5); otherwise strict.')
    ap.add_argument('--genre-map', type=str, default=None, help='CSV with columns: prefix,genre to normalize inferred genres')
    ap.add_argument('--out', type=str, default='.', help='Output folder for CSVs')
    args = ap.parse_args()

    pred_dir = Path(args.pred)
    gold_dir = Path(args.gold)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    genre_map = load_genre_map(Path(args.genre_map)) if args.genre_map else {}

    pred_files = sorted(pred_dir.glob('*.ann'))
    if not pred_files:
        print(f'No .ann files found in {pred_dir}')
        return

    per_doc = []
    per_genre_counts = defaultdict(lambda: {'tp':0,'fp':0,'fn':0})
    total = {'tp':0,'fp':0,'fn':0}
    mode = 'strict' if args.fuzzy is None else f'fuzzy@{args.fuzzy}'

    for p in pred_files:
        g = gold_dir / p.name
        pred = parse_ann_file(p)
        gold = parse_ann_file(g)
        if args.fuzzy is None:
            tp, fp, fn = match_strict(pred, gold)
        else:
            tp, fp, fn = match_fuzzy(pred, gold, args.fuzzy)
        prec, rec, f1 = prf(tp, fp, fn)
        genre = infer_genre(p.name, genre_map)
        per_doc.append({
            'document': p.name,
            'genre': genre,
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(prec,6),
            'recall': round(rec,6),
            'f1': round(f1,6),
            'mode': mode
        })
        per_genre_counts[genre]['tp'] += tp
        per_genre_counts[genre]['fp'] += fp
        per_genre_counts[genre]['fn'] += fn
        total['tp'] += tp; total['fp'] += fp; total['fn'] += fn

    # write per_document.csv
    doc_csv = out_dir / 'per_document.csv'
    with doc_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['document','genre','tp','fp','fn','precision','recall','f1','mode'])
        w.writeheader()
        w.writerows(per_doc)

    # per_genre.csv
    genre_csv = out_dir / 'per_genre.csv'
    with genre_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['genre','tp','fp','fn','precision','recall','f1','mode'])
        w.writeheader()
        for g, c in sorted(per_genre_counts.items()):
            prec, rec, f1 = prf(c['tp'], c['fp'], c['fn'])
            w.writerow({
                'genre': g,
                'tp': c['tp'], 'fp': c['fp'], 'fn': c['fn'],
                'precision': round(prec,6),
                'recall': round(rec,6),
                'f1': round(f1,6),
                'mode': mode
            })

    # overall.csv
    overall_csv = out_dir / 'overall.csv'
    with overall_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['tp','fp','fn','precision','recall','f1','mode'])
        w.writeheader()
        prec, rec, f1 = prf(total['tp'], total['fp'], total['fn'])
        w.writerow({
            'tp': total['tp'], 'fp': total['fp'], 'fn': total['fn'],
            'precision': round(prec,6),
            'recall': round(rec,6),
            'f1': round(f1,6),
            'mode': mode
        })

    print(f'Mode: {mode}')
    prec, rec, f1 = prf(total['tp'], total['fp'], total['fn'])
    print(f'Overall micro: P={prec:.4f} R={rec:.4f} F1={f1:.4f} over {total["tp"]+total["fp"]} predictions and {total["tp"]+total["fn"]} gold.')
    print(f'Wrote CSVs in {out_dir}:')
    print(f' - {doc_csv.name}')
    print(f' - {genre_csv.name}')
    print(f' - {overall_csv.name}')

if __name__ == '__main__':
    main()
