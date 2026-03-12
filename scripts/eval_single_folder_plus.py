#正确的在用的版本
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Enhanced evaluator with optional per-file match dumps (TP/FP/FN) and side-by-side diff TSV.
#
# Usage:
#   python eval_single_folder_plus.py --pred PRED_DIR --gold GOLD_DIR --out OUT_DIR [--fuzzy 0.5] [--genre-map map.csv] [--write-matches]
#
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional
from datetime import datetime

TARGET_TAGS = {"PERS", "LOC", "EVENT", "TIME", "PROD", "ORG"}

def normalize_tag(tag: str) -> str:
    return tag.split(".")[0].upper()

def parse_ann_file(path: Path):
    ents = []
    if not path.exists():
        return ents
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("T"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        span_info = parts[1]
        text = parts[2] if len(parts) >= 3 else ""
        span_tokens = span_info.split()
        if not span_tokens:
            continue
        tag = normalize_tag(span_tokens[0])
        if tag not in TARGET_TAGS:
            continue
        try:
            rest = " ".join(span_tokens[1:])
            first_seg = rest.split(";", 1)[0].strip()
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

def match_strict_pairs(pred, gold):
    used_g = set()
    pairs = []
    fp_idx = []
    for i, p in enumerate(pred):
        found = False
        for j, g in enumerate(gold):
            if j in used_g:
                continue
            if p[0]==g[0] and p[1]==g[1] and p[2]==g[2]:
                used_g.add(j)
                pairs.append((i, j, 1.0))
                found = True
                break
        if not found:
            fp_idx.append(i)
    fn_idx = [j for j in range(len(gold)) if j not in used_g]
    return pairs, fp_idx, fn_idx

def match_fuzzy_pairs(pred, gold, thr: float=0.5):
    used_g = set()
    pairs = []
    fp_idx = []
    for i, p in enumerate(pred):
        best_j = -1
        best_iou = 0.0
        for j, g in enumerate(gold):
            if j in used_g or p[0] != g[0]:
                continue
            iou = iou_char((p[1],p[2]), (g[1],g[2]))
            if iou >= thr and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            used_g.add(best_j)
            pairs.append((i, best_j, best_iou))
        else:
            fp_idx.append(i)
    fn_idx = [j for j in range(len(gold)) if j not in used_g]
    return pairs, fp_idx, fn_idx

def safe_div(a: int, b: int) -> float:
    return a/b if b else 0.0

def prf(tp: int, fp: int, fn: int):
    prec = safe_div(tp, tp+fp)
    rec = safe_div(tp, tp+fn)
    f1 = safe_div(2*prec*rec, (prec+rec)) if (prec+rec)>0 else 0.0
    return prec, rec, f1

GENRE_PREFIX_RE = re.compile(r'^([A-Za-zÀ-ÖØ-öø-ÿ]+)')

def infer_genre(filename: str, genre_map):
    stem = Path(filename).stem
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

def load_genre_map(path: Optional[Path]):
    mapping = {}
    if path and path.exists():
        with path.open(encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping[row['prefix']] = row['genre']
    return mapping

def write_match_csvs(out_dir: Path, doc_stem: str, pred, gold, pairs, fp_idx, fn_idx, mode: str):
    ddir = out_dir / "matches" / doc_stem
    ddir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # TP
    with (ddir / "tp.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode","timestamp","pred_tag","pred_start","pred_end","pred_text","gold_tag","gold_start","gold_end","gold_text","iou"])
        for (pi, gi, iou) in pairs:
            p = pred[pi]; g = gold[gi]
            w.writerow([mode, ts, p[0], p[1], p[2], p[3], g[0], g[1], g[2], g[3], f"{iou:.6f}"])

    # FP
    with (ddir / "fp.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode","timestamp","pred_tag","pred_start","pred_end","pred_text"])
        for pi in fp_idx:
            p = pred[pi]
            w.writerow([mode, ts, p[0], p[1], p[2], p[3]])

    # FN
    with (ddir / "fn.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode","timestamp","gold_tag","gold_start","gold_end","gold_text"])
        for gi in fn_idx:
            g = gold[gi]
            w.writerow([mode, ts, g[0], g[1], g[2], g[3]])

def write_side_by_side_diff(out_dir: Path, doc_stem: str, pred, gold, pairs, fp_idx, fn_idx, mode: str):
    '''
    Create a side-by-side TSV similar to a "diff", with two columns of entity tuples.
    Left column shows prediction-side entries, right shows gold-side entries.
    - TP: same row with "TP" on both sides (pred matched to gold)
    - FP: left has "FP ...", right is empty and a ">" marker in the middle
    - FN: right has "FN ...", left is empty and a "<" marker in the middle
    '''
    ddir = out_dir / "matches" / doc_stem
    ddir.mkdir(parents=True, exist_ok=True)
    out_file = ddir / "diff.tsv"
    sep = "\t"
    mid_pad = " " * 14  # visual gap between columns
    with out_file.open("w", encoding="utf-8", newline="") as f:
        header = sep.join(["Type","Tag","Start","End","Text"]) + mid_pad + sep.join(["Type","Tag","Start","End","Text"])
        f.write(header + "\n")
        # TPs
        for (pi, gi, _iou) in pairs:
            p = pred[pi]; g = gold[gi]
            left = sep.join(["TP", p[0], str(p[1]), str(p[2]), p[3]])
            right = sep.join(["TP", g[0], str(g[1]), str(g[2]), g[3]])
            f.write(left + mid_pad + right + "\n")
        # FPs (pred only)
        for pi in fp_idx:
            p = pred[pi]
            left = sep.join(["FP", p[0], str(p[1]), str(p[2]), p[3]])
            marker = mid_pad[:-2] + " > "
            f.write(left + marker + "\n")
        # FNs (gold only)
        for gi in fn_idx:
            g = gold[gi]
            right = sep.join(["FN", g[0], str(g[1]), str(g[2]), g[3]])
            marker = " < " + mid_pad[2:]
            f.write(marker + right + "\n")

def main():
    ap = argparse.ArgumentParser(description='Evaluate NER .ann predictions vs gold for a single folder; report micro F1 per document and per genre. Optional per-file TP/FP/FN dumps and side-by-side diff.')
    ap.add_argument('--pred', required=True, help='Folder with predicted .ann files')
    ap.add_argument('--gold', required=True, help='Folder with gold .ann files')
    ap.add_argument('--fuzzy', type=float, default=None, help='If set, use fuzzy IoU threshold (e.g., 0.5); otherwise strict.')
    ap.add_argument('--genre-map', type=str, default=None, help='CSV with columns: prefix,genre to normalize inferred genres')
    ap.add_argument('--out', type=str, default='.', help='Output folder for CSVs')
    ap.add_argument('--write-matches', action='store_true', help='Write TP/FP/FN CSVs + side-by-side diff under OUT/matches/<doc_stem>/')
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
            pairs, fp_idx, fn_idx = match_strict_pairs(pred, gold)
        else:
            pairs, fp_idx, fn_idx = match_fuzzy_pairs(pred, gold, args.fuzzy)

        tp = len(pairs); fp = len(fp_idx); fn = len(fn_idx)
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

        if args.write_matches:
            doc_stem = p.stem
            write_match_csvs(out_dir, doc_stem, pred, gold, pairs, fp_idx, fn_idx, mode)
            write_side_by_side_diff(out_dir, doc_stem, pred, gold, pairs, fp_idx, fn_idx, mode)

    with (out_dir / 'per_document.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['document','genre','tp','fp','fn','precision','recall','f1','mode'])
        w.writeheader()
        w.writerows(per_doc)

    with (out_dir / 'per_genre.csv').open('w', newline='', encoding='utf-8') as f:
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

    with (out_dir / 'overall.csv').open('w', newline='', encoding='utf-8') as f:
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
    if args.write_matches:
        print(f'Wrote per-file TP/FP/FN and side-by-side diffs under: {out_dir / "matches"}')
    print(f'Wrote CSVs in {out_dir}: per_document.csv, per_genre.csv, overall.csv')

if __name__ == '__main__':
    main()
