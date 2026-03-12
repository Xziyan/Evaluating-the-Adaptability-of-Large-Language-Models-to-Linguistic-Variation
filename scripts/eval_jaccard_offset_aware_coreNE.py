import os
from collections import defaultdict, Counter
import csv
# this script does one-to-one alignement, exact string match, 
# token-based jaccard (threshold 0.5), 
# offset proximity filtering (max_offset_diff), 
# plus core entity match then define new category like tp_full, tp_core (semantically aligned partial match) and fp, fn

#Need to have the same output format logs as eval_jaccard_offset_aware.py
TARGET_TAGS = {"PERS", "LOC", "EVENT", "TIME", "PROD", "ORG"}

def normalize_tag(tag):
    return tag.split('.')[0].upper()

def normalize_text(text):
    return ' '.join(text.lower().split())

def tokenize(text):
    return set(normalize_text(text).split())

def jaccard_similarity(tokens1, tokens2):
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)

def parse_ann_file(path):
    entities = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            tag_info = parts[1].split()
            if len(tag_info) < 3:
                continue
            raw_tag = tag_info[0]
            tag = normalize_tag(raw_tag)
            if tag not in TARGET_TAGS:
                continue
            try:
                start, end = int(tag_info[1]), int(tag_info[2])
            except ValueError:
                continue
            text = normalize_text(parts[2])
            entities.append((tag, text, start, end))
    return entities

def fuzzy_match_with_core(gold, pred, threshold=0.5, max_offset_diff=15):
    gold_data = [(tag, text, tokenize(text), start, end) for tag, text, start, end in gold]
    pred_data = [(tag, text, tokenize(text), start, end) for tag, text, start, end in pred]

    matches = []
    for i_g, (g_tag, g_text, g_tok, g_start, g_end) in enumerate(gold_data):
        for i_p, (p_tag, p_text, p_tok, p_start, p_end) in enumerate(pred_data):
            if g_tag != p_tag:
                continue
            sim = jaccard_similarity(g_tok, p_tok)
            offset_dist = abs(g_start - p_start)
            if sim >= threshold and offset_dist <= max_offset_diff:
                matches.append((sim, offset_dist, i_g, i_p))

    matches.sort(key=lambda x: (-x[0], x[1]))
    matched_gold = set()
    matched_pred = set()
    logs = []
    tp_full, tp_core = 0, 0

    for sim, offset_dist, i_g, i_p in matches:
        if i_g in matched_gold or i_p in matched_pred:
            continue
        matched_gold.add(i_g)
        matched_pred.add(i_p)
        g_tag, g_text, _, _, _ = gold_data[i_g]
        p_tag, p_text, _, _, _ = pred_data[i_p]
        if normalize_text(p_text) == normalize_text(g_text):
            tp_full += 1
            logs.append(f"[TP_full] ({p_tag}) '{p_text}' ↔ '{g_text}'")
        else:
            tp_core += 1
            logs.append(f"[TP_core] ({p_tag}) '{p_text}' ⊂ '{g_text}' (sim={sim:.2f})")

    for i_p, (p_tag, p_text, _, _, _) in enumerate(pred_data):
        if i_p not in matched_pred:
            logs.append(f"[FP] ({p_tag}) PRED='{p_text}'")

    for i_g, (g_tag, g_text, _, _, _) in enumerate(gold_data):
        if i_g not in matched_gold:
            logs.append(f"[FN] ({g_tag}) GOLD='{g_text}'")

    return tp_full, tp_core, len(pred_data) - tp_full - tp_core, len(gold_data) - tp_full - tp_core, logs

def compute_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1

def evaluate_all(pred_base_dir, gold_dir, threshold=0.5, max_offset_diff=15):
    results_total = defaultdict(Counter)
    results_by_tag = defaultdict(lambda: defaultdict(Counter))
    results_by_genre = defaultdict(lambda: defaultdict(Counter))

    for subdir in sorted(os.listdir(pred_base_dir)):
        pred_dir = os.path.join(pred_base_dir, subdir)
        model = subdir
        if not os.path.isdir(pred_dir):
            continue
        print(f"Evaluating model: {model}")

        for fname in os.listdir(pred_dir):
            if not fname.endswith(".ann"):
                continue
            pred_file = os.path.join(pred_dir, fname)
            gold_file = os.path.join(gold_dir, fname)
            if not os.path.exists(gold_file):
                print(f"[Warning] Missing gold file: {fname}")
                continue

            pred_ents = parse_ann_file(pred_file)
            gold_ents = parse_ann_file(gold_file)

            tp_full, tp_core, fp, fn, match_log = fuzzy_match_with_core(
                gold_ents, pred_ents, threshold=threshold, max_offset_diff=max_offset_diff
            )

            log_path = os.path.join("match_logs_core_OffSetaware_CORE", model)
            os.makedirs(log_path, exist_ok=True)
            with open(os.path.join(log_path, fname.replace(".ann", ".log")), "w", encoding="utf-8") as f:
                for line in match_log:
                    f.write(line + "\n")

            results_total[model]["TP_full"] += tp_full
            results_total[model]["TP_core"] += tp_core
            results_total[model]["FP"] += fp
            results_total[model]["FN"] += fn

            genre = fname.split("0")[0].lower()
            results_by_genre[model][genre]["TP_full"] += tp_full
            results_by_genre[model][genre]["TP_core"] += tp_core
            results_by_genre[model][genre]["FP"] += fp
            results_by_genre[model][genre]["FN"] += fn

    return results_total, results_by_genre

def export_core_csv(results_total, results_by_genre, out_dir="fuzzy_eval_OffSetaware_CORE", threshold=0.5):
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/micro_core_f1_summary.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Threshold", "TP_full", "TP_core", "FP", "FN", "Precision", "Recall", "F1"])
        for model in results_total:
            tp = results_total[model]["TP_full"] + results_total[model]["TP_core"]
            fp = results_total[model]["FP"]
            fn = results_total[model]["FN"]
            p, r, f1 = compute_f1(tp, fp, fn)
            writer.writerow([model, threshold, results_total[model]["TP_full"], results_total[model]["TP_core"], fp, fn, round(p, 4), round(r, 4), round(f1, 4)])

    with open(f"{out_dir}/per_genre_core_f1.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Genre", "TP_full", "TP_core", "FP", "FN", "Precision", "Recall", "F1"])
        for model in results_by_genre:
            for genre in results_by_genre[model]:
                tp = results_by_genre[model][genre]["TP_full"] + results_by_genre[model][genre]["TP_core"]
                fp = results_by_genre[model][genre]["FP"]
                fn = results_by_genre[model][genre]["FN"]
                p, r, f1 = compute_f1(tp, fp, fn)
                writer.writerow([model, genre, results_by_genre[model][genre]["TP_full"], results_by_genre[model][genre]["TP_core"], fp, fn, round(p, 4), round(r, 4), round(f1, 4)])

if __name__ == "__main__":
    gold_dir = "/Users/ziyanxu/AGLAGLA/dataog"
    pred_base_dir = "/Users/ziyanxu/AGLAGLA/output_gml_xml"
    threshold = 0.5
    max_offset_diff = 40

    total, by_genre = evaluate_all(pred_base_dir, gold_dir, threshold, max_offset_diff)
    export_core_csv(total, by_genre, threshold=threshold)
