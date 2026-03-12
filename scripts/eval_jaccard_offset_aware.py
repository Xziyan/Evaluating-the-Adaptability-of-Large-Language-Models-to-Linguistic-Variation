import os
from collections import defaultdict, Counter
import csv

# 6 main entity tags to evaluate
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

# Parser that reads offsets from .ann files
def parse_ann_surface_with_offsets(path):
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
            tag = normalize_tag(tag_info[0])
            if tag not in TARGET_TAGS:
                continue
            try:
                start, end = int(tag_info[1]), int(tag_info[-1])
            except ValueError:
                continue
            text = normalize_text(parts[2])
            entities.append((tag, text, start, end))
    return entities

#  One-to-one, 
# similarity-based fuzzy matcher 
# with offset distance constraint
def fuzzy_match_jaccard_detailed_with_offsets(gold, pred, threshold=0.5, max_offset_diff=15):
    gold_tokens = [(tag, text, tokenize(text), start, end) for tag, text, start, end in gold]
    pred_tokens = [(tag, text, tokenize(text), start, end) for tag, text, start, end in pred]
    
    candidates = []
    for i_g, (g_tag, g_text, g_tok, g_start, g_end) in enumerate(gold_tokens):
        for i_p, (p_tag, p_text, p_tok, p_start, p_end) in enumerate(pred_tokens):
            if g_tag != p_tag:
                continue
            sim = jaccard_similarity(g_tok, p_tok)
            offset_dist = abs(g_start - p_start)
            if sim >= threshold and offset_dist <= max_offset_diff:
                candidates.append((sim, offset_dist, i_g, i_p))
    
    # Sort by descending similarity, then ascending offset distance
    candidates.sort(key=lambda x: (-x[0], x[1]))
    
    matched_gold = set()
    matched_pred = set()
    log = []
    tp = 0

    for sim, offset_dist, i_g, i_p in candidates:
        if i_g in matched_gold or i_p in matched_pred:
            continue
        matched_gold.add(i_g)
        matched_pred.add(i_p)
        tp += 1
        g_tag, g_text, _, _, _ = gold_tokens[i_g]
        p_tag, p_text, _, _, _ = pred_tokens[i_p]
        log.append(f"[TP] ({p_tag}) PRED='{p_text}' ↔ GOLD='{g_text}' (sim={sim:.2f}, offset_diff={offset_dist})")

    # False Positives
    for i_p, (p_tag, p_text, _, _, _) in enumerate(pred_tokens):
        if i_p not in matched_pred:
            log.append(f"[FP] ({p_tag}) PRED='{p_text}'")

    # False Negatives
    for i_g, (g_tag, g_text, _, _, _) in enumerate(gold_tokens):
        if i_g not in matched_gold:
            log.append(f"[FN] ({g_tag}) GOLD='{g_text}'")

    fp = len(pred_tokens) - tp
    fn = len(gold_tokens) - tp
    return tp, fp, fn, log

# Now using offset-aware parser + matcher
def evaluate_all(pred_base_dir, gold_dir, threshold=0.5, max_offset_diff=15):
    results_total = defaultdict(Counter)
    results_by_tag = defaultdict(lambda: defaultdict(Counter))
    results_by_genre = defaultdict(lambda: defaultdict(Counter))

    for subdir in sorted(os.listdir(pred_base_dir)):
        pred_dir = os.path.join(pred_base_dir, subdir)
        model = subdir.replace("xml_output_", "").replace("balise_", "")
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

            #  Parse .ann with offsets
            pred_ents = parse_ann_surface_with_offsets(pred_file)
            gold_ents = parse_ann_surface_with_offsets(gold_file)

            #  Match with offset constraint
            tp, fp, fn, match_log = fuzzy_match_jaccard_detailed_with_offsets(
                gold_ents, pred_ents, threshold=threshold, max_offset_diff=max_offset_diff
            )

            # Logging
            log_path = os.path.join("match_logs", model)
            os.makedirs(log_path, exist_ok=True)
            with open(os.path.join(log_path, fname.replace(".ann", ".log")), "w", encoding="utf-8") as f:
                for line in match_log:
                    f.write(line + "\n")

            # Count by total
            results_total[model]["TP"] += tp
            results_total[model]["FP"] += fp
            results_total[model]["FN"] += fn

            # Count by tag
            for tag in TARGET_TAGS:
                gold_t = [(t, s, st, en) for (t, s, st, en) in gold_ents if t == tag]
                pred_t = [(t, s, st, en) for (t, s, st, en) in pred_ents if t == tag]
                tp_t, fp_t, fn_t, _ = fuzzy_match_jaccard_detailed_with_offsets(
                    gold_t, pred_t, threshold=threshold, max_offset_diff=max_offset_diff
                )
                results_by_tag[model][tag]["TP"] += tp_t
                results_by_tag[model][tag]["FP"] += fp_t
                results_by_tag[model][tag]["FN"] += fn_t

            # Count by genre
            genre = fname.split("0")[0].lower()
            results_by_genre[model][genre]["TP"] += tp
            results_by_genre[model][genre]["FP"] += fp
            results_by_genre[model][genre]["FN"] += fn

    return results_total, results_by_tag, results_by_genre


def compute_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def print_results(results_total, results_by_tag, results_by_genre):
    for model in results_total:
        print(f"\n\n=== Results for model: {model} ===")
        tp, fp, fn = results_total[model]["TP"], results_total[model]["FP"], results_total[model]["FN"]
        p, r, f1 = compute_f1(tp, fp, fn)
        print(f"\n→ Overall Micro F1: {f1:.3f} (P={p:.3f}, R={r:.3f}, TP={tp}, FP={fp}, FN={fn})")

        print("\n→ Micro F1 by Tag:")
        for tag in sorted(results_by_tag[model]):
            ttp = results_by_tag[model][tag]["TP"]
            tfp = results_by_tag[model][tag]["FP"]
            tfn = results_by_tag[model][tag]["FN"]
            p, r, f1 = compute_f1(ttp, tfp, tfn)
            print(f"{tag:6s}: F1={f1:.3f} (P={p:.3f}, R={r:.3f}, TP={ttp}, FP={tfp}, FN={tfn})")

        print("\n→ Micro F1 by Genre:")
        for genre in sorted(results_by_genre[model]):
            gtp = results_by_genre[model][genre]["TP"]
            gfp = results_by_genre[model][genre]["FP"]
            gfn = results_by_genre[model][genre]["FN"]
            p, r, f1 = compute_f1(gtp, gfp, gfn)
            print(f"{genre:12s}: F1={f1:.3f} (P={p:.3f}, R={r:.3f}, TP={gtp}, FP={gfp}, FN={gfn})")


def export_results_to_csv(results_total, results_by_tag, results_by_genre, output_dir="fuzzy_eval_csv", threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/micro_f1_summary_offset_jaccard.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Threshold", "TP", "FP", "FN", "Precision", "Recall", "F1"])
        for model in results_total:
            tp, fp, fn = results_total[model]["TP"], results_total[model]["FP"], results_total[model]["FN"]
            p, r, f1 = compute_f1(tp, fp, fn)
            writer.writerow([model, threshold, tp, fp, fn, round(p, 4), round(r, 4), round(f1, 4)])

    with open(f"{output_dir}/per_tag_micro_f1_offset_jaccard.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Tag", "Threshold", "TP", "FP", "FN", "Precision", "Recall", "F1"])
        for model in results_by_tag:
            for tag in sorted(results_by_tag[model]):
                tp = results_by_tag[model][tag]["TP"]
                fp = results_by_tag[model][tag]["FP"]
                fn = results_by_tag[model][tag]["FN"]
                p, r, f1 = compute_f1(tp, fp, fn)
                writer.writerow([model, tag, threshold, tp, fp, fn, round(p, 4), round(r, 4), round(f1, 4)])

    with open(f"{output_dir}/per_genre_micro_f1_offset_jaccard.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Genre", "Threshold", "TP", "FP", "FN", "Precision", "Recall", "F1"])
        for model in results_by_genre:
            for genre in sorted(results_by_genre[model]):
                tp = results_by_genre[model][genre]["TP"]
                fp = results_by_genre[model][genre]["FP"]
                fn = results_by_genre[model][genre]["FN"]
                p, r, f1 = compute_f1(tp, fp, fn)
                writer.writerow([model, genre, threshold, tp, fp, fn, round(p, 4), round(r, 4), round(f1, 4)])

if __name__ == "__main__":
    gold_dir = "/Users/ziyanxu/AGLAGLA/dataog"
    pred_base_dir = "/Users/ziyanxu/AGLAGLA/output_gml_xml/"
    threshold = 0.5
    max_offset_diff = 15

    total, by_tag, by_genre = evaluate_all(pred_base_dir, gold_dir, threshold, max_offset_diff)
    print_results(total, by_tag, by_genre)
    export_results_to_csv(total, by_tag, by_genre, threshold=threshold)
