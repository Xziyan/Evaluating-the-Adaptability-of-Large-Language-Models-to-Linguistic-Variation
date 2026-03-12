import os
import re
from collections import defaultdict, Counter
import csv

#this scirpt evaluate from the surface text output by llm
# it look at "type + string", not sensitive to offsets

# Only evaluate these 6 tags, normalized to uppercase
TARGET_TAGS = {"PERS", "LOC", "EVENT", "TIME", "PROD", "ORG"}

def normalize_tag(tag):
    """Keep only the first-level tag before the dot, and uppercase."""
    return tag.split('.')[0].upper()

def normalize_text(text):
    """Lowercase and normalize whitespace."""
    return ' '.join(text.lower().split())

def parse_ann_surface(path, normalize=True):
    """
    Parse a .ann file into a list of (tag, surface_text).
    Only keeps TARGET_TAGS if normalize is True.
    """
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
            tag = normalize_tag(raw_tag) if normalize else raw_tag
            if tag not in TARGET_TAGS:
                continue
            surface_text = parts[2]
            entities.append((tag, normalize_text(surface_text)))
    return entities

def evaluate_all_fuzzy(pred_base_dir, gold_dir):
    results_total = defaultdict(Counter)
    results_by_tag = defaultdict(lambda: defaultdict(Counter))
    results_by_genre = defaultdict(lambda: defaultdict(Counter))

    for subdir in sorted(os.listdir(pred_base_dir)):
        pred_dir = os.path.join(pred_base_dir, subdir)
        model = subdir.replace('xml_output_','').replace('balise_','')
        if not os.path.isdir(pred_dir):
            continue
        print(f"Evaluating subdir: {subdir}")

        for filename in os.listdir(pred_dir):
            if not filename.endswith(".ann"):
                continue
            pred_file = os.path.join(pred_dir, filename)
            gold_file = os.path.join(gold_dir, filename)
            if not os.path.exists(gold_file):
                print(f"[Warning] Gold file not found: {filename}")
                continue

            pred_ents = parse_ann_surface(pred_file)
            gold_ents = parse_ann_surface(gold_file)

            pred_counter = Counter(pred_ents)
            gold_counter = Counter(gold_ents)

            tp = sum((pred_counter & gold_counter).values())
            fp = sum((pred_counter - gold_counter).values())
            fn = sum((gold_counter - pred_counter).values())
            # 🔍 Log TP / FP / FN for error analysis
            log_dir = os.path.join("match_logs_surface", model)
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, filename.replace(".ann", ".log"))

            with open(log_file, "w", encoding="utf-8") as f:
                for ent in (pred_counter & gold_counter):
                    for _ in range(min(pred_counter[ent], gold_counter[ent])):
                        f.write(f"[TP] {ent[0]} '{ent[1]}'\n")
                for ent in (pred_counter - gold_counter):
                    for _ in range(pred_counter[ent]):
                        f.write(f"[FP] {ent[0]} '{ent[1]}'\n")
                for ent in (gold_counter - pred_counter):
                    for _ in range(gold_counter[ent]):
                        f.write(f"[FN] {ent[0]} '{ent[1]}'\n")
                        
            results_total[model]["TP"] += tp
            results_total[model]["FP"] += fp
            results_total[model]["FN"] += fn

            for tag in TARGET_TAGS:
                pred_tag_ents = Counter(e for e in pred_ents if e[0] == tag)
                gold_tag_ents = Counter(e for e in gold_ents if e[0] == tag)
                tp_tag = sum((pred_tag_ents & gold_tag_ents).values())
                fp_tag = sum((pred_tag_ents - gold_tag_ents).values())
                fn_tag = sum((gold_tag_ents - pred_tag_ents).values())
                results_by_tag[model][tag]["TP"] += tp_tag
                results_by_tag[model][tag]["FP"] += fp_tag
                results_by_tag[model][tag]["FN"] += fn_tag

            genre = filename.split("0")[0].lower()
            results_by_genre[model][genre]["TP"] += tp
            results_by_genre[model][genre]["FP"] += fp
            results_by_genre[model][genre]["FN"] += fn

    return results_total, results_by_tag, results_by_genre

def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall    = tp / (tp + fn) if tp + fn else 0.0
    f1        = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1

def print_results(results_total, results_by_tag, results_by_genre):
    for model in results_total:
        print('\n\nFuzzy results for model', model)
        print("\n===== Micro F1 over all 15 documents =====")
        tp, fp, fn = results_total[model]["TP"], results_total[model]["FP"], results_total[model]["FN"]
        p, r, f1 = compute_f1(tp, fp, fn)
        print(f"TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"Precision: {p:.3f}, Recall: {r:.3f}, F1 Micro: {f1:.3f}")

        print("\n===== Micro F1 by tag =====")
        for tag in sorted(results_by_tag[model]):
            tp, fp, fn = results_by_tag[model][tag]["TP"], results_by_tag[model][tag]["FP"], results_by_tag[model][tag]["FN"]
            p, r, f1 = compute_f1(tp, fp, fn)
            print(f"{tag:6s} - P: {p:.3f}, R: {r:.3f}, F1: {f1:.3f}  (TP: {tp}, FP: {fp}, FN: {fn})")

        print("\n===== Micro F1 by document genre =====")
        for genre in sorted(results_by_genre[model]):
            tp, fp, fn = results_by_genre[model][genre]["TP"], results_by_genre[model][genre]["FP"], results_by_genre[model][genre]["FN"]
            p, r, f1 = compute_f1(tp, fp, fn)
            print(f"{genre:12s} - P: {p:.3f}, R: {r:.3f}, F1: {f1:.3f}  (TP: {tp}, FP: {fp}, FN: {fn})")

def export_results_to_csv(results_total, results_by_tag, results_by_genre, output_dir="results_csv"):
    os.makedirs(output_dir, exist_ok=True)

    # Overall micro F1
    with open(os.path.join(output_dir, "micro_f1_total.csv"), "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "TP", "FP", "FN", "Precision", "Recall", "F1"])
        for model in results_total:
            tp, fp, fn = results_total[model]["TP"], results_total[model]["FP"], results_total[model]["FN"]
            p, r, f1 = compute_f1(tp, fp, fn)
            writer.writerow([model, tp, fp, fn, f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}"])

    # Per-tag F1
    with open(os.path.join(output_dir, "micro_f1_by_tag.csv"), "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ["Model", "Tag", "TP", "FP", "FN", "Precision", "Recall", "F1"]
        writer.writerow(header)
        for model in results_by_tag:
            for tag in sorted(results_by_tag[model]):
                tp = results_by_tag[model][tag]["TP"]
                fp = results_by_tag[model][tag]["FP"]
                fn = results_by_tag[model][tag]["FN"]
                p, r, f1 = compute_f1(tp, fp, fn)
                writer.writerow([model, tag, tp, fp, fn, f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}"])

    # Per-genre F1
    with open(os.path.join(output_dir, "micro_f1_by_genre.csv"), "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ["Model", "Genre", "TP", "FP", "FN", "Precision", "Recall", "F1"]
        writer.writerow(header)
        for model in results_by_genre:
            for genre in sorted(results_by_genre[model]):
                tp = results_by_genre[model][genre]["TP"]
                fp = results_by_genre[model][genre]["FP"]
                fn = results_by_genre[model][genre]["FN"]
                p, r, f1 = compute_f1(tp, fp, fn)
                writer.writerow([model, genre, tp, fp, fn, f"{p:.3f}", f"{r:.3f}", f"{f1:.3f}"])

if __name__ == "__main__":
    gold_dir = "/Users/ziyanxu/AGLAGLA/dataog"
    pred_base_dir = "/Users/ziyanxu/AGLAGLA/output_gml_xml/"

    total, by_tag, by_genre = evaluate_all_fuzzy(pred_base_dir, gold_dir)
    print_results(total, by_tag, by_genre)
    export_results_to_csv(total, by_tag, by_genre)