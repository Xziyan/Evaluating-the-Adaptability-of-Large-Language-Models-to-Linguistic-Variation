import os
import sys
from collections import defaultdict, Counter
import csv
import glob
import argparse

# 只保留6个我们评估的标签
TARGET_TAGS = {"PERS", "LOC", "EVENT", "TIME", "PROD", "ORG"}

def normalize_tag(tag):
    return tag.split('.')[0].upper()

def normalize_text(text):
    return ' '.join(text.lower().split())

# 读取 .ann 文件，返回符合条件的实体列表
def load_ann_file(file_path):
    entities = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('T'):
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    print(f"Invalid line in .ann file: {line.strip()} Not enough tabulations")
                    continue
                tag_info, span_text = parts[1], parts[2]
                tag_info_parts = tag_info.split()
                if len(tag_info_parts) < 3:
                    print(f"Invalid tag info in .ann file: {tag_info} Not enough tag info parts")
                    continue
                tag = normalize_tag(tag_info_parts[0])
                start = int(tag_info_parts[1])
                end = int(tag_info_parts[2])
                if tag in TARGET_TAGS:
                    entities.append((tag, start, end, normalize_text(span_text)))
    return entities

# 严格匹配实体，返回列表形式的 TP, FP, FN 实体（供后续分析）
def analyze_differences_strict(pred, gold):
    tp, fp, fn = [], [], []
    gold_used = set()
    pred_used = set()

    for i, p in enumerate(pred):
        
        for j, g in enumerate(gold):
            if j in gold_used:
                continue
            if p[0:2] == g[0:2]:  # tag 和位置完全匹配
                tp.append(p)
                gold_used.add(j)
                pred_used.add(i)
                break
    for i, p in enumerate(pred):
        if i not in pred_used:
            fp.append(p)
    for j, g in enumerate(gold):
        if j not in gold_used:
            fn.append(g)
    return tp, fp, fn

def analyze_differences_fuzzy(pred, gold,t=0.5):
    tp, fp, fn = [], [], []
    gold_used = set()
    pred_used = set()

    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            if j in gold_used:
                continue
            # perform fuzzy matching based on tag and position
            # using jaccard similarity for overlap
            #
            jaccard_similarity = len(set(range(p[1], p[2])) & set(range(g[1], g[2])))/ len(set(range(p[1], p[2])) | set(range(g[1], g[2])))
            if jaccard_similarity >= t and p[0] == g[0]:  # tag 相同且位置有重叠hhhhhhhhhhhhh
                tp.append(p)
                gold_used.add(j)
                pred_used.add(i)
                break
    for i, p in enumerate(pred):
        if i not in pred_used:
            fp.append(p)
    for j, g in enumerate(gold):
        if j not in gold_used:
            fn.append(g)
    return tp, fp, fn

# 生成通用 CSV 输出
def write_csv_dict(data_dict, file_name, level_name):
    with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [level_name, 'Model', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for model in sorted(data_dict):
            for level in sorted(data_dict[model]):
                counts = data_dict[model][level]
                tp = counts["TP"]
                fp = counts["FP"]
                fn = counts["FN"]
                prec = tp / (tp + fp) if tp + fp > 0 else 0.0
                rec = tp / (tp + fn) if tp + fn > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
                writer.writerow({
                    level_name: level,
                    'Model': model,
                    'TP': tp,
                    'FP': fp,
                    'FN': fn,
                    'Precision': f"{prec:.3f}",
                    'Recall': f"{rec:.3f}",
                    'F1': f"{f1:.3f}"
                })

# 主评估逻辑
def evaluate(pred_folder, gold_folder, fuzzy=False):
    results_total = defaultdict(lambda: Counter())
    results_tag = defaultdict(lambda: defaultdict(Counter))
    results_document = defaultdict(lambda: defaultdict(Counter))
    results_genre = defaultdict(lambda: defaultdict(Counter))
    structured_records = []

    # 清空旧的 error 分析输出
    os.makedirs("analysis_outputs", exist_ok=True)
    for old_file in glob.glob("analysis_outputs/*.tsv"):
        os.remove(old_file)

    subfolders = sorted([f for f in os.listdir(pred_folder) if not f.startswith('.')])

    for model_prompt_name in subfolders:
        model_path = os.path.join(pred_folder, model_prompt_name)
        if not os.path.isdir(model_path):
            continue

        # 假设目录名格式为 "Qwen3_fewshot"

        m_p_n_short = model_prompt_name.replace('xml_output_','').replace('balise_','')
        
        parts = m_p_n_short.split('_')
        for i in range(len(parts)):
            parts[i] = parts[i][0].upper() + parts[i][1:]  # 首字母大写
        prompt_name = parts[0]
        model_name = ''.join(parts[1:])

        model_prompt_name = f"{model_name}_{prompt_name}"


        print(f"📦 Evaluating: {model_name} with prompt [{prompt_name}]")

        # 准备 error 分析文件 
        # !!!有问题，erro分析文件并不是根据每个文件生成的，而是整个文件夹!!!
        error_file_path = os.path.join("analysis_outputs", f"{model_prompt_name}.tsv")
        with open(error_file_path, 'w', encoding='utf-8') as ef:
            ef.write("Type\tTag\tStart\tEnd\tText\n")

        for file_name in sorted(os.listdir(model_path)):
            print(f"  📄 Processing: {file_name} !!!!!!!!!!!!!!")
            if not file_name.endswith('.ann'):
                continue

            # genre = 文件名前缀（如 information01-xxx.ann → information）
            genre = file_name.split('0')[0].lower()

            pred_file = os.path.join(model_path, file_name)
            gold_file = os.path.join(gold_folder, file_name)

            if not os.path.exists(gold_file):
                print(f"⚠️ Missing gold file: {file_name}")
                continue

            pred_entities = load_ann_file(pred_file)
            gold_entities = load_ann_file(gold_file)

            print(f"    Predicted entities: {len(pred_entities)}")

            # 严格对齐
            if not fuzzy:
                tp_list, fp_list, fn_list = analyze_differences_strict(pred_entities, gold_entities)
            else:
                tp_list, fp_list, fn_list = analyze_differences_fuzzy(pred_entities, gold_entities)
            tp, fp, fn = len(tp_list), len(fp_list), len(fn_list)

            # 错误分析写入
            with open(error_file_path, 'a', encoding='utf-8') as ef:
                for tag, start, end, text in tp_list:
                    ef.write(f"TP\t{tag}\t{start}\t{end}\t{text}\n")
                for tag, start, end, text in fp_list:
                    ef.write(f"FP\t{tag}\t{start}\t{end}\t{text}\n")
                for tag, start, end, text in fn_list:
                    ef.write(f"FN\t{tag}\t{start}\t{end}\t{text}\n")

            # 汇总统计
            results_total[model_prompt_name].update({"TP": tp, "FP": fp, "FN": fn})
            results_document[model_prompt_name][file_name].update({"TP": tp, "FP": fp, "FN": fn})
            results_genre[model_prompt_name][genre].update({"TP": tp, "FP": fp, "FN": fn})

            # 每个标签的统计
            for tag in TARGET_TAGS:
                pred_t = [e for e in pred_entities if e[0] == tag]
                gold_t = [e for e in gold_entities if e[0] == tag]
                if not fuzzy:
                    t_tp, t_fp, t_fn = analyze_differences_strict(pred_t, gold_t)
                else: 
                    t_tp, t_fp, t_fn = analyze_differences_fuzzy(pred_t, gold_t)
                results_tag[model_prompt_name][tag].update({
                    "TP": len(t_tp), "FP": len(t_fp), "FN": len(t_fn)
                })

    # 写入结构化 per-model-prompt-genre 表格
    for model_prompt in results_genre:
        model_name, prompt_name = model_prompt.split('_', 1) if '_' in model_prompt else (model_prompt, "unknown")
        for genre in results_genre[model_prompt]:
            counts = results_genre[model_prompt][genre]
            tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
            prec = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
            structured_records.append({
                "Model": model_name,
                "Prompt": prompt_name,
                "Genre": genre,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Precision": f"{prec:.3f}",
                "Recall": f"{rec:.3f}",
                "F1": f"{f1:.3f}"
            })

    with open("eval_per_model_prompt_genre.csv", 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["Model", "Prompt", "Genre", "TP", "FP", "FN", "Precision", "Recall", "F1"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in structured_records:
            writer.writerow(row)

    # 写其他结果：按标签、按 genre、按文档、按模型
    write_csv_dict(results_tag, "eval_per_tag.csv", "Tag")
    write_csv_dict(results_genre, "eval_per_genre.csv", "Genre")
    write_csv_dict(results_document, "eval_per_document.csv", "Document")
    write_csv_dict(results_total, "eval_per_model.csv", "Model")

# 运行主逻辑
if __name__ == "__main__":
    pred_folder = "/Users/ziyanxu/AGLAGLA/"  
    gold_folder = "/Users/ziyanxu/AGLAGLA/dataog"           
    parser = argparse.ArgumentParser (prog="Do evaluation on hardocoded folder")
    parser.add_argument('--fuzzy', action='store_true', help='Use fuzzy matching instead of strict matching', default=False)
    prgs = parser.parse_args(sys.argv[1:])
    evaluate(pred_folder, gold_folder, fuzzy=prgs.fuzzy)
