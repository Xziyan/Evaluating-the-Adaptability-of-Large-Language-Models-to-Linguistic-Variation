import os
import json
import re
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

TOP_LEVEL_TAGS = ["PERS", "LOC", "PROD", "ORG", "EVENT", "TIME"]

# 从 LLM 输出中提取 JSON 数据块（考虑到了推理+JSON的输出结构）
def extract_json_from_llm_output(text):
    """Extracts the JSON block from an LLM output that may have reasoning first."""
    try:
        json_str = re.search(r"\{[\s\S]*\}", text).group()
        return json.loads(json_str)
    except Exception as e:
        print("❌ Failed to extract JSON:", e)
        return {}

def load_gold_entities(path):
    with open(path, encoding='utf-8') as f:
        gold = json.load(f)
    gold_entities = set()
    for cat, ents in gold.items():
        for ent in ents:
            gold_entities.add((cat, ent["entity"].strip()))
    return gold_entities

# Load LLM-predicted entities and track failed files
def load_llm_entities_with_logging(path, failed_files):
    with open(path, encoding='utf-8') as f:
        content = f.read()
    llm_data = extract_json_from_llm_output(content)
    if llm_data is None:
        failed_files.append(path.name)
        return None
    llm_entities = set()
    for cat, ents in llm_data.items():
        if cat not in TOP_LEVEL_TAGS:
            continue
        for ent in ents:
            llm_entities.add((cat, ent["entity"].strip()))
    return llm_entities

# ✅ 第一种评估方式：整体评估所有文档上的性能（总体 micro-F1 等指标）
def evaluate_predictions(gold_folder, llm_folder):
    all_true = []
    all_pred = []
    type_metrics = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

    for file in os.listdir(gold_folder):
        if not file.endswith(".json"):
            continue

        gold_path = os.path.join(gold_folder, file)
        llm_path = os.path.join(llm_folder, file)

        if not os.path.exists(llm_path):
            print(f"⚠️ Missing LLM output for {file}")
            continue

        gold_set = load_gold_entities(gold_path)
        pred_set = load_llm_entities_with_logging(llm_path,[])

        for tag in TOP_LEVEL_TAGS:
            gold_tag = {(t, e) for (t, e) in gold_set if t == tag}
            pred_tag = {(t, e) for (t, e) in pred_set if t == tag}

            tp = gold_tag & pred_tag
            fp = pred_tag - gold_tag
            fn = gold_tag - pred_tag

            type_metrics[tag]["TP"] += len(tp)
            type_metrics[tag]["FP"] += len(fp)
            type_metrics[tag]["FN"] += len(fn)

            all_true.extend([1] * len(tp) + [1] * len(fn))
            all_pred.extend([1] * len(tp) + [0] * len(fn))
            all_true.extend([0] * len(fp))
            all_pred.extend([1] * len(fp))
    # 输出每种标签（PERS、LOC等）的评估结果
    print("\n📊 Evaluation Results (by entity type):")
    for tag in TOP_LEVEL_TAGS:
        tp = type_metrics[tag]["TP"]
        fp = type_metrics[tag]["FP"]
        fn = type_metrics[tag]["FN"]

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

        print(f"{tag:<6} | P: {precision:.2f}  R: {recall:.2f}  F1: {f1:.2f}  (TP={tp}, FP={fp}, FN={fn})")
     # 计算总体 Precision/Recall/F1 分数（micro-F1）
    overall_p, overall_r, overall_f1, _ = precision_recall_fscore_support(all_true, all_pred, average='binary')
    print("\n🔢 Overall Scores:")
    print(f"Precision: {overall_p:.2f}")
    print(f"Recall:    {overall_r:.2f}")
    print(f"F1 Score:  {overall_f1:.2f}")

# Main evaluation function: compare predictions vs. gold per document type# ✅ 第二种评估方式：针对特定文档类型（如poetry）进行整体评估
def evaluate_document_type_overall(gold_dir, llm_dir, target_doc_type) -> dict:
    gold_dir = Path(gold_dir)
    llm_dir = Path(llm_dir)

    gold_all = set()
    pred_all = set()
    failed_files = []

    for file in gold_dir.glob(f"{target_doc_type}*.json"):
        gold_path = file
        llm_path = llm_dir / file.name

        gold_entities = load_gold_entities(gold_path)
        gold_all.update(gold_entities)

        if not llm_path.exists():
            failed_files.append(file.name)
            continue

        pred_entities = load_llm_entities_with_logging(llm_path, failed_files)
        if pred_entities is not None:
            pred_all.update(pred_entities)

    tp = gold_all & pred_all
    fp = pred_all - gold_all
    fn = gold_all - pred_all

    prec = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 0
    rec = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    print(f"📄 Overall LLM NER Performance on {target_doc_type} documents:")
    print(f"Precision: {prec:.2f}")
    print(f"Recall:    {rec:.2f}")
    print(f"F1 Score:  {f1:.2f}")
    print(f"TP: {len(tp)} | FP: {len(fp)} | FN: {len(fn)}")
    if failed_files:
        print("\n❌ Failed files:")
        for f in failed_files:
            print(" -", f)

    return {
        "doc_type": target_doc_type,
        "precision": round(prec, 2),
        "recall": round(rec, 2),
        "f1": round(f1, 2),
        "TP": len(tp),
        "FP": len(fp),
        "FN": len(fn),
        "failed_files": failed_files
    }
# 对每个文档类型都调用一次上面这个评估函数
def evaluate_all_types(gold_dir, llm_dir, doc_types: list[str]):
    results = []
    for doc_type in doc_types:
        result = evaluate_document_type_overall(gold_dir, llm_dir, doc_type)
        results.append(result)

    print("\n📊 Summary per Document Type:")
    print(f"{'Type':<12} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    for r in results:
        print(f"{r['doc_type']:<12} {r['precision']:<10} {r['recall']:<10} {r['f1']:<10}")

    return results


if __name__ == "__main__":
    gold_dir = "/Users/ziyanxu/AGLAGLA/converted_gold_formatjson"
    llm_dir = "/Users/ziyanxu/AGLAGLA/output_gml/output_CoT_4shots_qwen_qwq"
    #llm_dir = "/Users/ziyanxu/AGLAGLA/output_gml/output_CoT_simple_qwen_qwq"
    #llm_dir = "/Users/ziyanxu/AGLAGLA/output_gml/output_CoT_zeroshot_deepseek_r1"
    #llm_dir = "/Users/ziyanxu/AGLAGLA/output_gml/output_complex_4shots_deepseek_r1"
    #llm_dir = "/Users/ziyanxu/AGLAGLA/output_gml/output_simple_deepseek_r1"

    eva_results ="/Users/ziyanxu/AGLAGLA/eva_results_llm"
    # 第一种：整体评估
    evaluate_predictions(gold_dir, llm_dir)
    # 第二种：对每种文档类型单独评估
    doc_types = ["encyclopedia", "information", "multi", "poetry", "prose", "spoken"]
    
    results= evaluate_all_types(gold_dir, llm_dir, doc_types)
    
    #save csv results
    df = pd.DataFrame(results)
    output_csv_path = Path(eva_results) / "eva_CoT_4shots_qwen_qwq.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Evaluation results saved to {output_csv_path}")


