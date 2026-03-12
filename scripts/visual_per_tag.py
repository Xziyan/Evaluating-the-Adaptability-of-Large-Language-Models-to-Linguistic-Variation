import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Load your file (expects columns: tag, model, f1)
#df = pd.read_csv("/Users/ziyanxu/AGLAGLA/llm_apis/srtict_on_pprocessed_with_alignement/eval_per_tag.csv")
#df = pd.read_csv("/Users/ziyanxu/AGLAGLA/llm_apis/strict_on_raw_without_alignement/eval_per_tag.csv")
df = pd.read_csv("/Users/ziyanxu/AGLAGLA/llm_apis/eval_per_tag.csv")
df.columns = [c.strip().lower() for c in df.columns]  # normalize

# 2) Extract model/prompt from "model" like "DeepSeekR1_Fewshot"
def split_model_prompt(s):
    parts = str(s).split("_")
    if len(parts) >= 2:
        return parts[0], "_".join(parts[1:])
    return s, ""

mp = df["model"].apply(split_model_prompt)
df["model_name"] = mp.apply(lambda x: x[0])
df["prompt"] = mp.apply(lambda x: x[1])

# 3) Order columns (optional but nice)
models_pref  = ['DeepSeekR1', 'DeepSeekV3', 'Llama3', 'Nemotron', 'Qwen3']
prompts_pref = ['Fewshot', 'Oneshot', 'Zeroshot']

labels = []
for m in models_pref:
    for p in prompts_pref:
        label = f"{m}_{p}"
        if label in df["model"].unique():
            labels.append(label)
labels += [lab for lab in df["model"].unique() if lab not in labels]  # append any extras

# 4) Pivot to Tag × Model_Prompt
pivot = df.pivot_table(index="tag", columns="model", values="f1", aggfunc="mean")
pivot = pivot.reindex(columns=labels).sort_index()

# 5) Plot heatmap with YlGnBu
fig_w = max(8, 0.55 * pivot.shape[1] + 2)
fig_h = max(4, 0.60 * pivot.shape[0] + 2)
fig = plt.figure(figsize=(fig_w, fig_h))
ax = fig.add_subplot(111)

im = ax.imshow(pivot.values, aspect='auto', cmap="YlGnBu", vmin=0, vmax=1)

ax.set_xticks(np.arange(pivot.shape[1]))
ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
ax.set_yticks(np.arange(pivot.shape[0]))
ax.set_yticklabels(pivot.index)

ax.set_xlabel("Model_Prompt")
ax.set_ylabel("Tag")
ax.set_title("F1 by Tag × Model_Prompt Variation Heatmap")

# Contrast-aware annotations
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = pivot.values[i, j]
        if pd.notna(val):
            txt_color = "white" if val >= 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=txt_color)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("F1")

fig.tight_layout()
plt.savefig("tag_modelprompt_heatmap.png", dpi=300, bbox_inches='tight')
plt.close(fig)
