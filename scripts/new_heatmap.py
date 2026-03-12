import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#CSV = "/Users/ziyanxu/AGLAGLA/llm_apis/srtict_on_pprocessed_with_alignement/eval_per_model_prompt_genre.csv"
#CSV = "/Users/ziyanxu/AGLAGLA/llm_apis/strict_on_raw_without_alignement/eval_per_model_prompt_genre.csv"
# Orders for tidy alignment
models = ['DeepSeekR1', 'DeepSeekV3', 'Llama3', 'Nemotron', 'Qwen3']
genres = ['encyclopedia','information','multi','poetry','prose','spoken']
prompts_order = ['Fewshot','Oneshot','Zeroshot']

df = pd.read_csv(CSV)
df.columns = [c.strip().lower() for c in df.columns]
df['model'] = pd.Categorical(df['model'], categories=models, ordered=True)
df['genre'] = pd.Categorical(df['genre'], categories=genres, ordered=True)
df['prompt'] = pd.Categorical(df['prompt'], categories=prompts_order, ordered=True)

# Consistent model colors (for ytick labels)
color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', []) 
model_colors = {m: color_cycle[i % len(color_cycle)] for i, m in enumerate(models)}

def plot_heatmap_for_prompt(prompt_name: str, save_path: str):
    sub = df[df['prompt'] == prompt_name]
    pivot = sub.pivot_table(index='model', columns='genre', values='f1')
    pivot = pivot.loc[models, genres]

    fig = plt.figure(figsize=(6, 3.6))  # compact
    ax = fig.add_subplot(111)
    im = ax.imshow(pivot.values, aspect='auto', vmin=0, vmax=1, cmap="YlGnBu")

    ax.set_xticks(np.arange(len(genres)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(genres, rotation=30, ha='right')
    ax.set_yticklabels(models)

    # color ytick labels by model
    for tick, m in zip(ax.get_yticklabels(), models):
        tick.set_color(model_colors[m])
        tick.set_fontweight('semibold') 

    ax.set_xlabel('Genre')
    ax.set_ylabel('Model')
    ax.set_title(f'{prompt_name} — F1 heatmap')

    # write numbers in cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if pd.notna(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('F1')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

for p in prompts_order:
    plot_heatmap_for_prompt(p, f"heatmap_{p.lower()}.png")

# Optional: tiny legend image
fig = plt.figure(figsize=(3, 1.6))
ax = fig.add_subplot(111)
ax.axis('off')
for i, (m, col) in enumerate(model_colors.items()):
    ax.text(0.05, 0.85 - i*0.18, m, color=col, fontsize=10, fontweight='semibold')
fig.tight_layout()
fig.savefig("model_color_legend.png", dpi=300, bbox_inches='tight')
plt.close(fig)
