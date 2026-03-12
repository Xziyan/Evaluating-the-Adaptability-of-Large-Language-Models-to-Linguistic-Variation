import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取CSV文件
#csv_path = "/Users/ziyanxu/AGLAGLA/aligned_spaces_ecomercial_output/eval_per_model_prompt_genre.csv"  
df = pd.read_csv(csv_path)

# 按 genre 计算整体平均 F1，用于排序（难度高 = 分数低）
genre_difficulty = df.groupby("Genre")["F1"].mean().sort_values()
sorted_genres = genre_difficulty.index.tolist()  # genre 从难到易排序

# 创建 FacetGrid：每个 Prompt 一张热图
g = sns.FacetGrid(
    df,
    col="Prompt",
    col_order=sorted(df["Prompt"].unique()),
    height=5,
    aspect=1.2
)

# 每个子图绘制一个热图（Model × Genre, 值是 F1）
def draw_heatmap(data, **kwargs):
    pivot = data.pivot(index="Model", columns="Genre", values="F1")
    pivot = pivot[sorted_genres]  # genre 列按照整体难度排序
    sns.heatmap(
        pivot,
        annot=True,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        cbar=False,
        fmt=".2f",
        linewidths=0.5,
        linecolor="lightgray",
        **kwargs
    )

g.map_dataframe(draw_heatmap)
g.set_titles(col_template="{col_name}")
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("F1 Score Heatmaps by Prompt (x=Genre, y=Model)", fontsize=16)

# 保存为 PNG 图像（也可以改为 PDF）
plt.savefig("f1_heatmaps.png", dpi=300, bbox_inches="tight")
plt.show()
