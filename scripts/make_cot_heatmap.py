#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build strict vs fuzzy heatmaps (left/right blocks) for CoT experiments.

Folder layout assumed (examples):
OUT_DIR_COT_POST_PROCESSED/
  OUT_DIR_COT_ZERO_STRICT/
    eval_strict/per_genre.csv
  OUT_DIR_COT_ZERO_FUZZY/
    eval_fuzzy_0.5/per_genre.csv
  OUT_DIR_COT_ONE_STRICT/ ...
  OUT_DIR_COT_ONE_FUZZY/ ...
  OUT_DIR_COT_FEW_STRICT/ ...
  OUT_DIR_COT_FEW_FUZZY/ ...
  OUT_DIR_COT_FEW_PLUS_STRICT/ ...
  OUT_DIR_COT_FEW_PLUS_FUZZY/ ...

Per-genre CSV must have columns: genre, f1 (or F1). Genre 'multi' is excluded.
"""

import argparse
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROMPT_ORDER = ["ZERO", "ONE", "FEW", "FEW_PLUS"]

def infer_prompt(name: str) -> str:
    n = name.upper()
    if "FEW_PLUS" in n or "FEWPLUS" in n or "FEW_PLUS_" in n:
        return "FEW_PLUS"
    if re.search(r"\bFEW\b", n):
        return "FEWSHOT"
    if re.search(r"\bONE\b", n):
        return "ONESHOT"
    if re.search(r"\bZERO\b", n):
        return "ZEROSHOT"
    # fallback: look for 'COT_*' token
    m = re.search(r"COT_([A-Z_]+)", n)
    return m.group(1) if m else "UNKNOWN"

def infer_mode(name: str) -> str:
    n = name.upper()
    return "FUZZY" if "FUZZY" in n else "STRICT"

def find_per_genre_csv(run_dir: Path) -> Path | None:
    # look for eval_* subdir then per_genre.csv inside
    for eval_dir in sorted(run_dir.glob("eval_*")):
        csv_path = eval_dir / "per_genre.csv"
        if csv_path.exists():
            return csv_path
    # sometimes per_genre.csv is directly under the run_dir
    direct = run_dir / "per_genre.csv"
    return direct if direct.exists() else None

def load_one(run_dir: Path) -> pd.DataFrame | None:
    csv_path = find_per_genre_csv(run_dir)
    if not csv_path:
        return None
    df = pd.read_csv(csv_path)
    # normalize columns
    cols = {c.lower(): c for c in df.columns}
    if "f1" in cols:
        f1col = cols["f1"]
    elif "F1" in df.columns:
        f1col = "F1"
    else:
        raise ValueError(f"No F1 column in {csv_path}")
    gcol = "genre" if "genre" in df.columns else "Genre" if "Genre" in df.columns else None
    if gcol is None:
        raise ValueError(f"No genre column in {csv_path}")

    # clean genre, exclude multi
    df["_genre_norm"] = df[gcol].astype(str).str.strip().str.lower()
    df = df[~df["_genre_norm"].eq("multi")]
    df["_genre_display"] = df[gcol].astype(str).str.strip()

    # keep essentials
    out = df[["_genre_display", f1col]].copy()
    out.columns = ["genre", "f1"]
    return out

def gather(base_dir: Path) -> pd.DataFrame:
    rows = []
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        prompt = infer_prompt(child.name)
        mode = infer_mode(child.name)
        csv_df = load_one(child)
        if csv_df is None:
            continue
        for _, r in csv_df.iterrows():
            rows.append({
                "prompt": prompt,
                "mode": mode,
                "genre": r["genre"],
                "f1": float(r["f1"])
            })
    if not rows:
        raise RuntimeError(f"No per_genre.csv found under {base_dir}")
    df = pd.DataFrame(rows)
    # enforce prompt ordering
    df["prompt"] = pd.Categorical(df["prompt"], categories=PROMPT_ORDER, ordered=True)
    # stable genre order: alphabetic by default
    df["genre"] = df["genre"].astype(str)
    return df

def to_matrix(df: pd.DataFrame, mode: str) -> tuple[np.ndarray, list[str], list[str]]:
    sub = df[df["mode"] == mode]
    if sub.empty:
        # produce empty matrix
        return np.zeros((0,0)), [], []
    piv = sub.pivot_table(index="prompt", columns="genre", values="f1", aggfunc="mean")
    piv = piv.sort_index()  # by PROMPT_ORDER categorical
    piv = piv.reindex(index=PROMPT_ORDER, fill_value=np.nan)
    # sort genres alphabetically
    piv = piv.reindex(sorted(piv.columns), axis=1)
    return piv.to_numpy(), list(piv.index), list(piv.columns)

def plot_two_heatmaps(strict_mat, strict_y, strict_x, fuzzy_mat, fuzzy_y, fuzzy_x, out_path: Path, title: str):
    genres = sorted(set(strict_x) | set(fuzzy_x))
    prompts = PROMPT_ORDER  # y-axis union (fixed order)

    def pad(mat, ylabels, xlabels):
        # build full matrix with NaNs, then fill
        full = np.full((len(prompts), len(genres)), np.nan, dtype=float)
        ymap = {p:i for i,p in enumerate(prompts)}
        xmap = {g:i for i,g in enumerate(genres)}
        for yi, p in enumerate(ylabels):
            for xi, g in enumerate(xlabels):
                full[ymap[p], xmap[g]] = mat[yi, xi]
        return full

    strict_full = pad(strict_mat, strict_y, strict_x)
    fuzzy_full = pad(fuzzy_mat, fuzzy_y, fuzzy_x)

    # single figure with 2 subplots (left strict, right fuzzy)
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 2 + len(genres)), 4.5), constrained_layout=True)

    for ax, mat, block_title in zip(axes, (strict_full, fuzzy_full), ("Strict", "Fuzzy")):
        im = ax.imshow(mat, aspect="auto")
        ax.set_title(block_title)
        ax.set_xticks(range(len(genres)))
        ax.set_xticklabels(genres, rotation=45, ha="right")
        ax.set_yticks(range(len(prompts)))
        ax.set_yticklabels(prompts)
        # annotate values
        for i in range(len(prompts)):
            for j in range(len(genres)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
        # add colorbar per block
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved heatmap: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, help="Path to OUT_DIR_COT_POST_PROCESSED")
    ap.add_argument("--out", default="cot_strict_fuzzy_heatmap.png", help="Output image path")
    ap.add_argument("--title", default="CoT NER F1 by Prompt × Genre (Strict vs Fuzzy)")
    args = ap.parse_args()

    base = Path(args.base_dir)
    df = gather(base)
    # build matrices
    strict_mat, strict_y, strict_x = to_matrix(df, "STRICT")
    fuzzy_mat,  fuzzy_y,  fuzzy_x  = to_matrix(df, "FUZZY")

    # save combined CSV for reference
    combined_csv = Path(args.out).with_suffix(".csv")
    df.sort_values(["mode", "prompt", "genre"]).to_csv(combined_csv, index=False)
    print(f"Wrote combined table: {combined_csv}")

    plot_two_heatmaps(strict_mat, strict_y, strict_x, fuzzy_mat, fuzzy_y, fuzzy_x,
                      out_path=Path(args.out), title=args.title)

if __name__ == "__main__":
    main()
