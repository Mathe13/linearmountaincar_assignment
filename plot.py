#!/usr/bin/env python3
"""
plot.py - Generates boxplot comparing distributions of steps to 10 successes
for the 5 selected hyperparameter configurations.

Reads from results.csv / results.json and outputs boxplot.png.
"""

import os
import sys
import json
import csv
from typing import List, Dict, Any

# Environment fallback wrapper if executed with python lacking dependencies
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    import subprocess
    import shutil
    uv_bin = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
    if os.path.exists(uv_bin):
        cmd = [uv_bin, "run", "python"] + sys.argv
        sys.exit(subprocess.call(cmd))
    else:
        raise


def load_selected_configs(json_file: str = "top_5_configs.json", csv_file: str = "results.csv") -> List[Dict[str, Any]]:
    """Load the 5 selected configurations from top_5_configs.json or compute them from results.csv."""
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Neither {json_file} nor {csv_file} was found. Run run.py first!")

    # Parse results.csv
    data_by_cfg = {}
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lr = float(row["learning_rate"])
            nc = int(row["n_centers"])
            steps = float(row["steps_to_10_successes"])
            key = f"lr={lr}, nc={nc}"
            if key not in data_by_cfg:
                data_by_cfg[key] = {"config": key, "learning_rate": lr, "n_centers": nc, "steps": []}
            data_by_cfg[key]["steps"].append(steps)

    summaries = []
    for key, val in data_by_cfg.items():
        arr = np.array(val["steps"])
        summaries.append({
            "config": key,
            "learning_rate": val["learning_rate"],
            "n_centers": val["n_centers"],
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "steps": val["steps"],
        })

    # Default representative selection
    desired_keys = [
        "lr=0.05, nc=49",
        "lr=0.05, nc=25",
        "lr=0.01, nc=49",
        "lr=0.05, nc=9",
        "lr=0.01, nc=9",
    ]
    selected = [s for s in summaries if s["config"] in desired_keys]
    if len(selected) < 5:
        summaries.sort(key=lambda x: x["median"])
        selected = summaries[:3] + summaries[-2:]
    return selected


def generate_boxplot(selected_configs: List[Dict[str, Any]], output_path: str = "boxplot.png") -> None:
    """Generate and save publication-quality boxplot comparing 5 configurations."""
    # Sort configurations from best to worst median for clean presentation
    selected_configs = sorted(selected_configs, key=lambda x: x["median"])

    labels = []
    data = []
    for cfg in selected_configs:
        lr_str = f"α = {cfg['learning_rate']}"
        nc_str = f"{cfg['n_centers']} centros"
        label = f"{lr_str}\n({nc_str})"
        labels.append(label)
        data.append(cfg["steps"])

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Style elements
    box_colors = ["#2b5c8f", "#3b75af", "#5698d2", "#e28743", "#c94c4c"]
    median_color = "#d90429"
    mean_color = "#2b9348"

    # Boxplot
    bplot = ax.boxplot(
        data,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        tick_labels=labels,
        widths=0.55,
        medianprops=dict(color=median_color, linewidth=2.2, label="Mediana" if 0 == 0 else ""),
        meanprops=dict(color=mean_color, linewidth=2.0, linestyle="--", label="Média" if 0 == 0 else ""),
        flierprops=dict(marker="o", markersize=6, markerfacecolor="gray", alpha=0.7),
        whiskerprops=dict(linewidth=1.4, color="#333333"),
        capprops=dict(linewidth=1.4, color="#333333"),
    )

    for patch, color in zip(bplot["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_edgecolor("#1a1a1a")
        patch.set_linewidth(1.3)

    # Overlay jittered data points (scatter) for transparency
    np.random.seed(42)
    for i, pts in enumerate(data, start=1):
        x = np.random.normal(i, 0.04, size=len(pts))
        ax.scatter(x, pts, alpha=0.7, color="#1f1f1f", edgecolor="none", s=32, zorder=3)

    # Reference limit line at 50,000 steps
    ax.axhline(50000, color="#888888", linestyle=":", linewidth=1.2, label="Limite (50k passos)")

    # Labels and titles
    ax.set_title("Comparação de Desempenho do SARSA Linear no MountainCar\nPassos de Ambiente até Atingir 10 Sucessos (10 sementes)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Configuração de Hiperparâmetros (Taxa de Aprendizado α e Centros RBF)", fontsize=11, fontweight="semibold", labelpad=10)
    ax.set_ylabel("Passos de Ambiente até 10 Sucessos", fontsize=11, fontweight="semibold", labelpad=10)

    # Ticks and limits
    ax.set_ylim(0, 53000)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    # Custom legend
    handles = [
        plt.Line2D([0], [0], color=median_color, lw=2.2, label="Mediana"),
        plt.Line2D([0], [0], color=mean_color, lw=2.0, linestyle="--", label="Média"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f1f1f", markersize=6, label="Semente Individual (n=10)"),
        plt.Line2D([0], [0], color="#888888", lw=1.2, linestyle=":", label="Limite Máximo (50k passos)"),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.92, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Boxplot successfully generated and saved to: {output_path}")


def main():
    print("=" * 60)
    print("Generating Boxplot for Selected SARSA Configurations")
    print("=" * 60)

    selected = load_selected_configs()
    print(f"Loaded {len(selected)} configurations:")
    for i, s in enumerate(selected, 1):
        print(f"  {i}. {s['config']}: median={s['median']:.0f}, mean={s['mean']:.1f}")

    generate_boxplot(selected, "boxplot.png")


if __name__ == "__main__":
    main()

