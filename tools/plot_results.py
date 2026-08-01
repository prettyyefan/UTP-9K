from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent

METHODS = [
    "SegFormer",
    "Mask2Former",
    "SAM2 (Appearance Only)",
    "SAM2 + Naive Motion",
    "Ours (Full Model)",
]

TASK_A_METRICS = [
    "Overall mIoU",
    "Overall F1",
    "30 m IoU",
    "60 m IoU",
]

TASK_A_VALUES = np.array([
    [63.0, 64.8, 68.2, 56.6],
    [66.4, 68.1, 71.5, 60.1],
    [69.7, 71.8, 74.3, 64.1],
    [74.0, 75.9, 78.1, 68.9],
    [80.0, 81.4, 83.2, 76.0],
])

TASK_A_FP_METRICS = [
    "Moving-as-Obstacle FP",
    "Artifact-as-Obstacle FP",
]

TASK_A_FP_VALUES = np.array([
    [28.5, 22.1],
    [26.2, 19.8],
    [24.5, 18.5],
    [11.2, 16.3],
    [4.8, 5.2],
])

TASK_B_METRICS = [
    "Macro-F1",
    "F1-Stationary",
    "F1-Moving",
    "F1-Artifact / Shadow",
]

TASK_B_VALUES = np.array([
    [58.2, 64.8, 59.1, 50.7],
    [62.0, 68.1, 63.4, 54.5],
    [64.5, 71.8, 65.3, 56.4],
    [71.2, 75.9, 76.8, 60.9],
    [79.2, 81.6, 80.5, 75.5],
])


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 17,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 13,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def grouped_horizontal_bars(
    values: np.ndarray,
    metrics: list[str],
    title: str,
    xlabel: str,
    output_stem: str,
    xlim: tuple[float, float],
    lower_is_better: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.8))

    y = np.arange(len(metrics))
    n_methods = len(METHODS)
    group_height = 0.78
    bar_height = group_height / n_methods
    offsets = (np.arange(n_methods) - (n_methods - 1) / 2) * bar_height

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, method in enumerate(METHODS):
        is_ours = idx == n_methods - 1
        bars = ax.barh(
            y + offsets[idx],
            values[idx],
            height=bar_height * 0.92,
            label=method,
            color=default_colors[idx % len(default_colors)],
            alpha=0.95 if is_ours else 0.50,
            edgecolor="black" if is_ours else "none",
            linewidth=1.2 if is_ours else 0.0,
            hatch="///" if is_ours else None,
            zorder=3,
        )

        for bar, value in zip(bars, values[idx]):
            ax.text(
                value + (xlim[1] - xlim[0]) * 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}",
                va="center",
                ha="left",
                fontsize=10.5,
                fontweight="bold" if is_ours else "normal",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=12, fontweight="semibold")
    ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.35, zorder=0)

    ax.text(
        0.995,
        0.015,
        "Lower is better" if lower_is_better else "Higher is better",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        style="italic",
    )

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=3,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.2,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(OUT_DIR / f"{output_stem}.png", dpi=350, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT_DIR / f"{output_stem}.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main() -> None:
    configure_matplotlib()

    grouped_horizontal_bars(
        TASK_A_VALUES,
        TASK_A_METRICS,
        "Task A: Binary Obstacle Perception",
        "Performance (%)",
        "task_a_results",
        (52, 87),
        False,
    )

    grouped_horizontal_bars(
        TASK_A_FP_VALUES,
        TASK_A_FP_METRICS,
        "Task A: Region-Level False Positives",
        "False-positive rate (%)",
        "task_a_false_positive",
        (0, 32),
        True,
    )

    grouped_horizontal_bars(
        TASK_B_VALUES,
        TASK_B_METRICS,
        "Task B: Three-Way Role-Aware Perception",
        "F1 score (%)",
        "task_b_results",
        (48, 84),
        False,
    )


if __name__ == "__main__":
    main()
