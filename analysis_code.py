"""
Analysis script for evaluating AI systems from an Excel file.

This script loads evaluation data for several AI systems (Witsy, Xplain,
Yandex, AnythingLLM and Onyx) from an Excel spreadsheet.  It computes
descriptive statistics for every quality metric (accuracy and recall from
two human raters, GPT‑4.1 and o4‑mini).  It then derives aggregate
metrics such as the mean accuracy across raters, the mean recall, the
harmonic mean (F1‑like measure) and the simple arithmetic mean of
accuracy and recall.  Correlation matrices and associated p‑values are
calculated for each system to quantify relationships between metrics.

Several plots are produced to visualize the results:

  * A bar chart comparing average accuracy and recall across systems.
  * A heatmap summarizing the four aggregate metrics for each system.
  * A boxplot showing the distribution of the harmonic mean per system.
  * A set of heatmaps (one per system) showing correlations between
    individual metrics.

Run this script with a Python interpreter (Python 3.8 or newer).  The
plot images will be saved into a directory named ``analysis_images``
within the current working directory.
"""

import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import pearsonr


# Default file name; update as needed to point to the Excel file.
DEFAULT_FILE_NAME = "Сравнение ответов разных AI систем -v2.xlsx"


def load_data(file_path: Path) -> pd.DataFrame:
    """Read the Excel file into a DataFrame.

    Args:
        file_path: Location of the Excel file.

    Returns:
        DataFrame with all columns.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_excel(file_path)
    return df


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for each metric and system.

    Args:
        df: Full dataframe containing metrics for all systems.

    Returns:
        A DataFrame with mean, median, standard deviation, minimum and
        maximum for every metric and system.  The returned frame has
        columns ['metric', 'mean', 'median', 'std', 'min', 'max', 'system'].
    """
    systems = ["Witsy", "Xplain", "Yandex", "AnythingLLM", "Onyx"]
    base_metrics = [
        "Точность Человек 1", "Полнота Человек 1",
        "Точность Человек 2", "Полнота Человек 2",
        "Точность gpt-4.1", "Полнота gpt-4.1",
        "Точность o4-mini", "Полнота o4-mini",
    ]
    suffixes = {
        "Witsy": "",
        "Xplain": ".1",
        "Yandex": ".2",
        "AnythingLLM": ".3",
        "Onyx": ".4",
    }
    stat_records = []
    for system in systems:
        suffix = suffixes[system]
        system_cols = [m + suffix for m in base_metrics]
        sub = df[system_cols]
        stats = sub.agg(["mean", "median", "std", "min", "max"]).T
        stats = stats.reset_index().rename(columns={"index": "metric"})
        stats["system"] = system
        stat_records.append(stats)
    summary = pd.concat(stat_records, ignore_index=True)
    return summary


def compute_aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregated metrics (mean accuracy, mean recall, harmonic and arithmetic mean) for each system.

    Args:
        df: Full dataframe.

    Returns:
        DataFrame with rows for each system and columns ['system', 'mean_accuracy',
        'mean_recall', 'mean_hmean', 'mean_arith_mean'].
    """
    systems = ["Witsy", "Xplain", "Yandex", "AnythingLLM", "Onyx"]
    suffixes = {
        "Witsy": "",
        "Xplain": ".1",
        "Yandex": ".2",
        "AnythingLLM": ".3",
        "Onyx": ".4",
    }
    results = []
    for system in systems:
        suff = suffixes[system]
        acc_cols = [
            "Точность Человек 1" + suff,
            "Точность Человек 2" + suff,
            "Точность gpt-4.1" + suff,
            "Точность o4-mini" + suff,
        ]
        rec_cols = [
            "Полнота Человек 1" + suff,
            "Полнота Человек 2" + suff,
            "Полнота gpt-4.1" + suff,
            "Полнота o4-mini" + suff,
        ]
        sub = df[acc_cols + rec_cols].copy()
        sub["agg_accuracy"] = sub[acc_cols].mean(axis=1)
        sub["agg_recall"] = sub[rec_cols].mean(axis=1)
        denom = sub["agg_accuracy"] + sub["agg_recall"]
        sub["agg_hmean"] = np.where(
            denom != 0,
            2 * sub["agg_accuracy"] * sub["agg_recall"] / denom,
            np.nan,
        )
        sub["agg_am"] = (sub["agg_accuracy"] + sub["agg_recall"]) / 2
        mean_acc = sub["agg_accuracy"].mean()
        mean_rec = sub["agg_recall"].mean()
        mean_hm = sub["agg_hmean"].mean()
        mean_am = sub["agg_am"].mean()
        results.append(
            {
                "system": system,
                "mean_accuracy": mean_acc,
                "mean_recall": mean_rec,
                "mean_hmean": mean_hm,
                "mean_arith_mean": mean_am,
            }
        )
    agg_df = pd.DataFrame(results)
    return agg_df


def compute_correlation_matrices(df: pd.DataFrame) -> dict:
    """Compute correlation and p-value matrices for each system.

    Args:
        df: Full dataframe.

    Returns:
        Dictionary mapping system name to a pair (corr_df, pval_df) where
        corr_df is the correlation matrix and pval_df holds the
        corresponding p‑values (Pearson correlation significance).
    """
    systems = ["Witsy", "Xplain", "Yandex", "AnythingLLM", "Onyx"]
    base_metrics = [
        "Точность Человек 1",
        "Полнота Человек 1",
        "Точность Человек 2",
        "Полнота Человек 2",
        "Точность gpt-4.1",
        "Полнота gpt-4.1",
        "Точность o4-mini",
        "Полнота o4-mini",
    ]
    suffixes = {
        "Witsy": "",
        "Xplain": ".1",
        "Yandex": ".2",
        "AnythingLLM": ".3",
        "Onyx": ".4",
    }
    corr_results = {}
    for system in systems:
        suffix = suffixes[system]
        system_cols = [m + suffix for m in base_metrics]
        sub = df[system_cols].dropna()
        corr = sub.corr()
        # compute p-values
        pval = pd.DataFrame(index=corr.index, columns=corr.columns, dtype=float)
        for col_i in system_cols:
            for col_j in system_cols:
                if col_i == col_j:
                    pval.loc[col_i, col_j] = 0.0
                else:
                    r, p = pearsonr(sub[col_i], sub[col_j])
                    pval.loc[col_i, col_j] = p
        corr_results[system] = (corr, pval)
    return corr_results


def plot_bar_accuracy_recall(agg_df: pd.DataFrame, output_dir: Path) -> None:
    """Create a bar chart comparing average accuracy and recall across systems."""
    plt.figure(figsize=(8, 6))
    data_melt = agg_df.melt(
        id_vars="system",
        value_vars=["mean_accuracy", "mean_recall"],
        var_name="metric",
        value_name="value",
    )
    sns.barplot(x="system", y="value", hue="metric", data=data_melt)
    plt.title("Средняя точность и полнота по системам")
    plt.ylabel("Значение (%)")
    plt.xlabel("Система")
    plt.legend(title="Метрика", labels=["Средняя точность", "Средняя полнота"])
    plt.ylim(0, 100)
    plt.tight_layout()
    out_path = output_dir / "bar_accuracy_recall.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_heatmap_aggregates(agg_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot a heatmap of aggregate metrics for each system."""
    plot_df = agg_df.set_index("system")[
        ["mean_accuracy", "mean_recall", "mean_hmean", "mean_arith_mean"]
    ]
    # Normalize for color scaling
    norm = (plot_df - plot_df.min()) / (plot_df.max() - plot_df.min())
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        norm,
        annot=plot_df.round(1),
        fmt=".1f",
        cmap="YlGnBu",
        cbar=True,
    )
    plt.title("Агрегированные метрики по системам")
    plt.ylabel("Система")
    plt.xlabel("Метрика")
    plt.tight_layout()
    out_path = output_dir / "heatmap_aggregates.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_boxplot_hmean(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot a boxplot of harmonic mean distributions per system."""
    systems = ["Witsy", "Xplain", "Yandex", "AnythingLLM", "Onyx"]
    suffixes = {
        "Witsy": "",
        "Xplain": ".1",
        "Yandex": ".2",
        "AnythingLLM": ".3",
        "Onyx": ".4",
    }
    records = []
    for system in systems:
        suff = suffixes[system]
        acc_cols = [
            "Точность Человек 1" + suff,
            "Точность Человек 2" + suff,
            "Точность gpt-4.1" + suff,
            "Точность o4-mini" + suff,
        ]
        rec_cols = [
            "Полнота Человек 1" + suff,
            "Полнота Человек 2" + suff,
            "Полнота gpt-4.1" + suff,
            "Полнота o4-mini" + suff,
        ]
        sub = df[acc_cols + rec_cols].copy()
        sub["agg_accuracy"] = sub[acc_cols].mean(axis=1)
        sub["agg_recall"] = sub[rec_cols].mean(axis=1)
        denom = sub["agg_accuracy"] + sub["agg_recall"]
        sub["agg_hmean"] = np.where(
            denom != 0,
            2 * sub["agg_accuracy"] * sub["agg_recall"] / denom,
            np.nan,
        )
        for value in sub["agg_hmean"].dropna():
            records.append({"system": system, "hmean": value})
    long_df = pd.DataFrame(records)
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="system", y="hmean", data=long_df, palette="pastel")
    plt.title("Распределение гармонического среднего (F1) по системам")
    plt.ylabel("Гармоническое среднее (%)")
    plt.xlabel("Система")
    plt.ylim(0, 100)
    plt.tight_layout()
    out_path = output_dir / "boxplot_hmean.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_correlation_heatmaps(corr_dict: dict, output_dir: Path) -> None:
    """Generate correlation heatmaps for each system."""
    for system, (corr_df, _) in corr_dict.items():
        plt.figure(figsize=(8, 6))
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        sns.heatmap(
            corr_df,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            cbar_kws={"label": "Корреляция"},
        )
        plt.title(f"Корреляция метрик – {system}")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        out_path = output_dir / f"corr_heatmap_{system.lower()}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()


def main() -> None:
    # Determine paths
    cwd = Path(os.getcwd())
    data_path = cwd / DEFAULT_FILE_NAME
    output_dir = cwd / "analysis_images"
    output_dir.mkdir(exist_ok=True)
    # Load data
    df = load_data(data_path)
    # Compute statistics
    summary_stats = compute_summary_stats(df)
    summary_stats.to_csv(output_dir / "summary_statistics.csv", index=False)
    # Aggregates
    agg_df = compute_aggregate_metrics(df)
    agg_df.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    # Correlations
    corr_dict = compute_correlation_matrices(df)
    # Plots
    plot_bar_accuracy_recall(agg_df, output_dir)
    plot_heatmap_aggregates(agg_df, output_dir)
    plot_boxplot_hmean(df, output_dir)
    plot_correlation_heatmaps(corr_dict, output_dir)
    # Print simple ranking
    sorted_df = agg_df.sort_values(by="mean_hmean", ascending=False)
    print("Ranking (best to worst) based on harmonic mean of accuracy and recall:")
    print(sorted_df[["system", "mean_hmean"]])


if __name__ == "__main__":
    main()