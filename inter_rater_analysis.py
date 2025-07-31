"""
Inter‑rater agreement analysis for AI system evaluations.

This script computes several measures of agreement between different raters
for each AI system contained in the Excel file:

* Cronbach's α for accuracy and recall — a measure of internal
  consistency across four raters (two human experts, GPT‑4.1 and o4‑mini).
* Pearson correlation between the two human experts for accuracy and
  recall.
* Average absolute difference between the two human experts for
  accuracy and recall (in percentage points).

The results provide insight into how consistently raters evaluate
responses across different systems.

To run the script, ensure that ``pandas`` and ``scipy`` are installed and
place the Excel file ``Сравнение ответов разных AI систем -v2.xlsx`` in
the same directory as this script (or adjust the `FILE_NAME` constant
below).
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Name of the Excel file containing the evaluation data.
FILE_NAME = "Сравнение ответов разных AI систем -v2.xlsx"


def cronbach_alpha(matrix: np.ndarray) -> float:
    """Compute Cronbach's alpha for a matrix of scores.

    The matrix should have shape (n_subjects, n_raters).  Each column
    represents ratings from one rater across multiple subjects.  Cronbach's
    alpha assesses how closely related the set of ratings are as a group.

    Args:
        matrix: 2D numpy array of shape (n_subjects, n_raters).

    Returns:
        Cronbach's alpha value (float).  Returns NaN if the total
        variance is zero.
    """
    n_raters = matrix.shape[1]
    # Variance of each rater across subjects
    raters_var = matrix.var(axis=0, ddof=1)
    # Total score per subject (sum across raters)
    total_scores = matrix.sum(axis=1)
    # Variance of total scores across subjects
    total_var = total_scores.var(ddof=1)
    if total_var == 0:
        return float("nan")
    alpha = (n_raters / (n_raters - 1)) * (1 - raters_var.sum() / total_var)
    return alpha


def compute_agreement_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute inter‑rater agreement metrics for each system.

    Args:
        df: DataFrame containing evaluation metrics for all systems.

    Returns:
        DataFrame summarising Cronbach's alpha for accuracy and recall,
        Pearson correlation between the two human raters and mean absolute
        differences between the human raters.
    """
    systems = ["Witsy", "Xplain", "Yandex", "AnythingLLM", "Onyx"]
    suffixes: Dict[str, str] = {
        "Witsy": "",
        "Xplain": ".1",
        "Yandex": ".2",
        "AnythingLLM": ".3",
        "Onyx": ".4",
    }
    records: List[Dict[str, float]] = []
    for system in systems:
        suff = suffixes[system]
        # Column names for accuracy and recall across raters
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
        # Drop rows with missing values for the selected columns
        acc_matrix = df[acc_cols].dropna().to_numpy()
        rec_matrix = df[rec_cols].dropna().to_numpy()
        # Cronbach's alpha for accuracy and recall
        alpha_acc = cronbach_alpha(acc_matrix)
        alpha_rec = cronbach_alpha(rec_matrix)
        # Pearson correlation between human raters (first two columns)
        human_acc_corr, human_acc_p = pearsonr(
            df[acc_cols[0]].dropna(), df[acc_cols[1]].dropna()
        )
        human_rec_corr, human_rec_p = pearsonr(
            df[rec_cols[0]].dropna(), df[rec_cols[1]].dropna()
        )
        # Mean absolute differences between human raters
        mean_abs_diff_accuracy = (df[acc_cols[0]] - df[acc_cols[1]]).abs().mean()
        mean_abs_diff_recall = (df[rec_cols[0]] - df[rec_cols[1]]).abs().mean()
        records.append(
            {
                "system": system,
                "alpha_accuracy": alpha_acc,
                "alpha_recall": alpha_rec,
                "human_acc_corr": human_acc_corr,
                "human_acc_p": human_acc_p,
                "human_rec_corr": human_rec_corr,
                "human_rec_p": human_rec_p,
                "mean_abs_diff_accuracy": mean_abs_diff_accuracy,
                "mean_abs_diff_recall": mean_abs_diff_recall,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    # Locate the Excel file
    data_file = Path(FILE_NAME)
    if not data_file.exists():
        raise FileNotFoundError(
            f"Could not find {FILE_NAME} in the current directory."
        )
    df = pd.read_excel(data_file)
    result = compute_agreement_metrics(df)
    # Display results
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()