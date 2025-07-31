"""
Compute average accuracy, recall and derived metrics for each rater and system.

This script loads the evaluation data from the Excel file
``Сравнение ответов разных AI систем -v2.xlsx`` and computes, for each AI
system and each rater (Человек 1, Человек 2, GPT‑4.1, o4‑mini), the
following metrics:

* Mean accuracy (в процентах) — средняя точность оценщика для данной
  системы.
* Mean recall (в процентах) — средняя полнота оценщика для данной
  системы.
* Harmonic mean (F1) — гармоническое среднее между средней точностью
  и средней полнотой; показывает баланс двух метрик【571433778968177†L149-L160】.
* Arithmetic mean — простое среднее между средней точностью и
  средней полнотой.

Запустите скрипт командой

    python rater_metrics_summary.py

Он выведет таблицу с результатами в консоль.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Name of the Excel file containing evaluation results
FILE_NAME = "Сравнение ответов разных AI систем -v2.xlsx"

# Systems and their suffixes used in column names
SUFFIXES: Dict[str, str] = {
    "Witsy": "",
    "Xplain": ".1",
    "Yandex": ".2",
    "AnythingLLM": ".3",
    "Onyx": ".4",
}

# List of raters in the column names
RATERS = ["Человек 1", "Человек 2", "gpt-4.1", "o4-mini"]


def compute_rater_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean metrics per system and rater.

    Args:
        df: DataFrame loaded from the Excel file.

    Returns:
        DataFrame with columns: system, rater, mean_accuracy,
        mean_recall, harmonic_mean_f1, arithmetic_mean.
    """
    records: List[Dict[str, object]] = []
    for system, suffix in SUFFIXES.items():
        for rater in RATERS:
            acc_col = f"Точность {rater}{suffix}"
            rec_col = f"Полнота {rater}{suffix}"
            acc_values = df[acc_col].dropna()
            rec_values = df[rec_col].dropna()
            mean_acc = acc_values.mean()
            mean_rec = rec_values.mean()
            # Harmonic mean (F1)
            denom = mean_acc + mean_rec
            f1 = (2 * mean_acc * mean_rec / denom) if denom != 0 else np.nan
            # Arithmetic mean
            arithmetic = (mean_acc + mean_rec) / 2
            records.append({
                "system": system,
                "rater": rater,
                "mean_accuracy": mean_acc,
                "mean_recall": mean_rec,
                "harmonic_mean_f1": f1,
                "arithmetic_mean": arithmetic,
            })
    return pd.DataFrame(records)


def main() -> None:
    path = Path(FILE_NAME)
    if not path.exists():
        raise FileNotFoundError(f"Could not find {FILE_NAME} in the current directory")
    df = pd.read_excel(path)
    summary = compute_rater_metrics(df)
    # Format floats to two decimal places for neat printing
    pd.set_option("display.max_rows", None)
    print(summary.to_string(index=False, formatters={
        "mean_accuracy": lambda x: f"{x:.2f}",
        "mean_recall": lambda x: f"{x:.2f}",
        "harmonic_mean_f1": lambda x: f"{x:.2f}",
        "arithmetic_mean": lambda x: f"{x:.2f}",
    }))


if __name__ == "__main__":
    main()