from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DatasetProfile:
    rows: int
    columns: int
    column_names: list[str]
    dtypes: dict[str, str]
    missing: dict[str, int]
    missing_percent: dict[str, float]
    duplicate_rows: int
    unique_counts: dict[str, int]
    numeric_stats: dict[str, dict[str, float | None]]
    categorical_distributions: dict[str, dict[str, int]]
    date_columns: list[str]
    outliers: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    numeric = df.select_dtypes(include=np.number)
    stats: dict[str, dict[str, float | None]] = {}
    outliers: dict[str, int] = {}
    for column in numeric.columns:
        series = numeric[column].dropna()
        stats[column] = {
            "mean": _safe_float(series.mean()),
            "median": _safe_float(series.median()),
            "std": _safe_float(series.std()),
            "min": _safe_float(series.min()),
            "max": _safe_float(series.max()),
        }
        if len(series):
            q1, q3 = series.quantile([0.25, 0.75])
            spread = q3 - q1
            outliers[column] = int(((series < q1 - 1.5 * spread) | (series > q3 + 1.5 * spread)).sum())
        else:
            outliers[column] = 0

    categorical = df.select_dtypes(exclude=np.number)
    distributions = {
        col: {str(k): int(v) for k, v in df[col].value_counts(dropna=False).head(8).items()}
        for col in categorical.columns
    }
    date_columns: list[str] = []
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            date_columns.append(column)
        elif any(token in column.lower() for token in ("date", "time", "timestamp")):
            parsed = pd.to_datetime(df[column], errors="coerce")
            if parsed.notna().mean() >= 0.65:
                date_columns.append(column)

    missing = df.isna().sum()
    return DatasetProfile(
        rows=int(len(df)),
        columns=int(len(df.columns)),
        column_names=[str(c) for c in df.columns],
        dtypes={str(c): str(t) for c, t in df.dtypes.items()},
        missing={str(k): int(v) for k, v in missing.items() if v},
        missing_percent={str(k): round(float(v / max(len(df), 1) * 100), 1) for k, v in missing.items() if v},
        duplicate_rows=int(df.duplicated().sum()),
        unique_counts={str(c): int(df[c].nunique(dropna=True)) for c in df.columns},
        numeric_stats=stats,
        categorical_distributions=distributions,
        date_columns=date_columns,
        outliers=outliers,
    )


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
        return round(number, 4) if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None
