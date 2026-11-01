from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


TOOL_DESCRIPTIONS = {
    "dataset_summary": "Summarize the dataset dimensions, columns, and basic statistics.",
    "missing_analysis": "Quantify missing values by column.",
    "duplicate_check": "Count exact duplicate records.",
    "correlation_analysis": "Measure pairwise correlations among numeric columns.",
    "groupby_aggregation": "Compare a numeric metric across a categorical dimension.",
    "segment_change_analysis": "Compare equal-length date windows by category to find which segment changed most.",
    "trend_analysis": "Aggregate a numeric metric over a date or ordered dimension.",
    "outlier_detection": "Find unusually high or low numeric observations with the IQR rule.",
    "distribution_analysis": "Describe the spread of a numeric or categorical column.",
    "feature_relationships": "Estimate non-linear association between a numeric target and features.",
    "statistical_analysis": "Compute robust descriptive statistics for numeric columns.",
}


def run_tool(name: str, df: pd.DataFrame, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one registered tool. Parameters are checked against actual columns."""
    params = params or {}
    if name == "dataset_summary":
        return {"rows": int(len(df)), "columns": int(len(df.columns)), "column_names": [str(c) for c in df.columns],
                "dtypes": {str(c): str(t) for c, t in df.dtypes.items()}}
    if name == "missing_analysis":
        values = df.isna().sum()
        return {"missing": {str(k): int(v) for k, v in values.items() if v},
                "total_missing": int(values.sum()), "rows_with_missing": int(df.isna().any(axis=1).sum())}
    if name == "duplicate_check":
        return {"duplicate_rows": int(df.duplicated().sum())}
    if name in {"correlation_analysis", "statistical_analysis"}:
        numeric = df.select_dtypes(include=np.number)
        if name == "correlation_analysis":
            if numeric.shape[1] < 2:
                return {"message": "At least two numeric columns are needed for correlation.", "matrix": {}}
            return {"matrix": numeric.corr().round(3).replace({np.nan: None}).to_dict(),
                    "strongest_pairs": _strongest_pairs(numeric)}
        return {"statistics": numeric.describe().round(3).replace({np.nan: None}).to_dict()}
    if name == "groupby_aggregation":
        group = _column(df, params.get("group_by"), "group_by")
        metric = _column(df, params.get("metric"), "metric")
        if metric not in df.select_dtypes(include=np.number).columns:
            raise ValueError(f"'{metric}' must be numeric for aggregation.")
        result = df.groupby(group, dropna=False)[metric].agg(["mean", "sum", "count"]).sort_values("sum", ascending=False).head(20)
        return {"group_by": group, "metric": metric, "groups": _records(result.reset_index())}
    if name == "segment_change_analysis":
        group = _column(df, params.get("group_by"), "group_by")
        metric = _column(df, params.get("metric"), "metric")
        date = _column(df, params.get("date_column"), "date_column")
        if metric not in df.select_dtypes(include=np.number).columns:
            raise ValueError(f"'{metric}' must be numeric.")
        dates = pd.to_datetime(df[date], errors="coerce")
        valid = dates.notna() & df[metric].notna()
        if valid.sum() < 2:
            return {"message": f"Not enough parseable dates in '{date}' to compare periods."}
        midpoint = dates[valid].min() + (dates[valid].max() - dates[valid].min()) / 2
        earlier = valid & (dates <= midpoint)
        later = valid & (dates > midpoint)
        before = df.loc[earlier].groupby(group, dropna=False)[metric].sum().rename("earlier_total")
        after = df.loc[later].groupby(group, dropna=False)[metric].sum().rename("later_total")
        comparison = pd.concat([before, after], axis=1).fillna(0)
        comparison["change"] = comparison["later_total"] - comparison["earlier_total"]
        comparison["change_percent"] = np.where(
            comparison["earlier_total"].abs() > 1e-12,
            comparison["change"] / comparison["earlier_total"].abs() * 100,
            np.nan,
        )
        comparison = comparison.sort_values("change").head(20).reset_index()
        return {"date_column": date, "group_by": group, "metric": metric,
                "earlier_window": f"{dates[earlier].min().date()} to {dates[earlier].max().date()}",
                "later_window": f"{dates[later].min().date()} to {dates[later].max().date()}",
                "comparison_basis": "Sum per segment across two equal-length windows split at the dataset date midpoint.",
                "groups": _records(comparison)}
    if name == "trend_analysis":
        date = _column(df, params.get("date_column"), "date_column")
        metric = _column(df, params.get("metric"), "metric")
        dates = pd.to_datetime(df[date], errors="coerce")
        valid = dates.notna() & df[metric].notna()
        if not valid.any():
            return {"message": f"No parseable dates found in '{date}'."}
        trend = pd.DataFrame({"date": dates[valid], "value": df.loc[valid, metric]}).set_index("date")
        grouped = trend.resample("ME").agg(total=("value", "sum"), average=("value", "mean"), count=("value", "count")).dropna(how="all")
        return {"date_column": date, "metric": metric, "period": "month", "points": [
            {"period": str(index.date()), **{k: _number(v) for k, v in row.items()}} for index, row in grouped.iterrows()
        ]}
    if name == "outlier_detection":
        column = _column(df, params.get("column"), "column")
        if column not in df.select_dtypes(include=np.number).columns:
            raise ValueError(f"'{column}' must be numeric.")
        series = df[column].dropna()
        if series.empty:
            return {"column": column, "outlier_count": 0, "lower_bound": None, "upper_bound": None}
        q1, q3 = series.quantile([.25, .75])
        spread = q3 - q1
        low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
        return {"column": column, "outlier_count": int(((series < low) | (series > high)).sum()),
                "lower_bound": _number(low), "upper_bound": _number(high)}
    if name == "distribution_analysis":
        column = _column(df, params.get("column"), "column")
        if column in df.select_dtypes(include=np.number).columns:
            series = df[column].dropna()
            hist, edges = np.histogram(series, bins=min(12, max(1, int(np.sqrt(max(len(series), 1))))))
            return {"column": column, "kind": "numeric", "count": int(series.count()),
                    "mean": _number(series.mean()), "median": _number(series.median()),
                    "histogram": [{"range": f"{edges[i]:.2f}–{edges[i+1]:.2f}", "count": int(hist[i])} for i in range(len(hist))]}
        counts = df[column].astype("string").fillna("(missing)").value_counts().head(12)
        return {"column": column, "kind": "categorical", "categories": [{"label": str(k), "count": int(v)} for k, v in counts.items()]}
    if name == "feature_relationships":
        target = _column(df, params.get("target"), "target")
        if target not in df.select_dtypes(include=np.number).columns:
            raise ValueError(f"'{target}' must be numeric.")
        scores = {}
        mutual_information = {}
        y = df[target]
        for column in df.select_dtypes(include=np.number).columns:
            if column == target:
                continue
            valid = df[[column, target]].dropna()
            if len(valid) > 2 and valid[column].nunique() > 1:
                scores[column] = round(float(valid[column].corr(valid[target], method="spearman")), 3)
                try:
                    mutual_information[column] = round(float(mutual_info_regression(
                        valid[[column]], valid[target], random_state=42
                    )[0]), 4)
                except (ValueError, TypeError):
                    pass
        return {"target": target, "spearman_relationships": scores, "mutual_information": mutual_information,
                "caveat": "Association only; this does not establish causation."}
    raise ValueError(f"Tool '{name}' is not registered.")


def _column(df: pd.DataFrame, value: Any, label: str) -> str:
    if value not in df.columns:
        raise ValueError(f"'{label}' must name a column in this dataset.")
    return str(value)


def _strongest_pairs(df: pd.DataFrame) -> list[dict[str, Any]]:
    corr = df.corr().abs()
    pairs = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1:]:
            value = corr.loc[left, right]
            if pd.notna(value):
                pairs.append({"left": str(left), "right": str(right), "correlation": round(float(df[[left, right]].corr().iloc[0, 1]), 3)})
    return sorted(pairs, key=lambda x: abs(x["correlation"]), reverse=True)[:8]


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): _plain(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def _plain(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 4)
    return value.item() if hasattr(value, "item") else value


def _number(value: Any) -> float | None:
    try:
        return round(float(value), 4) if pd.notna(value) and np.isfinite(value) else None
    except (TypeError, ValueError):
        return None
