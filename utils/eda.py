import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


def run_eda(df: pd.DataFrame, outlier_method: str = "IQR (1.5×)", max_cat_unique: int = 20) -> dict:
    """Run full exploratory data analysis on a dataframe."""
    n_rows, n_cols = df.shape

    # Detect column types
    num_cols = []
    cat_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        elif df[col].nunique() <= max_cat_unique:
            cat_cols.append(col)
        else:
            cat_cols.append(col)

    # Summary dataframe
    summary_rows = []
    for col in df.columns:
        col_type = "numeric" if col in num_cols else "categorical"
        missing = df[col].isna().sum()
        missing_pct = round(missing / n_rows * 100, 1)
        unique = df[col].nunique()
        if col in num_cols:
            summary_rows.append({
                "Column": col,
                "Type": col_type,
                "Missing": f"{missing} ({missing_pct}%)",
                "Unique": unique,
                "Mean": round(df[col].mean(), 2),
                "Std": round(df[col].std(), 2),
                "Min": round(df[col].min(), 2),
                "Max": round(df[col].max(), 2),
            })
        else:
            top_val = df[col].value_counts().index[0] if not df[col].value_counts().empty else "—"
            summary_rows.append({
                "Column": col,
                "Type": col_type,
                "Missing": f"{missing} ({missing_pct}%)",
                "Unique": unique,
                "Mean": "—",
                "Std": "—",
                "Min": "—",
                "Max": f"Top: {top_val}",
            })
    summary_df = pd.DataFrame(summary_rows)

    # Missing values table
    miss_data = []
    for col in df.columns:
        m = df[col].isna().sum()
        if m > 0:
            miss_data.append({"Column": col, "Missing Count": m, "Missing %": round(m / n_rows * 100, 2)})
    missing_df = pd.DataFrame(miss_data)

    # Outliers
    outliers = {}
    for col in num_cols:
        series = df[col].dropna()
        if outlier_method == "IQR (1.5×)":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        elif outlier_method == "IQR (3×)":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        else:  # Z-score
            mean, std = series.mean(), series.std()
            lower, upper = mean - 3 * std, mean + 3 * std

        mask = (series < lower) | (series > upper)
        out_vals = series[mask]
        if len(out_vals) > 0:
            outliers[col] = {
                "count": len(out_vals),
                "pct": round(len(out_vals) / len(series) * 100, 2),
                "lower": lower,
                "upper": upper,
                "min_out": out_vals.min(),
                "max_out": out_vals.max(),
                "values": out_vals.tolist()[:20]
            }

    total_outliers = sum(v["count"] for v in outliers.values())

    # Correlations
    top_correlations = pd.DataFrame()
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr()
        pairs = []
        cols = corr_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append({
                    "Column A": cols[i],
                    "Column B": cols[j],
                    "Correlation": round(corr_matrix.iloc[i, j], 3)
                })
        top_correlations = (
            pd.DataFrame(pairs)
            .assign(Abs=lambda x: x["Correlation"].abs())
            .sort_values("Abs", ascending=False)
            .drop("Abs", axis=1)
            .head(10)
            .reset_index(drop=True)
        )

    num_stats = {}
    for col in num_cols:
        s = df[col].dropna()
        num_stats[col] = {
            "mean": round(s.mean(), 2),
            "std": round(s.std(), 2),
            "min": round(s.min(), 2),
            "max": round(s.max(), 2),
            "median": round(s.median(), 2),
            "skew": round(s.skew(), 2),
            "outliers": outliers.get(col, {}).get("count", 0)
        }

    cat_stats = {}
    for col in cat_cols:
        vc = df[col].value_counts()
        cat_stats[col] = {
            "unique": int(df[col].nunique()),
            "top_values": vc.head(5).to_dict(),
            "missing": int(df[col].isna().sum())
        }

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "summary_df": summary_df,
        "missing": missing_df,
        "outliers": outliers,
        "total_outliers": total_outliers,
        "top_correlations": top_correlations,
        "num_stats": num_stats,
        "cat_stats": cat_stats,
    }
