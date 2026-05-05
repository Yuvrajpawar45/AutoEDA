import json
import os
import numpy as np
from typing import Generator
from groq import Groq
from dotenv import load_dotenv

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

load_dotenv()
_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key or api_key == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY not set in .env file.")
        _client = Groq(api_key=api_key)
    return _client


def generate_insights_stream(df, eda: dict) -> Generator[str, None, None]:
    """Stream AI insights using Groq's free LLaMA model."""
    summary = {
        "rows": eda["n_rows"],
        "columns": eda["n_cols"],
        "numeric_columns": eda["num_stats"],
        "categorical_columns": eda["cat_stats"],
        "missing_columns": (
            eda["missing"].to_dict(orient="records")
            if not eda["missing"].empty else []
        ),
        "total_outliers": eda["total_outliers"],
        "outlier_columns": {
            col: {"count": v["count"], "pct": v["pct"]}
            for col, v in eda["outliers"].items()
        },
        "top_correlations": (
            eda["top_correlations"].head(5).to_dict(orient="records")
            if not eda["top_correlations"].empty else []
        ),
    }

    for col in summary["categorical_columns"]:
        tv = summary["categorical_columns"][col].get("top_values", {})
        truncated = {}
        for k, v in list(tv.items())[:5]:
            truncated[str(k)[:60]] = v
        summary["categorical_columns"][col]["top_values"] = truncated

    prompt_summary = json.dumps(summary, indent=2, cls=_NumpyEncoder)
    if len(prompt_summary) > 6000:
        prompt_summary = prompt_summary[:6000] + "\n... (truncated)"

    prompt = f"""You are a senior data scientist. Analyze this EDA summary and give clear, actionable insights.

EDA Summary:
{prompt_summary}

Write a structured analysis with these sections:

## 📊 Dataset Overview
Summarize the dataset — size, column types, overall quality.

## 📈 Key Statistical Findings
Highlight 3-4 most interesting stats. Be specific with actual numbers and column names.

## 🚨 Data Quality Issues
Missing values, outliers, suspicious distributions. What needs fixing before modeling?

## 🔗 Patterns & Relationships
What relationships likely exist? What do the top correlations suggest?

## 🎯 Recommendations
Top 4 specific next steps — feature engineering, transformations, modeling approaches.

Be concise and specific. Use actual column names and numbers."""

    try:
        client = _get_client()
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
    except ValueError as e:
        yield f"❌ **Config error:** {str(e)}"
    except Exception as e:
        yield f"❌ **Error ({type(e).__name__}):** {str(e)}"
