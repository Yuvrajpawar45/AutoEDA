# 📊 Automated Data Analysis Agent

An AI-powered EDA tool built with Streamlit + Claude API.  
Upload any CSV → get instant charts, outlier detection, and plain-English AI insights.

---

## 🚀 Quick Setup (VS Code)

### 1. Install dependencies
Open a terminal in this folder and run:

```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`

---

## 🔑 Anthropic API Key (for AI Insights)

1. Go to [console.anthropic.com](https://console.anthropic.com) and create an API key
2. Paste it in the **sidebar** of the app when it opens
3. Click **"Generate Insights"** in the AI Insights tab

> You can also set it as an environment variable:  
> `export ANTHROPIC_API_KEY=sk-ant-...`

---

## 📁 Project Structure

```
eda_agent/
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── README.md
└── utils/
    ├── __init__.py
    ├── eda.py              # EDA logic (stats, outliers, correlations)
    ├── charts.py           # Matplotlib/Seaborn visualizations
    └── insights.py         # Claude API streaming insights
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📋 Overview | Column types, missing values, data preview |
| 📈 Distributions | Histograms, box plots, bar charts per column |
| 🔥 Correlations | Seaborn heatmap + ranked correlation table |
| 🚨 Outliers | IQR & Z-score detection with visual highlights |
| 🤖 AI Insights | Claude streams a full written analysis |

---

## 🛠️ Tech Stack

- **Streamlit** — UI framework
- **Pandas** — data loading & stats
- **Matplotlib / Seaborn** — visualizations
- **SciPy** — statistical methods
- **Anthropic Claude API** — AI-generated insights
