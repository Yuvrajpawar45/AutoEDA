<div align="center">

# AutoEDA

**Upload any CSV. Get instant charts, outlier detection, and AI-written insights.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-F55036?style=flat)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat)](LICENSE)

[Live Demo](#) &nbsp;·&nbsp; [Report Bug](https://github.com/Yuvrajpawar45/AutoEDA/issues) &nbsp;·&nbsp; [Request Feature](https://github.com/Yuvrajpawar45/AutoEDA/issues)

</div>

---

## What is AutoEDA?

AutoEDA is a fully automated Exploratory Data Analysis app built with Streamlit. Drop in any CSV file and the app instantly computes statistics, renders charts, detects outliers, and uses a free LLaMA 3.1 model via Groq to write a plain-English analysis of your data.

Built as a portfolio project to demonstrate end-to-end data science thinking: from raw data ingestion to AI-assisted interpretation.

---

## Features

| Module | What it does |
|--------|-------------|
| **Overview** | Column types, missing value detection, data preview |
| **Distributions** | Histograms and box plots for numeric columns, bar charts for categorical columns |
| **Correlations** | Seaborn heatmap with ranked correlation table |
| **Outliers** | IQR and Z-score detection with visual highlights |
| **AI Insights** | LLaMA 3.1 streams a structured written analysis through Groq |

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| UI | Streamlit + HTML/CSS | App framework and custom styling |
| Data | Pandas, NumPy, SciPy | Loading, statistics, outlier math |
| Charts | Matplotlib, Seaborn | Visualizations |
| AI | Groq API, LLaMA 3.1-8B | Streaming AI insights |
| Config | python-dotenv | Local API key management via `.env` |

---

## Project Structure

```text
AutoEDA/
├── app.py                  # Main Streamlit app
├── .env.example            # Template showing required environment variables
├── .gitignore
├── requirements.txt
└── utils/
    ├── __init__.py
    ├── eda.py              # Core EDA logic
    ├── charts.py           # Matplotlib and Seaborn chart functions
    └── insights.py         # Groq API streaming integration
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- A free Groq API key from [console.groq.com](https://console.groq.com)

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/Yuvrajpawar45/AutoEDA.git
cd AutoEDA
```

**2. Create and activate a virtual environment**

```bash
# Windows PowerShell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

```bash
cp .env.example .env
```

Open `.env` and add your Groq key:

```text
GROQ_API_KEY=gsk_your_key_here
```

**5. Run the app**

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes, for AI insights | Free key from [console.groq.com](https://console.groq.com) |

The `.env` file is listed in `.gitignore` and should never be committed to version control.

---

## Deploying to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **Create app** and select `Yuvrajpawar45/AutoEDA`.
3. Set the branch to `main` and the main file path to `app.py`.
4. Under **Advanced settings -> Secrets**, add:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

5. Click **Deploy**.

---

## How the AI Insights Work

```text
CSV uploaded
     |
     v
EDA module computes summary stats, outliers, and correlations
     |
     v
Summary serialized to JSON and truncated for token efficiency
     |
     v
Sent to Groq API -> LLaMA 3.1-8B-Instant
     |
     v
Response streamed back into the Streamlit UI
```

The model returns a structured analysis covering dataset overview, statistical findings, data quality issues, patterns, and recommendations.

---

## Limitations

- Groq free tier has rate limits.
- Very large CSVs may be slow on free Streamlit Cloud resources.
- The app currently supports CSV files only.

---

## Roadmap

- [ ] Support Excel and JSON uploads
- [ ] Add time series detection and trend charts
- [ ] Export a full EDA report as PDF
- [ ] Add column-level AI chat
- [ ] Add dark mode

---

## License

MIT License. Free to use, modify, and distribute.

---

<div align="center">
Built by <strong>Yuvraj Pawar</strong> &nbsp;·&nbsp;
<a href="https://github.com/Yuvrajpawar45">GitHub</a>
</div>
