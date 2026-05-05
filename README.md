![header](https://capsule-render.vercel.app/api?type=waving&color=1A6BFF&height=120&section=header)

# AutoEDA - Automated EDA Agent

![Typing SVG](https://readme-typing-svg.demolab.com?font=Instrument+Serif&size=22&pause=1000&color=1A6BFF&center=true&vCenter=true&width=600&lines=Automated+Data+Analysis+Agent;Upload+CSV+%E2%86%92+Instant+Charts+%2B+AI+Insights;Built+for+Data+Scientists)

---

**Drop any CSV. Get instant EDA, visualizations, outlier detection, and AI-written insights.**  
No code required. Powered by Streamlit, Pandas, Matplotlib, Seaborn, and Groq.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-F55036?style=flat-square)](https://groq.com)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557C?style=flat-square)](https://matplotlib.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-4C72B0?style=flat-square)](https://seaborn.pydata.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

> Upload a CSV -> the agent automatically runs full EDA, generates charts, detects outliers, and streams a plain-English analysis written by LLaMA 3.1.

![rect](https://capsule-render.vercel.app/api?type=rect&color=1c201c&height=2)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [How the AI Insights Work](#how-the-ai-insights-work)
- [Deploy for Free](#deploy-for-free)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

**AutoEDA** is an automated Exploratory Data Analysis agent built with Streamlit. It solves a real problem for data scientists and analysts: the EDA phase is repetitive, time-consuming, and often requires writing the same boilerplate code for every new dataset.

AutoEDA simplifies that workflow:

```text
Without AutoEDA  ->  Write pandas code -> run cells -> iterate -> explain manually
With AutoEDA     ->  Upload CSV -> full analysis in seconds -> AI writes the summary
```

Every tab is computed automatically on upload. The AI Insights tab sends a compact EDA summary to LLaMA 3.1 through Groq and streams back a structured analysis covering statistics, data quality issues, patterns, and modeling recommendations.

---

## Features

| Module | What it does |
|--------|-------------|
| **Overview** | Column type detection, missing value report, data preview table |
| **Distributions** | Histograms and box plots for numeric columns, bar charts for categorical columns |
| **Correlations** | Seaborn heatmap with ranked correlation pairs table |
| **Outliers** | IQR and Z-score detection with visual summaries per column |
| **AI Insights** | LLaMA 3.1 streams a structured written analysis through Groq |
| **Export** | Download EDA report as `.txt` |

---

## System Architecture

### Application Overview

```text
+-----------------------------------------------------------------+
|                            AutoEDA                              |
|                                                                 |
|  +--------------+    +--------------------------------------+   |
|  |   Sidebar    |    |              Main Panel              |   |
|  |              |    |                                      |   |
|  |  CSV Upload  | -> | Overview | Distributions | Corr      |   |
|  |  Options     |    | Outliers | AI Insights               |   |
|  |  Export Btn  |    |                                      |   |
|  +--------------+    +--------------------------------------+   |
|                                      |                          |
|                                      v                          |
|                                utils/eda.py                     |
|                         stats | outliers | types                |
|                                      |                          |
|                         +------------+------------+             |
|                         v                         v             |
|                  utils/charts.py          utils/insights.py     |
|              matplotlib | seaborn       Groq API | LLaMA 3.1   |
+-----------------------------------------------------------------+
```

### EDA Pipeline

```text
CSV Upload
     |
     v
Data Loader
     |
     v
Type Detection -> numeric stats | categorical stats | missing values
     |
     v
Outlier Scan -> IQR 1.5x | IQR 3x | Z-score
     |
     v
Correlation Analysis -> Pearson matrix | top ranked pairs
     |
     v
Chart Rendering -> histograms | box plots | bar charts | heatmap
     |
     v
Rendered in Streamlit tabs
```

### AI Insights Pipeline

```text
EDA results dict
       |
       v
Summary Builder -> num_stats | cat_stats | outliers | correlations
       |
       v
JSON serialization with NumPy type handling
       |
       v
Groq API call -> llama-3.1-8b-instant
       |
       v
Streamlit UI updates in real time
```

---

## Project Structure

```text
AutoEDA/
|
├── app.py                  # Main Streamlit app - layout, tabs, UI logic
|
├── utils/
│   ├── __init__.py
│   ├── eda.py              # Core EDA - type detection, stats, outliers, correlations
│   ├── charts.py           # Matplotlib and Seaborn chart functions
│   └── insights.py         # Groq API integration, streaming, NumPy encoder
|
├── .env.example            # Safe template showing required variables
├── .gitignore              # Blocks .env, venv, __pycache__
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **UI Framework** | Streamlit | Python-native, rapid deployment, no frontend build step |
| **Styling** | HTML + CSS injected via `st.markdown` | Custom cards, navbar, tabs, and app styling |
| **Data** | Pandas + NumPy | Dtype inference, data wrangling, vectorized stats |
| **Statistics** | SciPy | Z-score computation and statistical utilities |
| **Visualization** | Matplotlib + Seaborn | Flexible chart generation |
| **AI / LLM** | Groq API + LLaMA 3.1-8B-Instant | Fast streaming text generation with a free tier |
| **Config** | python-dotenv | Local `.env`-based API key management |
| **Fonts** | Google Fonts CDN | Consistent typography |

### Design Decisions

**Why Streamlit over Flask/FastAPI + React?**  
This is a data science tool for analysts and data scientists. Streamlit keeps the entire app in Python, while custom HTML/CSS fills the visual design gaps.

**Why Groq + LLaMA instead of OpenAI?**  
Groq offers a fast free tier, and LLaMA 3.1-8B is capable enough for structured EDA explanation tasks.

**Why serialize EDA to JSON for the prompt?**  
The app sends a compact statistical summary instead of raw data rows. This keeps token usage low and makes the AI tab work across different dataset sizes.

**Why a custom NumPy encoder?**  
Pandas statistics often produce `numpy.int64`, `numpy.float64`, and arrays. Python's default `json` encoder cannot serialize those values directly, so the app converts them to plain Python types.

---

## Quick Start

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | Required |
| Groq API Key | Free at [console.groq.com](https://console.groq.com) |

### Step 1 - Clone

```bash
git clone https://github.com/Yuvrajpawar45/AutoEDA.git
cd AutoEDA
```

### Step 2 - Virtual Environment

```bash
# Windows PowerShell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 - Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 - Configure

```bash
cp .env.example .env
```

Open `.env` and set your key:

```text
GROQ_API_KEY=gsk_your_key_here
```

### Step 5 - Run

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes, for AI tab only | Free key from [console.groq.com](https://console.groq.com) |

The `.env` file is listed in `.gitignore` and will never be committed.  
All other tabs work without an API key.

---

## How the AI Insights Work

The AI Insights tab does not send raw CSV data to the API. Instead:

1. The EDA module computes a compact statistical summary.
2. That summary is serialized to JSON using a custom encoder that handles NumPy types.
3. The JSON is capped to keep prompts small.
4. A structured prompt is sent to `llama-3.1-8b-instant` through Groq's streaming API.
5. Tokens stream back and update the Streamlit UI in real time.
6. The response is stored in `st.session_state` so it persists across re-renders.

**Result:** The LLM sees pre-computed statistics, not raw dataset rows.

---

## Deploy for Free

### Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** and select `Yuvrajpawar45/AutoEDA`.
4. Set the branch to `main` and main file to `app.py`.
5. Click **Advanced settings -> Secrets** and add:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

6. Click **Deploy**.

Your app gets a public Streamlit URL after deployment.

| Service | Cost | Notes |
|---------|------|-------|
| Streamlit Community Cloud | Free | Good for Streamlit portfolio apps |
| Groq API | Free tier | Rate limited |
| GitHub | Free | Stores the code |

---

## Roadmap

| Phase | Status | Feature |
|-------|--------|---------|
| Phase 1 | Complete | CSV upload, type detection, basic stats |
| Phase 2 | Complete | Outlier detection and correlation heatmap |
| Phase 3 | Complete | AI insights via Groq, streaming, professional UI |
| Phase 4 | Planned | Excel and JSON file support |
| Phase 5 | Planned | Time series detection and trend analysis |
| Phase 6 | Planned | Column-level AI chat |
| Phase 7 | Planned | Export full EDA report as PDF |
| Phase 8 | Planned | Automatic feature engineering suggestions |

---

## License

MIT License. Free to use, modify, and distribute.

---

![footer](https://capsule-render.vercel.app/api?type=waving&color=1A6BFF&height=100&section=footer)

<div align="center">
  Built by <strong>Yuvraj Pawar</strong> &nbsp;·&nbsp;
  <a href="https://github.com/Yuvrajpawar45">GitHub</a>
  <br><br>
  If this project helped you, consider giving it a star.
</div>
