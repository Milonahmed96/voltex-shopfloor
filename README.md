# ⚡ Voltex ShopFloor Analyst

> AI-powered store operations analyst for Voltex Retail — generates daily priority action briefs for store managers by reasoning over multi-store performance data.

[![CI](https://github.com/Milonahmed96/voltex-shopfloor/actions/workflows/ci.yml/badge.svg)](https://github.com/Milonahmed96/voltex-shopfloor/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Tests](https://img.shields.io/badge/tests-37%20passing-brightgreen)

---

## What This Is

Voltex ShopFloor Analyst is an LLM-powered store operations tool that transforms raw retail performance data into prioritised, actionable daily briefs for store managers. It simulates the kind of AI tooling being deployed at UK tech retailers like Currys — where Action AI (built with Quorso) consolidates sales, footfall, customer satisfaction, and promotional data into a single dashboard and surfaces the highest-value actions for each store.

The core insight: the problem is not data availability. Store managers already have access to sales reports, footfall counts, and satisfaction scores. The problem is signal extraction — identifying which of the dozens of available metrics actually requires attention today, understanding the root cause, and knowing what action to take. That reasoning layer is where LLMs create genuine value.

**Voltex Retail is a fictional business** created for this project series. All data is synthetically generated with realistic patterns, seasonality, and embedded operational problems.

---

## Business Problem

A Voltex store manager oversees a team of 15–40 colleagues across a 10,000–30,000 sq ft store. On any given day they are looking at:

- Sales performance vs daily and weekly targets across 6 product categories
- Footfall and conversion rates
- VoltCare, VoltMobile, and VoltInstall attach rates
- Customer satisfaction scores
- Staff scheduling vs actual hours
- Active promotions and their uplift

By the time a manager has assembled this picture manually from separate systems, the trading day is already underway. More critically, they have no way of knowing which of their 20+ metrics actually needs attention today vs which ones are within normal variation.

ShopFloor Analyst solves this by:
1. Computing all metrics and comparing them against the store's own baseline and the network average
2. Identifying statistically meaningful deviations
3. Using Claude to reason about root causes and generate a ranked action brief
4. Surfacing peer store benchmarks — what are the top 3 stores in your region doing differently?

---

## Architecture

```
Synthetic store data (20 stores, 12 weeks)
              │
              ▼
┌─────────────────────────────┐
│  Data Pipeline              │  Compute metrics, baselines, anomalies
│  StoreSnapshot (Pydantic)   │  Structured context for LLM
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  LLM Reasoning Layer        │  Claude Sonnet
│  StoreActionBrief (Pydantic)│  Ranked priorities, root causes,
│                             │  peer benchmarks, trend narrative
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Streamlit Dashboard        │  Store selector, metric cards,
│                             │  AI brief, network overview,
│                             │  trend sparklines
└─────────────────────────────┘
```

### Key Design Decisions

**Synthetic data with embedded problems** — the data generator creates 20 stores with realistic seasonality, promotional effects, and six embedded operational problems:
- Declining laptop attach rates correlated with a staff change
- High footfall but poor conversion (browse not buy pattern)
- Small store outperforming its tier through specific attach behaviour
- Same promotion, different execution quality across two stores
- NPS deterioration correlated with weekend understaffing
- Recovering store where trend is positive but absolute numbers still look weak

**Structured context for LLM reasoning** — rather than passing raw CSV data to Claude, the pipeline pre-computes all metrics and formats a structured `StoreSnapshot` object. This reduces token consumption, improves reasoning accuracy, and makes the system faster.

**Peer benchmarking** — every store brief includes what the top 3 stores in the same region are doing differently. This makes recommendations specific and credible rather than generic.

**Pydantic output schema** — the analyst returns a typed `StoreActionBrief` with ranked priorities, root cause analysis, and confidence levels — making the output safe to render in the UI without parsing.

---

## Data Schema

### Store Network
20 synthetic Voltex stores across UK regions with realistic size tiers:

| Region | Stores | Size Tiers |
|---|---|---|
| London | 4 | Large (2), Medium (2) |
| South East | 3 | Large (1), Medium (2) |
| Midlands | 4 | Large (2), Medium (1), Small (1) |
| North West | 3 | Large (1), Medium (2) |
| North East | 3 | Medium (2), Small (1) |
| Scotland | 3 | Medium (2), Small (1) |

### Daily Metrics Per Store
- Units sold by category (Computing, TV, White Goods, Phones, Gaming, Smart Home)
- Revenue by category
- Footfall and conversion rate
- VoltCare attach rate by category
- VoltMobile SIM attach rate
- VoltInstall attach rate
- Customer satisfaction (NPS)
- Staff scheduled vs actual hours
- Active promotions and uplift

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Claude Sonnet (Anthropic API) |
| Data generation | Python, NumPy, Pandas |
| Output schema | Pydantic v2 |
| Dashboard | Streamlit + Plotly |
| Testing | pytest |
| CI | GitHub Actions |
| Python | 3.11 |

---

## Project Structure

```
voltex-shopfloor/
├── data/                        # Generated synthetic data (gitignored)
├── tests/                       # pytest test suite
├── .github/workflows/ci.yml     # GitHub Actions CI
├── generate_data.py             # Synthetic data generator
├── data_pipeline.py             # Metrics computation and StoreSnapshot
├── analyst.py                   # LLM reasoning layer
├── app.py                       # Streamlit dashboard
├── requirements.txt
└── .env                         # API keys (gitignored)
```

---

## Getting Started

### Prerequisites
- Python 3.11
- Anthropic API key

### Installation

```bash
git clone https://github.com/Milonahmed96/voltex-shopfloor.git
cd voltex-shopfloor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Configuration

Create a `.env` file:

```
ANTHROPIC_API_KEY=your_key_here
```

### Generate Data

```bash
python generate_data.py
```

### Run the Dashboard

```bash
streamlit run app.py
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## Part of the Voltex AI Engineering Series

This is Project A in a three-project series simulating AI engineering for a UK omnichannel tech retailer:

| Project | Repo | Description |
|---|---|---|
| B — Contact Centre Co-Pilot | [voltex-copilot](https://github.com/Milonahmed96/voltex-copilot) | RAG-powered agent assistant, 82% evaluation accuracy |
| A — ShopFloor Analyst | voltex-shopfloor (this repo) | LLM store operations reasoning |
| C — Repair Triage Agent | coming soon | LangGraph agentic repair routing |

---

## Author

**Milon Ahmed**
MSc Data Science with Advanced Research
University of Hertfordshire, UK.

[GitHub](https://github.com/Milonahmed96) · [LinkedIn](https://linkedin.com/in/milonahmed96)

---

## Licence

MIT