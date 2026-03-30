<p align="center">
  <h1 align="center">UPI Behaviour Mystery</h1>
  <p align="center">
    <strong>Why do most UPI users go dormant — and who responds when you intervene?</strong>
  </p>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
  <a href="https://github.com/NiShITa-code/upi-behaviour-mystery/actions"><img src="https://img.shields.io/badge/tests-95%20passed-brightgreen.svg" alt="Tests"></a>
  <a href="https://streamlit.io"><img src="https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg" alt="Streamlit"></a>
</p>

---

<p align="center">
  <em>Applies retention & causal inference methods used at</em><br/>
  <img src="https://img.shields.io/badge/Google-4285F4?style=flat&logo=google&logoColor=white" alt="Google">
  <img src="https://img.shields.io/badge/PhonePe-5F259F?style=flat&logo=phonepe&logoColor=white" alt="PhonePe">
  <img src="https://img.shields.io/badge/Uber-000000?style=flat&logo=uber&logoColor=white" alt="Uber">
  <img src="https://img.shields.io/badge/Flipkart-F7D000?style=flat&logoColor=black" alt="Flipkart">
  <img src="https://img.shields.io/badge/Razorpay-0C2451?style=flat&logoColor=white" alt="Razorpay">
</p>

---

An open-source retention analytics toolkit for India's UPI payment ecosystem. Upload your data (or use NPCI-calibrated simulation), run churn prediction + causal uplift modelling, and get plain-English strategy recommendations — all through an interactive dashboard.

**491 million registered UPI users. Most barely transact.** This tool helps you figure out why, identify who will respond to interventions, and generate actionable recommendations.

## Who Is This For?

- **Product Managers** — get a strategy memo with prioritized, quantified recommendations you can take straight to a stakeholder meeting
- **Growth & Retention Analysts** — upload your user-level CSV and get churn segments, uplift scores, and ROI projections in minutes
- **Data Scientists** — a production-grade reference implementation of T-Learner uplift modelling with DeLong significance testing, SHAP explanations, A/B test power analysis, fairness auditing, sklearn pipelines, and 95 tests

## Core Finding

> Users who completed ≥8 transactions and made their first *merchant payment* within 14 days retained at **82%**. Users who didn't: **23%**. Day 14 is the make-or-break window.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/NiShITa-code/upi-behaviour-mystery.git
cd upi-behaviour-mystery

# Install with dev dependencies
pip install -e ".[dev]"

# Verify installation
pytest tests/ -v
```

**Requirements:** Python 3.10+, pip. All dependencies are in `pyproject.toml`.

## Usage

### Interactive Dashboard (recommended)

```bash
streamlit run app.py
```

This launches a 6-tab dashboard where you can:
1. Choose your data source (simulated or upload your own CSV)
2. Explore cohort analysis and retention curves
3. Inspect churn model performance (ROC, calibration, feature importance)
4. View uplift segments and targeting ROI
5. Get plain-English strategy recommendations
6. Explore SHAP explanations (global + individual user)
7. Design A/B experiments with power analysis
8. Run a model fairness audit across demographics
9. Download a stakeholder-ready decision memo

### Command Line

```bash
# Default run (10,000 users, seed=42)
python -m src.pipeline

# Custom parameters
python -m src.pipeline --n-users 5000 --seed 7 --cashback 30 --budget 50000
```

### Makefile Shortcuts

```bash
make pipeline        # full analysis, CLI output
make pipeline-small  # quick run (2,000 users)
make test            # test suite with coverage
make run             # launch Streamlit dashboard
```

---

## Bring Your Own Data

This tool is designed to work with **your real data**, not just simulations.

**Upload a user-level CSV** with these columns:

| Column | Type | Description |
|---|---|---|
| `user_id` | str | Unique user identifier |
| `city_tier` | int | 1 (metro), 2 (tier-2), 3 (tier-3) |
| `txn_d7` | int | Transaction count in first 7 days |
| `value_d7` | float | Total transaction value in first 7 days (₹) |
| `txn_d14` | int | Transaction count in first 14 days |
| `value_d14` | float | Total transaction value in first 14 days (₹) |
| `has_first_p2m_d14` | bool | Made a merchant payment within 14 days? |
| `first_p2m_day` | int | Day of first merchant payment (0 if none) |
| `cat_diversity` | int | Distinct categories used (0–8) |
| `p2m_ratio` | float | Fraction of transactions that are merchant payments |
| `churned` | int | 1 = churned, 0 = retained |

Download a sample template from the dashboard sidebar to see the exact format.

**What if I only have raw transaction logs?** The CSV loader also accepts raw transaction-level data and computes the features automatically.

---

## How It Works

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Data Input  │────▶│   Feature    │────▶│    Churn     │────▶│   Uplift     │
│  CSV or Sim  │     │  Engineering │     │  Prediction  │     │  Modelling   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                       │
                     ┌──────────────┐     ┌──────────────┐            │
                     │   Strategy   │◀────│   Segment    │◀───────────┘
                     │   Recommender│     │  Assignment  │
                     └──────────────┘     └──────────────┘
```

| Stage | Method | Why This Choice |
|---|---|---|
| Data simulation | Poisson arrival + log-normal values | Calibrated to NPCI published aggregate stats |
| CSV loader | Schema validation + type coercion | BYOD with clear, actionable error messages |
| Schema validation | Custom validators on DataFrame | Fail fast at data boundary, not model boundary |
| Feature engineering | sklearn `Pipeline` + `TransformerMixin` | No train/test leakage; picklable for serving |
| Churn model | LightGBM + 5-fold stratified CV | Handles class imbalance; early stopping on val split |
| Model comparison | DeLong test (1988) | Statistical significance of AUC difference |
| Baseline | Logistic Regression | Interpretable reference for AUC comparison |
| Calibration | Brier score | Ensures predicted probabilities are meaningful |
| Uplift model | T-Learner (two LightGBM models) | Estimates individual treatment effect (ITE) |
| Segmentation | 2×2 P0/P1 threshold matrix | Persuadable / Sure Thing / Lost Cause / Sleeping Dog |
| Strategy | Rule-based recommender | Plain-English, PM-ready recommendations with ROI |
| SHAP | TreeExplainer (exact) | Global + individual-level model explanations |
| Experiment design | Two-proportion z-test power analysis | Sample size, duration, Bonferroni-corrected subgroups |
| Fairness audit | Disparate impact + equalised opportunity | 80% rule, TPR parity, calibration per group |
| SQL | BigQuery equivalents for all analyses | Production-ready queries in `sql/` |
| Config | YAML → frozen dataclasses | Single source of truth, no magic numbers in code |
| Tests | pytest with business-logic assertions | 95 tests — not just smoke tests |

## Key Results (n=10,000, seed=42)

| Metric | Value |
|---|---|
| CV AUC (5-fold) | 0.895 ± 0.010 |
| Test AUC (LightGBM) | 0.900 |
| Test AUC (Logistic baseline) | 0.903 |
| DeLong test (LGB vs LR) | p-value + 95% CI reported |
| Brier score | 0.127 |
| Persuadables (% of users) | ~15% |
| Efficiency gain from targeting | 2–7× depending on budget |

---

## Project Structure

```
upi-behaviour-mystery/
├── app.py                         # Streamlit dashboard (9 tabs)
├── src/
│   ├── config.py                  # Typed config loader (YAML → dataclasses)
│   ├── pipeline.py                # End-to-end orchestration + Click CLI
│   ├── data/
│   │   ├── schema.py              # Data contracts + validation
│   │   ├── simulator.py           # NPCI-calibrated UPI data simulator
│   │   └── csv_loader.py          # CSV upload + validation (BYOD mode)
│   ├── features/
│   │   └── engineer.py            # sklearn-compatible feature pipeline
│   ├── models/
│   │   ├── churn.py               # LightGBM churn model with CV + DeLong
│   │   ├── uplift.py              # T-Learner causal uplift model
│   │   ├── statistical_tests.py   # DeLong test for AUC comparison
│   │   └── explainability.py      # SHAP-based model explanations
│   └── analysis/
│       ├── cohorts.py             # SQL-equivalent cohort analysis
│       ├── strategy.py            # Plain-English strategy recommender
│       ├── experiment_design.py   # A/B test power analysis
│       └── fairness.py            # Model fairness audit
├── sql/
│   ├── cohort_queries.sql         # 7 production BigQuery queries
│   └── README.md                  # Query index + pandas equivalents
├── tests/                         # 95 tests with business-logic assertions
├── config/
│   └── config.yaml                # All parameters (no magic numbers)
├── METHODOLOGY.md                 # Calibration sources + validation protocol
├── CONTRIBUTING.md                # How to contribute
├── LICENSE                        # MIT
├── pyproject.toml
├── Makefile
└── requirements.txt
```

---

## Design Decisions

**Config in YAML, not code.** Every parameter — archetype shares, LightGBM hyperparameters, cashback defaults — lives in `config/config.yaml`. Changing a parameter doesn't touch code.

**Typed dataclasses throughout.** `SimulationResult`, `ChurnModelResult`, `UpliftResult`, `StrategyReport` — all typed. No passing raw DataFrames between modules with unclear contracts.

**Schema validation at the boundary.** `validate_user_dataframe()` runs before any model sees data — whether from the simulator or a user-uploaded CSV. Bad data fails loudly at the right place.

**Pipeline is callable and testable.** `run_pipeline()` returns a `PipelineResult` dataclass. Tests call it directly. The CLI wraps it with Click. The dashboard imports the same functions. Nothing is duplicated.

**Statistical rigour.** Model comparison uses the DeLong test, not just eyeballing AUC differences. Recommendations include confidence levels. Brier score validates calibration.

---

## Why Simulation?

Real user-level UPI data is confidential. NPCI publishes monthly aggregate statistics (volume, P2M/P2P split, average ticket ₹1,293) which calibrate the simulation. The methodology — Poisson arrivals, log-normal values, archetype-based churn — transfers directly to real data by swapping `simulate_users()` for a CSV upload. The interface is identical.

See [METHODOLOGY.md](METHODOLOGY.md) for calibration sources, validation protocol, and documented limitations.

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Some areas where help is especially useful:

- **More uplift estimators** — S-Learner, X-Learner, doubly robust
- **Real-world validation** — if you have anonymized UPI retention data, we'd love to benchmark
- **Deployment** — Docker, Streamlit Cloud, or FastAPI serving layer
- **Additional simulators** — extend to other fintech verticals (lending, insurance)

## License

This project is licensed 