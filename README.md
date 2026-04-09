# Macro Pipeline

Production data pipeline that ingests macroeconomic indicators from 27 government and central bank APIs across 8 G10 currencies, computes derived analytics with point-in-time correctness, and outputs normalised FX trading signals on a 5-minute polling cadence.

## Overview

This system replaces a manual macro research workflow with an automated, serverless pipeline running on AWS Lambda. It fetches ~300 economic time series (GDP, CPI, industrial production, labour market, retail sales, business confidence, yield curves, policy rates) from statistical agencies worldwide, stores them in PostgreSQL with full revision tracking, then runs a multi-layer signal generation engine that produces cross-sectionally normalised composite scores for all 28 G10 currency pairs. The pipeline runs every 5 minutes, detects new data releases automatically, and sends structured email alerts with updated signal ladders within minutes of publication. It is designed for a single operator managing systematic FX positions.

## Architecture

```
                         EventBridge (5-min cron)
                                  |
                                  v
                     +------------------------+
                     |    AWS Lambda (macro)   |
                     |    lambda_function.py   |
                     +------------------------+
                                  |
              +-------------------+-------------------+
              |                                       |
              v                                       v
   +---------------------+                +---------------------+
   |  Fetch / Clean Loop |                |  AWS Secrets Manager|
   |  27 providers, ~300 |                |  DB creds, API keys |
   |  series definitions |                +---------------------+
   +---------------------+
              |
              v
   +---------------------+
   |  PostgreSQL (RDS)   |
   |  macro.series_data  |
   |  revision tracking  |
   +---------------------+
              |
   +----------+----------+----------+
   |          |          |          |
   v          v          v          v
 Layer 1   Layer 1   Layer 1   Layer 1
 Growth    Labour    Monetary  Rates
 Derived   Derived   Derived   Derived
   |          |          |          |
   v          v          v          v
 Layer 2   Layer 2   Layer 2   Layer 2
 Growth    Labour    Monetary  Rates
 Signals   Signals   Signals   Signals
 (MAD z)   (MAD z)   (MAD z)   (MAD z)
   |          |          |          |
   +----------+----------+----------+
              |
              v
   +---------------------+
   | Layer 3: Composite  |
   | Cross-sectional z   |
   | 8 countries -> 28   |
   | pair signals         |
   +---------------------+
              |
              v
   +---------------------+
   |  Email Alerts       |
   |  Release detection  |
   |  Daily digest       |
   |  Failure alerts     |
   +---------------------+
```

## Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.13 | Core language, all pipeline logic |
| AWS Lambda | Serverless execution, 5-min EventBridge schedule |
| PostgreSQL (RDS) | Time series storage with revision tracking |
| AWS Secrets Manager | Credential management with 5-min TTL cache |
| Docker | Lambda container image (ECR deployment) |
| psycopg2 | Direct PostgreSQL wire protocol, no ORM overhead |
| pandas / NumPy | Time series transforms and panel operations |
| requests | HTTP client with retry/backoff per provider |
| Mailjet (SMTP) | Transactional email for alerts and digests |
| EventBridge | Cron scheduling for incremental, digest, and equity runs |

## Key Features

- **27 data providers** with a unified fetch/clean interface: BLS, FRED, BEA, ECB, Eurostat, ONS, StatCan, ABS, BoJ, METI, and 17 others. Each provider implements `BaseProvider.fetch()` and `BaseProvider.clean()` behind a registry pattern, so the orchestrator is provider-agnostic.

- **Point-in-time signal construction.** Derived metrics carry `estimated_release_date` columns, and the signal layer pivots on release date rather than observation date. This prevents look-ahead bias in backtests and ensures signals reflect only publicly available information at each point in time.

- **MAD-based panel z-scoring** aligned to JPMaQS methodology. Expanding-window mean absolute deviation pooled across all currencies, with monthly re-estimation, in-sample backfill, and winsorisation at +/-3 MADs. Configurable neutral (zero, mean, median) and panel weighting.

- **Two-pass normalisation architecture.** Pass 1: time-series z-scores on each constituent metric at native frequency. Pass 2: cross-sectional z-score across 8 currencies at each date, converting absolute macro strength into relative currency ranking. No intermediate rescoring, no compounding normalisation artifacts.

- **Revision tracking and audit trail.** Every upsert preserves `previous_value` and `revised_at` so data revisions (common in GDP, employment) are captured automatically. Pipeline runs are logged to `macro_run_state` with STARTED/SUCCESS/FAILED/SKIPPED lifecycle and failure diagnostics.

- **Required-series gating.** Series are marked `required` or optional in JSON configs. If any required series fails, the pipeline halts before derived computation (DERIVED_GATE), preventing stale signals from propagating. Optional failures are logged and reported but do not block downstream stages.

- **Hardened HTTP with per-provider tuning.** Shared session with 6 retries, exponential backoff, 429/5xx retry, and connection pooling. Slow providers (METI) get reduced retries to avoid burning Lambda time. BLS batch pre-fetch pulls up to 50 series in a single API call.

- **Release detection and alerting.** After each run, the pipeline checks for series whose values changed in the last 10 minutes, groups them by publication name, and sends a structured HTML email with the updated pair signal ladder. Deduplication via `release_alerts_log` prevents repeat emails across polling cycles.

## Project Structure

```
├── lambda_function.py           # Lambda entry point (macro pipeline)
├── lambda_function_equity.py    # Lambda entry point (equity pipeline)
├── Dockerfile                   # Lambda container image build
├── deploy.ps1                   # ECR build/push/deploy script (macro)
├── deploy-equity.ps1            # ECR build/push/deploy script (equity)
├── requirements.txt             # Python dependencies
│
├── pipeline/
│   ├── engine/                  # Core orchestration layer
│   │   ├── core.py              # Main pipeline: fetch loop, derived stages, error handling
│   │   ├── config.py            # Environment-based config (frozen dataclass)
│   │   ├── secrets.py           # AWS Secrets Manager client with TTL cache
│   │   ├── db_config.py         # PostgreSQL connection with TCP keepalive
│   │   ├── http.py              # Retry-enabled HTTP session factory
│   │   ├── series.py            # Series definition loader (JSON catalog)
│   │   ├── insert_to_db.py      # Generic upsert with revision tracking
│   │   ├── run_state.py         # Pipeline audit trail (STARTED -> SUCCESS/FAILED)
│   │   ├── email_alerts.py      # Failure emails, daily digest, pair signal ladder
│   │   ├── release_alert.py     # New data release detection and alerting
│   │   ├── release_dates.py     # Publication lag mapper for PIT correctness
│   │   ├── logger.py            # Structured logging with secret scrubbing
│   │   ├── run_incremental.py   # CLI: incremental run
│   │   ├── run_reconcile.py     # CLI: full-history reconciliation
│   │   ├── run_backfill.py      # CLI: historical backfill
│   │   └── run_digest.py        # CLI: daily digest email
│   │
│   ├── providers/               # Data source adapters (27 providers)
│   │   ├── base.py              # Abstract BaseProvider interface
│   │   ├── registry.py          # Provider registry and lookup
│   │   ├── bls.py               # US Bureau of Labor Statistics (batch API)
│   │   ├── fred.py              # Federal Reserve Economic Data
│   │   ├── bea.py               # US Bureau of Economic Analysis
│   │   ├── ecb.py               # European Central Bank
│   │   ├── eurostat.py          # Eurostat (EU statistics)
│   │   ├── ons.py               # UK Office for National Statistics
│   │   ├── boj.py               # Bank of Japan
│   │   └── ...                  # + 19 more national statistics agencies
│   │
│   ├── series/                  # Series catalog (JSON per currency)
│   │   ├── usd.json             # ~60 US series (CPI, GDP, NFP, retail, IP, ...)
│   │   ├── eur.json             # Eurozone series
│   │   ├── gbp.json             # UK series
│   │   └── ...                  # + AUD, CAD, JPY, NZD, CHF, yields, policy rates
│   │
│   ├── derived/                 # Layer 1: raw metric transforms
│   │   └── macro/
│   │       ├── growth_derived.py    # GDP/IP/retail YoY, excess vs trend, BCI
│   │       ├── labour_derived.py    # Employment, wages, unemployment
│   │       ├── monetary_derived.py  # Money supply, credit growth
│   │       ├── rates_derived.py     # Yield curves, real rates, carry
│   │       └── yields_derived.py    # Nominal/real yield differentials
│   │
│   ├── signals/                 # Layer 2-3: z-scored signals and composites
│   │   ├── zn_scores.py         # MAD z-score engine (JPMaQS-aligned)
│   │   ├── growth_signals.py    # Growth block: output + wages factors
│   │   ├── labour_signals.py    # Labour block scoring
│   │   ├── monetary_signals.py  # Monetary block scoring
│   │   ├── rates_signals.py     # Rates block scoring
│   │   └── composite.py         # Country composites + 28 pair signals
│   │
│   ├── equities/                # Separate equity pipeline
│   │   ├── core.py              # Equity orchestrator
│   │   ├── prices.py            # Price data ingestion
│   │   └── ...                  # Breadth, relative strength, VCP screening
│   │
│   └── runners/                 # Signal runner entry points
│       ├── run_signals.py
│       └── run_signals_v2.py
│
└── backtesting/                 # Backtest framework and reports
    ├── backtest_engine.py       # Backtest runner
    ├── backtest_production.py   # Production signal backtests
    └── reports/                 # Generated PDF backtest reports
```

## Setup

### Environment Variables

```
# Required
DB_SECRET_ID=<aws-secrets-manager-secret-id>
AWS_REGION=<aws-region>

# API keys (one per provider that requires authentication)
API_SECRET__FRED=<secrets-manager-secret-name>
API_SECRET__BLS=<secrets-manager-secret-name>
API_SECRET__BEA=<secrets-manager-secret-name>
API_SECRET__EODHD=<secrets-manager-secret-name>

# Email alerts (optional — pipeline runs without these)
SMTP_USER=<mailjet-api-key>
SMTP_PASS=<mailjet-secret-key>
SENDER_EMAIL=<sender-address>
RECEIVER_EMAIL=<recipient-address>

# Optional overrides
ENV=local|production
DB_SCHEMA=macro
STRICT_DB=true
LOOKBACK_MONTHS=24
```

The database secret in AWS Secrets Manager should contain: `host`, `port`, `dbname`, `username`, `password`.

### Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Incremental run (last 24 months)
python -m pipeline.engine.run_incremental

# Single currency
python -m pipeline.engine.run_incremental --currency usd

# Full historical backfill
python -m pipeline.engine.run_backfill

# Monthly reconciliation (catch revisions)
python -m pipeline.engine.run_reconcile
```

### Deployment

The pipeline deploys as a Docker container image to AWS Lambda via ECR:

```bash
./deploy.ps1    # macro pipeline
```

EventBridge schedules: incremental every 5 minutes, daily digest once per day, equity pipeline after US market close.
