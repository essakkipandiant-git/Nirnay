# NIRNAY — Infrastructure Risk Intelligence & Decision Support

NIRNAY is an AI-powered infrastructure project risk intelligence platform designed to help administrators monitor large government project portfolios, identify potential schedule and cost risks, understand the factors driving those risks, and prioritize corrective action.

## Live Demo

**Vercel:** [https://nirnaysih.vercel.app/](https://nirnaysih.vercel.app/)

## Problem

Large infrastructure projects generate huge amounts of financial, schedule, progress, and reporting data. Traditional monitoring can make it difficult to quickly identify which projects require immediate attention.

Key challenges:

* Detecting potential delays early
* Identifying cost and schedule risks
* Understanding why a project is risky
* Monitoring thousands of projects across ministries and sectors
* Converting risk information into actionable interventions

## Solution

NIRNAY transforms project-monitoring data into an integrated decision-support workflow:

```text
Project Data
     ↓
Data Processing & Validation
     ↓
Feature Engineering
     ↓
ML Risk Prediction
     ↓
Risk Classification
     ↓
Explainable Risk Drivers
     ↓
Early Warning Signals
     ↓
Intervention Recommendations
     ↓
Administrative Decision
     ↓
Continuous Monitoring
```

NIRNAY is **not only an AI prediction dashboard**. It connects prediction with explanation, early warning, intervention, and project monitoring.

## Key Features

### Portfolio Intelligence

* Portfolio-wide project monitoring
* Risk distribution
* Ministry and sector analysis
* Cost and schedule risk visibility
* Project search and filtering

### AI-Based Risk Prediction

* ML-based delay/risk probability
* Low, Medium, High, and Critical risk classification
* Project-level risk assessment
* Model-derived risk drivers

### Explainable AI

NIRNAY helps answer:

> **Why is this project considered risky?**

Feature-contribution/explainability information helps identify important factors behind predictions.

### Early Warning Signals

Potential signals include:

* Schedule deterioration
* Progress stagnation
* Cost escalation
* Financial/physical progress mismatch
* High remaining work relative to available time

### Project Intelligence

Each project can be explored through:

* Overview
* Financial
* Schedule
* Milestones
* Risk Analysis
* Interventions
* History

### Intervention Center

Risk signals can be translated into possible administrative actions.

Example:

```text
Schedule Risk
    ↓
Schedule Review / Acceleration Plan

Cost Risk
    ↓
Financial Review
```

NIRNAY supports decision-making; it does not automatically make administrative decisions.

### Historical Monitoring

Historical/monthly project information can be used to understand how project conditions and risk indicators change over time.

### Scenario / What-If Analysis

Selected project conditions can be changed to compare scenario-based estimated risk.

## Dataset

The system works with structured infrastructure project-monitoring data containing information such as:

* Project ID and project name
* Ministry
* Sector
* Approved cost
* Revised cost
* Expenditure
* Physical progress
* Financial progress
* Start and completion dates
* Project status
* Historical/monthly reporting information

Derived datasets are used for feature engineering and ML prediction.

## Machine Learning

The current prediction pipeline uses a **HistGradientBoosting-based classifier** for structured/tabular project data.

```text
Raw Project Data
      ↓
Cleaning & Validation
      ↓
Feature Engineering
      ↓
Preprocessing
      ↓
HistGradientBoosting Model
      ↓
Risk / Delay Probability
      ↓
Risk Thresholds
      ↓
Risk Category
```

The model output is a probability estimate, not a guaranteed future outcome.

## Explainability

NIRNAY uses feature-contribution information to make model predictions more understandable.

Instead of only showing:

```text
Risk: Critical
```

the system can provide context such as:

```text
Major Risk Drivers
• Remaining time vs overrun
• Cost variation
• Progress-related factors
```

## Technology Stack

* **Frontend:** Next.js
* **UI:** React + TypeScript
* **Styling:** Tailwind CSS
* **Icons/UI:** Lucide and reusable UI components
* **Machine Learning:** Python + HistGradientBoosting
* **Data:** CSV / JSON datasets
* **Deployment:** Vercel

## System Architecture

```text
                 Government Project Data
                           │
                           ▼
                 Data Processing
                           │
                           ▼
                  Feature Engineering
                           │
                           ▼
                    ML Prediction
                           │
                           ▼
                Risk Probability
                           │
                           ▼
                Risk Classification
                     │           │
                     ▼           ▼
              Explainability  Early Warnings
                     │           │
                     └─────┬─────┘
                           ▼
                  Intervention Engine
                           │
                           ▼
                 Administrative Review
                           │
                           ▼
                     Action & Monitoring
```

## Risk Interpretation

Risk categories are generated from model outputs and configured thresholds.

A risk prediction should be interpreted as:

> **An analytical signal requiring appropriate review, not a guaranteed event.**

Completed projects can be useful for historical analysis and future model validation, while active projects are the primary focus for current monitoring and intervention.

## Running Locally

### Prerequisites

* Node.js
* pnpm
* Git

### Installation

```bash
git clone https://github.com/essakkipandiant-git/Nirnay.git
cd Nirnay
pnpm install
```

### Development

```bash
pnpm dev
```

Open:

```text
http://localhost:3000
```

### Production Build

```bash
pnpm build
pnpm start
```

## Project Structure

```text
Nirnay/
├── app/                         # Next.js application
├── components/                 # UI components
├── lib/                        # Data and application utilities
├── Dataset/                    # ML/project datasets
├── NIRNAY_PAIMANA_Dataset.../  # PAIMANA-derived datasets
├── models/                     # ML artifacts and metadata
├── outputs/                    # ML outputs and evaluation results
├── public/                     # Static assets/data
├── train_nirnay_model.py       # ML training pipeline
├── package.json
├── pnpm-lock.yaml
└── README.md
```

## Human-in-the-Loop Design

NIRNAY follows a human-in-the-loop approach:

```text
AI Prediction
      ↓
Human Review
      ↓
Administrative Decision
      ↓
Action
```

The platform assists administrators rather than replacing them.

## Limitations

* Model performance depends on training-data quality and representativeness.
* Missing or inconsistent project information can affect predictions.
* Predictions require periodic validation and recalibration.
* Early warnings are analytical signals, not guarantees.
* Production deployment requires appropriate authentication, authorization, security, audit logging, and governance controls.

## Future Scope

* Real-time government data integration
* Automated periodic model inference
* Continuous model retraining using validated outcomes
* Better probability calibration
* Ministry/sector-specific models
* Advanced time-series forecasting
* Role-based access control
* Automated notifications
* Human feedback loops
* Improved outcome-based model validation

## SIH Value Proposition

NIRNAY helps move infrastructure monitoring from:

**Reactive Monitoring → Predictive & Proactive Decision Support**

The central workflow is:

**Predict → Explain → Warn → Recommend → Act → Monitor**

## Disclaimer

NIRNAY is a technology demonstration and decision-support prototype. Model outputs should be validated against authoritative operational data and reviewed by qualified officials before being used for real-world administrative decisions.

---

## Live Application

**[https://nirnaysih.vercel.app/](https://nirnaysih.vercel.app/)**
