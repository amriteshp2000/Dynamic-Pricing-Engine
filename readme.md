# Trust-Aware Causal Dynamic Pricing Engine

## A Bayesian Contextual Bandit System with Safety Constraints

## Overview

This repository implements a **production-oriented dynamic pricing system** using a **contextual multi-armed bandit** framework with Bayesian learning.  
The system is designed for **short-horizon revenue maximization under uncertainty**, where demand feedback is sparse, delayed, and non-stationary.

Unlike rule-based or static ML pricing approaches, pricing decisions here are made **online**, balancing exploration and exploitation using **Thompson Sampling**, while remaining compatible with **off-line scientific estimation and a safety governor**.

The architecture mirrors how pricing systems are deployed in mature marketplaces rather than academic reinforcement learning benchmarks.

---

## Problem Statement

Given a listing with contextual features `x_t`, select a price `p_t` at time `t` to maximize expected revenue:

$$\max_{p_t} \; \mathbb{E}[p_t \cdot \mathbb{1}(\text{booking}) \mid x_t]$$

Key challenges addressed:

- Sparse and binary feedback (booking / no booking)
- Cold-start listings with no historical data
- Non-stationary demand due to seasonality and events
- Safety requirements preventing extreme or unstable prices
- Need for explainable, auditable decision logic


---

## Endogeneity in Pricing Data

In historical marketplace logs, price is endogenous. High prices frequently coincide with periods of high demand (e.g., holidays, local events), inducing a spurious correlation:

`Price <-> Demand`

Predictive models trained directly on such data often learn an incorrect positive relationship between price and booking probability, leading to unstable pricing behavior in production.

### Causal Estimation Approach

We explicitly estimate the causal effect of price on demand (intervention notation):

$$P(\text{Booking} \mid do(\text{Price}))$$

using Double Machine Learning (DML). Orthogonalization removes bias from observed confounders such as seasonality, location, and listing attributes, yielding a stable estimate of price elasticity.

Empirically, the recovered elasticity is:

$$\beta \approx -0.035$$

which is consistent with standard economic expectations for short-term accommodation markets.

---

## System Architecture

The system enforces a strict separation between **offline scientific estimation** and **online control**, ensuring reproducibility, auditability, and low-latency serving.

```
Offline / Batch (Science)
------------------------
Raw Logs
   ↓
Data Audit & Validation
   ↓
Causal Identification (DML)
   ↓
Hierarchical Demand Modeling
   ↓
Model Registry

Online / Real-Time (Control)
----------------------------
Pricing Request
   ↓
Contextual Thompson Sampling
   ↓
Safety & Compliance Governor
   ↓
Final Price Output
```

Offline components are retrained periodically and versioned. Online components operate in real time with deterministic latency bounds.

---

## Installation

### Requirements

- Python 3.10 or newer
- pip or conda

### Environment Setup

```bash
git clone https://github.com/yourusername/pricing-engine.git
cd pricing-engine

python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

pip install -r requirements.txt
```

---

## Data

The system is evaluated using **real historical marketplace data** (Seattle Airbnb, 2016 snapshot). This version is chosen because pricing and availability fields are fully populated and consistent.

Required files:

- `calendar.csv`
- `listings.csv`

These should be placed in the project root or configured via environment variables.

---

## Repository Structure

```
pricing_engine/
├── audit.py            # Market Audit (Identifiablity, sparsity, Confounding)
├── causal_model.py     # Causal elasticity estimation (Double ML)
├── demand_model.py     # Hierarchical Bayesian demand model
├── bandit.py           # Contextual Thompson Sampling agent
├── safety.py           # Low-latency constraint enforcement
├── data_loader.py      # ETL and validation utilities
├── offline_eval.py     # Doubly robust offline evaluation
├── benchmark.py        # Benchmark of the whole system

notebooks/
├── 00_data_audit.ipynb
├── 01_causal_identification.ipynb
├── 02_demand_modeling.ipynb
├── 03_bandit_adaptation.ipynb
├── 04_safety_validation.ipynb
├── 05_offline_evaluation.ipynb
├── 06_benchmark_test.ipynb
```

---

## Module Breakdown

### Module 01 — Data Audit

**Purpose**  
Prepare stable, leakage-free contextual features for both offline estimation and online inference.

**Key Components**
- Temporal features (day-of-week, seasonality)
- Listing attributes (capacity, location clusters)
- Market context (local demand intensity)
- Log-transformed price features to stabilize variance

**Design Notes**
- No future leakage is permitted
- Feature schemas are shared across offline and online pipelines

---

### Module 02 — Cause Identification

**Purpose**  
Estimate baseline demand sensitivity and price elasticity from historical data.

**Approach**
- Regularized regression models for demand estimation
- Causal controls to mitigate confounding between price and bookings
- Aggregation at neighborhood / city levels to build robust priors

**Outputs**
- Hierarchical Bayesian priors for cold-start listings
- Price elasticity distributions
- Feasible price bounds for safety constraints

Offline models **do not** set prices directly.  
They inform the bandit via priors only.

---

### Module 03 — Contextual Pricing Bandit

#### Bandit Formulation

- **Context** `x_t`: listing + temporal + market features  
- **Action** `p_t`: candidate price  
- **Reward**:

$$r_t = p_t \cdot \mathbb{1}(\text{booking})$$

The problem is treated as a **contextual bandit**, not full reinforcement learning, as actions do not induce long-horizon state transitions.


#### Bayesian Demand Model

Demand is modeled using **Streaming Bayesian Ridge Regression**:

$$\mathbb{P}(\text{booking} \mid x, p) = f(x, \log p)$$

Posterior updates:

$$A_t = \lambda A_{t-1} + x_t x_t^\top,\quad
b_t = \lambda b_{t-1} + y_t x_t$$

Where:
- `lambda` is an exponential forgetting factor
- Enables adaptation to non-stationary demand
- Prevents overfitting to stale data


#### Thompson Sampling Policy

At each decision step:

1. Sample parameters from the posterior:

$$\tilde{\theta}_t \sim \mathcal{N}(\mu_t, \Sigma_t)$$

2. Evaluate expected revenue for candidate prices:

$$\hat{r}(p) = p \cdot \mathbb{P}_{\tilde{\theta}_t}(\text{booking} \mid x_t, p)$$

3. Select:

$$p_t^* = \arg\max_p \hat{r}(p)$$

Exploration arises **naturally from posterior uncertainty**, eliminating heuristic exploration schedules.



#### Cold Start via Prior–Online Fusion

For new listings, posterior parameters are initialized using **precision-weighted fusion**:

$$\mu = \frac{\mu_{\text{prior}}/\sigma_{\text{prior}}^2 + \mu_{\text{online}}/\sigma_{\text{online}}^2}
{1/\sigma_{\text{prior}}^2 + 1/\sigma_{\text{online}}^2}$$

As online evidence accumulates, the model **automatically transitions** to listing-specific learning without manual thresholds.



#### Panic Mechanism (Demand Collapse Protection)

To handle sparse negative feedback:

- Consecutive no-booking outcomes are tracked
- After a threshold, the effective price ceiling is exponentially decayed
- Forces exploration in lower price regions

This prevents runaway pricing strategies under uncertainty while remaining bounded and reversible.

---


### Module 04 — Safety Governor

**Purpose**  
Enforce deterministic constraints on bandit output.

**Constraints**
- Price floors and ceilings
- Max daily price change
- Elasticity sanity checks
- Regulatory or business rules

The bandit proposes; the governor validates.  
This separation ensures explainability and fault isolation.

---

### Module 05 — Offline Evaluation & Trust Verification

**Objective**  
Evaluate counterfactual revenue impact while explicitly verifying that the pricing **bandit** behaves within acceptable trust, safety, and stability boundaries.

This system treats *trust* as a measurable, enforceable property rather than a qualitative side effect of optimization.


#### Doubly Robust Revenue Estimation

Policy value is estimated using a **Doubly Robust (DR)** estimator that combines:

- Direct outcome modeling from the causal demand model
- Bias correction using logged historical outcomes when the agent’s action is close to the logged price

For each replayed decision:

$$\hat{V}_{DR}(p_{agent}) =
\hat{r}(p_{agent}) +
\mathbb{1}(|p_{hist} - p_{agent}| < \epsilon)
\cdot
(r_{obs} - \hat{r}(p_{hist}))$$

This estimator remains unbiased if **either** the demand model or the historical logging process is correctly specified.


#### Trust Metrics (Behavioral Guarantees)

In addition to revenue uplift, the evaluation produces a structured **Trust Report** capturing behavioral risk:

- **Override Rate**  
  Fraction of bandit prices deviating more than 30% from historical host prices.  
  Serves as a proxy for host trust and acceptance risk.

- **Safety Clamp Rate**  
  Percentage of bandit outputs modified by the Safety Governor.  
  Expected during early rollout and monitored as a leading indicator of instability.

- **Safety Violations (Critical)**  
  Hard constraint breaches after governance.  
  Any non-zero value is considered a deployment-blocking failure.

- **Volatility Reduction**  
  Relative reduction in price variance compared to historical pricing.  
  Acts as a proxy for perceived price stability and user trust.

These metrics ensure the **contextual bandit** optimizes revenue *without* exhibiting erratic or unsafe behavior.


#### Evaluation Protocol

- Historical data is replayed row-by-row to preserve realistic constraints
- The **pricing bandit** proposes candidate prices
- The **Safety Governor** deterministically validates and clamps outputs
- Revenue and trust metrics are computed jointly

The final output is a strongly typed `TrustMetrics` report suitable for rollout gating, experimentation review, and executive reporting.



---

## Why Contextual Bandits (Not RL)?

This design choice is intentional:

- Superior sample efficiency under sparse rewards
- No Markovian state assumptions
- Easier offline validation
- Predictable, explainable exploration behavior

This mirrors industry pricing systems used in marketplaces, travel, and ad auctions.

---
## Engineering Considerations

- Fully online posterior updates (O(d²))
- Stateless inference-compatible design
- Clear failure modes and fallbacks
- Model decisions are inspectable and reproducible

---

## Empirical Results (Seattle 2016)

Evaluation on 50 held-out listings:

| Metric | Value | Notes |
|------|------|------|
| Revenue Uplift (DR) | +40.37% | Relative to static pricing baseline |
| Safety Violations | 0 | All constraints satisfied |
| P99 Latency | 2.2 μs | Online decision path |
| Governor Override Rate | 93.5% | Indicates conservative rollout behavior |

---

## Design Principles

- Causal estimation over correlation
- Explicit uncertainty modeling
- Cold-start safety by construction
- Constraint-first decision-making
- Auditable, modular components

## ⚖️ License
Distributed under the MIT License. See LICENSE for more information.
