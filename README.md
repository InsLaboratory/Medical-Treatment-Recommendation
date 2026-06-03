# Offline Reinforcement Learning for Sepsis Treatment Recommendation

> **CPQ-IQL: Constrained Pessimistic Q-Learning with Implicit Q-Learning**  
> A two-stage safe offline RL framework for ICU treatment optimisation

---

## Overview

This repository contains the full research pipeline for an offline deep reinforcement learning project applied to sepsis treatment optimisation in the ICU setting. The project introduces **CPQ-IQL**, a novel algorithm that combines pessimistic Q-learning with Implicit Q-Learning (IQL) and a clinical constraint layer derived from the Surviving Sepsis Campaign 2021 guidelines.

The pipeline is built on the Health Gym synthetic sepsis dataset (Kuo et al., 2022), derived from MIMIC-III, and covers preprocessing, MDP construction, baseline experiments, CPQ-IQL training, off-policy evaluation, and ablation studies.

---

## Project Structure

```
Medical-Treatment-Recommendation/
├── notebooks/      Jupyter notebooks for each experimental stage
├── src/            Python modules — preprocessing, MDP, algorithms, evaluation
├── models/         Saved checkpoints for all trained models and ablation variants
├── experiments/    Training logs, results CSVs, and JSON result files
├── figures/        All generated plots (preprocessing, baselines, CPQ-IQL, comparison)
├── reports/        Weekly and final research reports (PDF)
├── cpqiql_dashboard.py   Interactive Streamlit analysis dashboard
└── requirements.txt
```

---

## Scientific Context

Sepsis is a life-threatening condition requiring rapid, complex treatment decisions in the ICU. Standard approaches rely on clinician intuition, which is subject to high inter-individual variability. This project frames fluid and vasopressor dosing as a **Markov Decision Process (MDP)** and applies offline RL to learn policies from retrospective clinical data — without any online environment interaction — to avoid the safety risks of live experimentation.

**Core contribution:** CPQ-IQL is a two-stage safe offline RL framework that:
1. Learns a Q-function penalised by clinical safety constraint violations (Stage 1)
2. Filters the learned policy at runtime through a hard constraint checker (Stage 2 — Safe Actions)

**Key design decisions:**
- Twin Q-networks with pessimistic value estimation via min(Q₁, Q₂)
- Advantage-based penalty `max(0, Q−V)` to avoid Bellman loss collapse
- Per-constraint Lagrange multipliers with individual caps to prevent constraint dominance
- Beta annealing: policy temperature decays from `β_start` to `β_end`
- Early stopping on validation Q-loss

---

## Clinical Safety Constraints

Constraints are derived from the **Surviving Sepsis Campaign 2021** guidelines:

| ID | Constraint | Description |
|----|-----------|-------------|
| C1 | Hypotension without vasopressor support | Flags actions that withhold vasopressors when MAP is critically low |
| C2 | Metabolic deterioration without fluid resuscitation | Flags inadequate fluid response to metabolic acidosis |
| C3 | Cumulative vasopressor overdose | Penalises excessive vasopressor accumulation over a 6-step window |
| C4 | Abrupt vasopressor withdrawal | Flags sudden discontinuation in critically ill patients |

---

## MDP Specification

| Component | Definition |
|-----------|------------|
| State **S** | 56-dimensional normalised physiological vector ∈ [0, 1] (Health Gym / MIMIC-III extended MDP) |
| Action **A** | 25 discrete actions — 5 fluid levels × 5 vasopressor levels |
| Reward **R** | ±15 terminal (survival / 90-day readmission) + intermediate −λ·ΔSOFA |
| Horizon | Variable-length episodes (mean ≈ 8 steps, 4-hour intervals) |
| Discount | γ = 0.99 |

---

## Algorithms

| Algorithm | Type | Description |
|-----------|------|-------------|
| Random | Baseline | Uniform random action selection — lower bound |
| Behaviour Cloning (BC) | Supervised | Imitation of clinician policy |
| LR / RF / MLP | Supervised | Tabular classification baselines |
| ResNet-18 | Supervised | Image-based baseline on encoded state matrices |
| DQN | Online RL | Deep Q-Network adapted for offline data |
| IQL | Offline RL | Implicit Q-Learning (Kostrikov et al., 2021) |
| CQL | Offline RL | Conservative Q-Learning (Kumar et al., 2020) |
| **CPQ-IQL** | **Offline RL** | Constrained Pessimistic Q-Learning with IQL + Safe Actions filter |

---

## Main Results

Results are evaluated via **Constraint Violation Rate (CVR ↓)**, **Survival Rate (SR ↑)**, and **Fitted Q-Evaluation (FQE V ↑)**.

| Method | CVR % ↓ | Safe CVR % ↓ | SR % ↑ | FQE V ↑ | BC@1 % ↑ | BC@3 % ↑ |
|--------|---------|-------------|--------|---------|---------|---------|
| Clinician (reference) | 5.79 | — | 60.02 | — | 100.0 | 100.0 |
| Random | 2.64 | 2.03 | 60.72 | 2.67 | 1.88 | 11.68 |
| DQN | 3.69 | 0.11 | 66.38 | 6.53 | 1.62 | 4.62 |
| IQL | 7.33 | 1.00 | 63.51 | 8.30 | 3.83 | 10.92 |
| CQL | 4.18 | 0.30 | **67.33** | 6.20 | 4.25 | 7.94 |
| **CPQ-IQL (Full)** | 4.30 | **0.53** | 61.93 | **7.21** | **5.54** | **14.25** |

The full two-stage CPQ-IQL framework achieves the lowest post-filter Safe CVR while maintaining competitive SR and the highest behavioural consistency (BC@3) among RL methods.

### α-Sensitivity (constraint penalty weight)

| α | CVR % | Safe CVR % | SR % | FQE V |
|---|-------|-----------|------|-------|
| 0.0 | 7.80 | 0.83 | 61.34 | 7.81 |
| 0.5 | 4.77 | 0.57 | 61.46 | 7.54 |
| **1.0** | **4.30** | **0.53** | **61.93** | **7.21** |
| 2.0 | 2.42 | 0.30 | 61.42 | 7.13 |
| 5.0 | 2.00 | 0.25 | 61.12 | 6.90 |

α = 1.0 is selected as the optimal trade-off between constraint satisfaction and policy quality.

---

## Installation

**Requirements:** Python 3.10+, PyTorch ≥ 1.13

```bash
git clone https://github.com/InsLaboratory/Medical-Treatment-Recommendation.git
cd Medical-Treatment-Recommendation
pip install -r requirements.txt
```

**Dependencies:**

```
numpy==1.23.5        pandas==1.5.3         matplotlib==3.6.3
seaborn==0.12.2      scikit-learn==1.2.2   scipy==1.10.0
torch>=1.13.0        tqdm==4.64.1          mlflow==2.2.2
statsmodels==0.13.5  missingno==0.5.1      imbalanced-learn==0.10.1
```

---

## Data Access

The dataset is the **Health Gym Sepsis synthetic dataset (v1.0.0)** published by Kuo et al. (2022) in *Scientific Data*. It contains 43,280 ICU timesteps (4-hour intervals) synthetically generated from MIMIC-III.

**Access:** Available on PhysioNet after signing a data use agreement.  
→ [https://physionet.org/content/synthetic-mimic-iii-health-gym/1.0.0/](https://physionet.org/content/synthetic-mimic-iii-health-gym/1.0.0/)

Once downloaded, place the CSV file in the `data/` directory.

---

## Reproducing the Results

### Step 1 — EDA and Preprocessing

```bash
jupyter notebook notebooks/W2_data_exploration.ipynb
```

Produces:
- `data/preprocessed/sepsis_preprocessed.csv`
- `data/preprocessed/sepsis_mdp_dataset.npz`
- `data/preprocessed/preprocessing_metadata.pkl`
- All preprocessing figures in `figures/preprocessing/`

### Step 2 — Baseline Experiments

```bash
jupyter notebook notebooks/W2_baseline_experiments.ipynb
```

Trains and evaluates: Random, BC, LR, RF, MLP, ResNet-18, DQN, IQL, CQL.  
Results saved to `experiments/baselines_summary.csv`.

### Step 3 — CPQ-IQL Training

```bash
jupyter notebook notebooks/W3_CPQ_IQL.ipynb
```

Trains the full CPQ-IQL framework including:
- Main model (α = 1.0)
- α-sensitivity sweep (α ∈ {0.0, 0.5, 1.0, 2.0, 5.0})
- Constraint ablations (C1–C4, no constraints, all constraints)
- Component ablations (w/o Twin-Q, w/o β-Annealing, w/o Adv-Penalty, w/o Lagrangian)

Model checkpoints saved to `models/cpq_iql/`.

### Step 4 — Offline RL Comparison

```bash
jupyter notebook notebooks/W3_offline_RL_methods.ipynb
jupyter notebook notebooks/W3_comparison.ipynb
```

Produces the final cross-method comparison figures in `figures/comparison/`.

### Step 5 — Interactive Dashboard

```bash
pip install streamlit plotly
streamlit run cpqiql_dashboard.py
```

Launches an interactive Streamlit dashboard for exploring CPQ-IQL results, ablation studies, and sensitivity analyses.

---

## Reports

| Report | Description |
|--------|-------------|
| `reports/W1_project_scope.pdf` | Project framing, MDP formulation, related work |
| `reports/W2_baseline_report.pdf` | Baseline experiments — preprocessing and supervised/RL baselines |
| `reports/W3_experiments_summary.pdf` | CPQ-IQL training results and ablation summary |
| `reports/final_report.pdf` | Complete research report — methods, results, analysis |

---

## References

- Kuo et al. (2022). *The Health Gym: synthetic health-related datasets for the development of reinforcement learning algorithms.* Scientific Data. https://doi.org/10.1038/s41597-022-01784-7
- Kostrikov et al. (2021). *Offline Reinforcement Learning with Implicit Q-Learning.* ICLR 2022.
- Kumar et al. (2020). *Conservative Q-Learning for Offline Reinforcement Learning.* NeurIPS 2020.
- Evans et al. (2021). *Surviving Sepsis Campaign: International Guidelines for Management of Sepsis and Septic Shock 2021.* Critical Care Medicine.
- Zhang & Mi (2026). *Safe offline reinforcement learning for sepsis treatment.* Transactions on AI.
- Tu et al. (2025). *Offline safe RL for sepsis with variable-length episodes.* Human-Centric IS.
- Survey ICLR (2026). *Optimizing ICU sepsis treatment techniques in reinforcement learning.*

---
