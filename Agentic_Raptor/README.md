# RAPTOR

*Retrieval-Augmented Preference-Guided Multi-Agent Topology and Sizing Optimization via Reinforcement Learning for Analog Circuits*

RAPTOR is an open-source multi-agent framework for jointly coordinating analog circuit topology generation and circuit sizing under a shared SPICE budget. The framework combines retrieval-augmented generation (RAG), an SFT-adapted large language model (LLM), topology refinement, contextual-bandit selection, model-based soft actor-critic (MB-SAC) sizing, surrogate guidance, preference ranking, PVT analysis, and final SPICE verification.

The central goal is to reduce expensive SPICE evaluations while improving feasibility, figure of merit (FoM), runtime, and robustness.

## Overview

Analog circuit design requires two strongly coupled decisions:

- **Topology generation** determines the circuit structure.
- **Circuit sizing** determines transistor dimensions, bias parameters, compensation components, and other continuous design variables.

A topology that looks structurally reasonable may still fail after sizing, while an effective sizing method cannot compensate for a topology that lacks the capability to meet the requested specifications. RAPTOR therefore coordinates both stages in one design loop rather than treating them independently.

## Architecture

```text
Design Specifications
        |
        v
Topology Generation Agent
  - specification analysis
  - RAG retrieval
  - SFT-adapted LLM generation
        |
        v
Topology Critic Agent
  - structural validation
  - diversity checks
  - specification-aware refinement
        |
        v
Topology Selection Agent
  - 24-D context representation
  - contextual-bandit ranking
  - select two structurally diverse candidates
        |
        v
Circuit Sizing Agent
  - probe both candidates
  - allocate SPICE budget
  - MB-SAC continuous sizing
  - surrogate-assisted action selection
        |
        v
Preference Ranking
        |
        v
PVT Analysis
        |
        v
Final SPICE Verification
        |
        +---- recovery attempt if verification fails
```

### 1. Topology Generation Agent

The Topology Generation Agent creates the initial candidate topology pool from the target gain, UGBW, phase margin, and load capacitance requirements.

Before generation, RAPTOR retrieves previously measured designs with specifications close to the current target. Retrieval uses the normalized distance

```math
\frac{|pm-pm_i|}{15} + \frac{|gain-gain_i|}{30}
```

Up to six stable prior designs are retrieved, with at most two examples from the same topology family to preserve diversity. The retrieved designs are used as context rather than being directly inserted into the candidate pool.

The target specifications, retrieved examples, structural constraints, and generation guidance are then provided to a LoRA-adapted Qwen3-4B-Instruct-2507 model trained with supervised fine-tuning (SFT) for 1,500 steps. The model generates structured topology graphs that are checked for invalid connections and duplicate structures.

### 2. Topology Critic Agent

The Topology Critic Agent evaluates and refines generated candidates before sizing. It checks:

- structural validity,
- stage-count diversity,
- compensation diversity, and
- whether the candidate set contains structures appropriate for the requested gain and stability requirements.

When necessary, the critic sends feedback to the Topology Generation Agent for another limited generation round. New candidates are validated and deduplicated before being added to the topology pool.

### 3. Topology Selection Agent

The Topology Selection Agent ranks validated candidates using a contextual-bandit model.

Each topology is represented by a 24-dimensional context vector containing structural characteristics and normalized target specifications. Expected sizing reward is estimated using an offline ridge-regression model:

```math
\mathbf{w}^{\top}\tilde{\mathbf{z}}_i+b
```

The trained model is fixed during evaluation. RAPTOR selects the highest-ranked topology and, when possible, a second candidate from a different stage-count/compensation family. These two candidates form the primary and alternative sizing branches.

### 4. Circuit Sizing Agent

The Circuit Sizing Agent optimizes the two selected topologies under the remaining SPICE budget.

It first probes both candidates with a small number of SPICE simulations and compares their performance. More simulation budget is then assigned to the stronger candidate, while a small reserve is kept for refinement or recovery.

Sizing is formulated as a reinforcement-learning problem. The state contains the target specifications, remaining simulation budget, best feasibility reward, and current specification margins. Continuous actions modify parameters such as transistor dimensions, bias current, compensation capacitance, and resistance.

The sizing reward is

```math
r_{\mathrm{PM}} + r_{\mathrm{gain}} + 0.5\,r_{\mathrm{UGBW}} + b_{\mathrm{pass}}
```

where $b_{\mathrm{pass}}=1$ when all target specifications are satisfied.

RAPTOR uses model-based soft actor-critic (MB-SAC) for continuous sizing. An online surrogate predicts circuit performance before SPICE simulation and helps select promising candidate actions. SPICE remains the final evaluation authority.

After sizing, invalid or unstable designs are removed. Feasible candidates are preference-ranked using specification margins, FoM, and efficiency. The remaining budget may be used to further improve the selected design before PVT analysis and final SPICE verification.

If the selected design fails final verification, RAPTOR performs one recovery attempt using the alternative topology while reserving one SPICE call for final verification.

## Experimental Setup

### Topology generation

- **Benchmark:** Heldout29
- **Specifications:** 29
- **Seeds:** 3
- **Total runs per method:** 87
- **Process:** SKY130
- **Nominal condition:** TT, $V_{\mathrm{DD}}=1.8$ V
- **Gain targets:** 42.9–108.3 dB
- **UGBW targets:** 10 kHz–1 MHz
- **Phase-margin targets:** 45–60 degrees
- **Load capacitance:** 50–1000 pF
- **SPICE budget:** 65 calls
- **PVT corners:** $\{\mathrm{SS},\mathrm{TT},\mathrm{FF}\}$, $V_{\mathrm{DD}}\in\{1.62,1.98\}$ V, $T\in\{0,70\}^{\circ}\mathrm{C}$, for 12 conditions per design

The topology search space contains five amplifier families:

- 2-stage, no compensation
- 2-stage, Miller compensation
- 2-stage, RC compensation
- 3-stage, Miller compensation
- 3-stage, RC compensation

### Circuit sizing

- **Benchmark:** 1,000 current-mirror OTA specifications
- **Process model:** 45 nm BPTM
- **$V_{\mathrm{DD}}$:** 1.2 V
- **$C_L$:** 1 pF
- **Gain targets:** 20–55 dB
- **UGBW targets:** 1–30 MHz
- **Phase-margin targets:** 45–60 degrees
- **Maximum bias current:** $2\times10^{-4}$–$2\times10^{-2}$ A
- **SPICE budget:** 120 calls
- **Seeds:** 3
- Up to 10 feasible designs are searched for each target

## Baselines

### Topology generation

RAPTOR is compared with:

- AnalogCoder-Pro
- CktGen
- AnalogToBi

### Circuit sizing

RAPTOR is compared with:

- **ORACLE-Cosine** — primary SOTA RL sizing baseline
- **ABCMOBO** — non-RL comparison
- **PPAAS** — additional SOTA RL comparison

## Results

### Topology-generation comparison

| Method | FinalPass ↑ | P@5 ↑ | Calls | Runtime (h) ↓ | FoM ↑ | PVT (%) ↑ |
|---|---|---|---|---|---|---|
| AnalogCoder-Pro | 0.057 | 0.057 | 65 | 14.1 | 11.0 | 0.0 |
| CktGen | 0.770 | 0.770 | 65 | 43.5 | 73.4 | 99.3 |
| AnalogToBi | 0.092 | 0.092 | 65 | 25.4 | 1.9 | 15.6 |
| RAPTOR | 0.851 | 0.529 | 65 | 6.3 | 231.3 | 100.0 |

Across the evaluated topology baselines, RAPTOR:

- reduces runtime by 2.2x–6.9x,
- achieves 3.2x–121.7x higher FoM,
- improves FinalPass by 8.1–79.4 percentage points, and
- achieves 100.0% PVT robustness.

Against AnalogCoder-Pro specifically, training-inclusive runtime decreases from 14.1 h to 6.3 h, corresponding to a 2.2x speedup, while RAPTOR achieves 21.0x the FoM.

### Circuit sizing: primary SOTA RL comparison

| Method | FinalPass ↑ | Calls | Runtime (s) ↓ | FoM ↑ | PVT (%) ↑ | All-Corner (%) ↑ |
|---|---|---|---|---|---|---|
| ORACLE-Cosine | 0.638 | 120 | 43.79 | 4.30 | 83.7 | 64.6 |
| RAPTOR | 1.000 | 120 | 13.70 | 13.19 | 98.7 | 96.6 |

Compared with ORACLE-Cosine under the same 120-call budget, RAPTOR:

- increases FinalPass from 0.638 to 1.000, a 56.7% relative improvement,
- increases FoM from 4.30 to 13.19, a 207% improvement,
- reduces runtime from 43.79 s to 13.70 s, a 3.2x speedup,
- increases PVT robustness from 83.7% to 98.7%, and
- increases all-corner robustness from 64.6% to 96.6%, a 49.5% relative improvement.

### Circuit sizing: additional baselines

| Method | Budget | FinalPass ↑ | Runtime (s) ↓ | FoM ↑ | PVT (%) ↑ | All-Corner (%) ↑ |
|---|---|---|---|---|---|---|
| ABCMOBO | 120 | 0.850 | 134.3 | 9.79 | 73.4 | 78.8 |
| PPAAS | 120 | 0.655 | 23.4 | 3.01 | 83.1 | 64.8 |
| RAPTOR | 120 | 1.000 | 13.7 | 13.19 | 98.7 | 96.6 |

Against ABCMOBO, RAPTOR improves FoM from 9.79 to 13.19, PVT robustness from 73.4% to 98.7%, and all-corner robustness from 78.8% to 96.6%, while reducing runtime from 134.3 s to 13.7 s.

### Component Ablation

The ablation study is performed on Heldout29 under nominal-only evaluation.

**(a) Feasibility and search efficiency**

| Variant | Pass ↑ | P@4 ↑ | Sims @ B 4/8/16/32 | Sims/run |
|---|---|---|---|---|
| RAPTOR_FULL | 87/87 (100%) | 99% | 43/52/54/54 | 65 |
| no-agents | 87/87 (100%) | 83% | 72/87/87/87 | 65 |
| no-RAG | 87/87 (100%) | 83% | 24/87/87/87 | 65 |
| no-diversity | 87/87 (100%) | 83% | 24/87/87/87 | 65 |
| no-ranker | 87/87 (100%) | 83% | 24/87/87/87 | 65 |
| no-MB-SAC | 87/87 (100%) | 93% | 81/87/87/87 | 65 |
| no-surrogate | 87/87 (100%) | 93% | 81/87/87/87 | 65 |
| no-LLM | 87/87 (100%) | 53% | 46/70/85/87 | 65 |
| no-selector | 87/87 (100%) | 72% | 63/87/87/87 | 65 |
| no-SFT | 0/87 (0%) | 0% | 0/0/0/0 | 0 |

**(b) Design quality and runtime**

| Variant | FoM Median ↑ | FoM Best ↑ | Runtime (s) ↓ |
|---|---|---|---|
| RAPTOR_FULL | 13,665 | 60,594 | 182 |
| no-agents | 4,921 | 9,815 | 353 |
| no-RAG | 4,921 | 9,815 | 245 |
| no-diversity | 4,921 | 9,815 | 430 |
| no-ranker | 4,921 | 9,815 | 321 |
| no-MB-SAC | 5,182 | 16,863 | 343 |
| no-surrogate | 1,815 | 17,702 | 349 |
| no-LLM | 4,182 | 47,124 | 508 |
| no-selector | 4,869 | 9,815 | 351 |
| no-SFT | -- | -- | 580 |

The full system reaches 100% nominal pass rate, 99% P@4, a median FoM of 13,665, a best FoM of 60,594, and 182 s runtime.

Removing MB-SAC reduces median FoM to 5,182, reduces best FoM to 16,863, and increases runtime to 343 s. Removing the LLM reduces P@4 from 99% to 53%, while removing SFT produces no feasible designs.

## Statistical Analysis

The paper uses:

- the two-sided McNemar test for paired feasibility outcomes,
- the Wilcoxon signed-rank test for paired continuous metrics, and
- significance threshold $\alpha=0.05$.

The statistical table currently reported in the paper is:

| Comparison | Metric | Test | p-value |
|---|---|---|---|
| RAPTOR vs. AnalogCoder-Pro | FoM | Wilcoxon | $1.1\times10^{-10}$ |
| RAPTOR vs. AnalogCoder-Pro | Runtime | Wilcoxon | $1.6\times10^{-5}$ |
| RAPTOR vs. AnalogCoder-Pro | FinalPass | McNemar | $<10^{-15}$ |
| RAPTOR vs. ORACLE-Cosine | FinalPass | McNemar | $1.4\times10^{-188}$ |
| RAPTOR vs. ORACLE-Cosine | FoM | Wilcoxon | $8.7\times10^{-57}$ |
| RAPTOR vs. no-MB-SAC | FoM | Wilcoxon | $3.4\times10^{-11}$ |
| RAPTOR vs. no-MB-SAC | Runtime | Wilcoxon | $1.4\times10^{-11}$ |
| RAPTOR vs. no-SFT | Feasibility | McNemar | $<0.001$ |

**Note:** This table mirrors the current manuscript exactly. If the statistical values are recomputed from the final paired raw outcomes, the manuscript and README should be updated together so they remain synchronized.

## Main Paper-Level Results

The current paper reports the following headline findings:

- **Topology generation:** 2.2x–6.9x runtime reduction, 3.2x–121.7x higher FoM, 8.1–79.4 percentage-point FinalPass improvement, and 100% PVT robustness over the evaluated topology baselines.
- **Circuit sizing:** compared with the primary SOTA RL baseline, 3.2x lower runtime, 207% higher FoM, 56.7% higher FinalPass, and 49.5% higher all-corner robustness.
- RAPTOR reaches 100% FinalPass, 98.7% PVT robustness, and 96.6% all-corner robustness on the circuit-sizing benchmark.

## Implementation

The implementation described in the paper uses:

- Python / PyTorch
- LoRA-adapted Qwen3-4B-Instruct-2507
- 1,500 SFT training steps for topology generation
- contextual-bandit topology selection
- OpenAI Gym environment for circuit sizing
- MB-SAC as the continuous sizing RL engine
- online surrogate guidance
- SPICE-based evaluation
- preference ranking
- PVT analysis
- final verification with one recovery attempt

## Reproducibility

For reproducible comparisons, methods within each benchmark use the same:

- design specifications,
- SPICE-call budget, and
- evaluation criteria,

with training time included where applicable.

The topology-generation and circuit-sizing benchmarks are intentionally reported separately because they use different process models, design spaces, and simulation protocols.
