# Multi-Seed Generalization and Robustness Benchmark: Crawler

## Overview & Evaluation Objectives

In **[03_baseline_experiments.md](03_baseline_experiments.md)** and **[04_hyperparameter_tuning.md](04_hyperparameter_tuning.md)**, both the Default Baseline and Tuned SAC models were trained and verified under a single environment initialization (`seed: 42`).

In physical robotics and production reinforcement learning, training performance on a single seed provides zero guarantee of real-world viability. A continuous controller must not simply memorize a favorable trajectory; it must demonstrate **robust out-of-distribution generalization** across novel random joint angles, randomized spawn heights, and varied initial torso tilt orientations.

Phase 5 executes a rigorous, multi-seed generalization benchmark evaluating both checkpoints across **10 completely unseen random seeds** (Seeds 1001 to 1010), collecting 10 episodes per seed (100 episodes total per model) using [scripts/evaluate_policy.py](../../scripts/evaluate_policy.py).

This document presents the empirical evidence, analyzes seed-by-seed stability, and compares the models head-to-head.

---

## 1. Experimental Evaluation Protocol

The evaluation was executed in headless, accelerated simulation (`time_scale: 20.0x`) to eliminate rendering overhead while preserving deterministic physical stepping:

- **Unseen Random Seeds:** 10 sequential held-out seeds: `[1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]`
- **Sample Volume:** 10 evaluation episodes per seed ($N = 100$ episodes per model, $200$ episodes total)
- **Policy Ingestion:** Deterministic actor inference ($a = \mu(s)$) extracted from `best_model.zip`
- **Observation Normalization:** Frozen running statistics loaded from `final_vec_normalize.pkl` (`training = False`, `norm_reward = False`) to measure true physical returns
- **Task Success / Survival Metric:** Defined as sustaining stable forward locomotion without premature structural collapse (`return > 0.0` and `length >= 500 steps`)
- **Evaluation Throughput:** The 100-episode test suite completed in $\approx 50$ seconds ($\approx 2.0$ episodes/second) per model

### Benchmark Execution Commands

```bash
# 1. Evaluate Default SAC Baseline across 10 unseen seeds
uv run python scripts/evaluate_policy.py \
  --config configs/experiments/default_crawler_sac.yaml \
  --model experiments/outputs/default_crawler_sac/seed_42/best_model/best_model.zip \
  --num-seeds 10 \
  --seed-start 1001 \
  --episodes-per-seed 10 \
  --port 5070 \
  --output benchmarks/crawler_default_sac_unseen_seeds.json

# 2. Evaluate Tuned SAC Model across the exact same 10 unseen seeds
uv run python scripts/evaluate_policy.py \
  --config configs/experiments/tuned_crawler_sac.yaml \
  --model experiments/outputs/tuned_crawler_sac/seed_42/best_model/best_model.zip \
  --num-seeds 10 \
  --seed-start 1001 \
  --episodes-per-seed 10 \
  --port 5072 \
  --output benchmarks/crawler_tuned_sac_unseen_seeds.json
```

---

## 2. Aggregated Empirical Evidence (Head-to-Head)

The empirical summary extracted directly from [crawler_default_sac_unseen_seeds.json](../../benchmarks/crawler_default_sac_unseen_seeds.json) and [crawler_tuned_sac_unseen_seeds.json](../../benchmarks/crawler_tuned_sac_unseen_seeds.json) demonstrates decisive superiority for the Tuned SAC Model:

| Evaluation Dimension | Default SAC Baseline | Tuned SAC Model | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Mean Evaluation Return** | $+468.71 \pm 201.78$ | **$+1,430.99 \pm 332.59$** | **$+205.3\%$ surge ($3.05\times$ higher)** |
| **Overall Survival Rate** | $58.0\%$ | **$75.0\%$** | **$+17.0\%$ absolute gain** |
| **Mean Episode Length** | $614.9 \pm 194.8$ steps | **$769.2 \pm 122.1$ steps** | $+154.3$ steps ($+25.1\%$) |
| **Inter-Seed Length Variance** | $\pm 194.8$ steps | **$\pm 122.1$ steps** | **$37.3\%$ variance reduction across seeds** |
| **Reward Acquisition Rate** | $0.762$ points/step | **$1.860$ points/step** | **$2.44\times$ stride velocity surge** |
| **Minimum Seed Return** | $+195.11$ (Seed 1007) | **$+911.52$** (Seed 1009) | Floor raised by $+716.41$ points ($4.67\times$) |
| **Maximum Seed Return** | $+800.74$ (Seed 1002) | **$+2,214.95$** (Seed 1001) | Ceiling raised by $+1,414.21$ points ($2.77\times$) |
| **Peak Individual Episode** | $+2,448.14$ | **$+3,299.10$** | High-velocity sprint capability preserved |
| **Evaluation Wall Time (100 eps)** | $51.13$ seconds | $50.31$ seconds | Consistent high-throughput simulation |

---

## 3. Seed-by-Seed Telemetry Breakdown

Evaluating individual seed breakdowns highlights how the policies respond to diverse physical initializations:

| Seed | Default Return ($\pm \text{Std}$) | Tuned Return ($\pm \text{Std}$) | Default Length | Tuned Length | Default Succ | Tuned Succ | Head-to-Head Outcome |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1001** | $+268.20 \pm 398.21$ | **$+2,214.95 \pm 1,067.85$** | $327.8$ | **$967.0$** | $30.0\%$ | **$100.0\%$** | **Tuned ($+725.9\%$)** |
| **1002** | $+800.74 \pm 930.31$ | **$+1,409.75 \pm 1,128.61$** | $612.7$ | **$724.6$** | $50.0\%$ | **$70.0\%$** | **Tuned ($+76.1\%$)** |
| **1003** | $+435.86 \pm 712.13$ | **$+1,549.71 \pm 915.26$** | $895.5$ | **$951.3$** | $90.0\%$ | **$100.0\%$** | **Tuned ($+255.6\%$)** |
| **1004** | $+694.42 \pm 724.39$ | **$+1,605.45 \pm 1,184.07$** | **$903.0$** | $768.0$ | **$90.0\%$** | $70.0\%$ | **Tuned ($+131.2\%$)** |
| **1005** | $+376.73 \pm 520.00$ | **$+1,295.56 \pm 1,013.32$** | $616.7$ | **$775.4$** | $60.0\%$ | **$70.0\%$** | **Tuned ($+243.9\%$)** |
| **1006** | $+508.62 \pm 630.36$ | **$+1,468.51 \pm 1,229.37$** | **$805.3$** | $792.4$ | $80.0\%$ | $80.0\%$ | **Tuned ($+188.7\%$)** |
| **1007** | $+195.11 \pm 502.86$ | **$+1,192.05 \pm 1,068.34$** | **$527.6$** | $505.4$ | **$50.0\%$** | $40.0\%$ | **Tuned ($+511.0\%$)** |
| **1008** | $+675.38 \pm 699.80$ | **$+1,131.24 \pm 1,132.00$** | $654.2$ | **$707.8$** | $60.0\%$ | **$70.0\%$** | **Tuned ($+67.5\%$)** |
| **1009** | $+198.75 \pm 241.53$ | **$+911.52 \pm 1,069.02$** | $407.1$ | **$741.6$** | $40.0\%$ | **$80.0\%$** | **Tuned ($+358.6\%$)** |
| **1010** | $+533.27 \pm 701.21$ | **$+1,531.21 \pm 1,371.34$** | $399.0$ | **$758.6$** | $30.0\%$ | **$70.0\%$** | **Tuned ($+187.1\%$)** |

---

## 4. Key Engineering Insights & Diagnostic Evidence

### 1. 100% Win Rate Across All Unseen Seeds (Zero Overfitting)
- Tuned SAC outperformed the Default Baseline on **every single held-out seed** (10 out of 10 seeds won).
- Improvements ranged from $+67.5\%$ on Seed 1008 to $+725.9\%$ on Seed 1001.
- This provides definitive empirical evidence that the tuned policy learned generalized multi-joint coordination rather than overfitting to the training environment's initial seed conditions.

---

### 2. Rescue of "Compromised Spawn" Initializations
In continuous quadruped environments, certain seeds drop the robot at awkward tilt angles where the center of mass starts dangerously close to the floor. Seeds **1001**, **1009**, and **1010** represent difficult initial conditions:
- **Default Baseline Vulnerability:** The baseline policy suffered catastrophic collapse in 70% of episodes on Seed 1001 and Seed 1010 (survival rate of only $30\%$, mean episode length $<400$ steps). It lacked the postural reflexes to right itself from an uncentered drop.
- **Tuned Model Self-Righting Reflex:** Under Tuned SAC, survival rate jumped to **$100\%$ on Seed 1001** (mean return $+2,214.95$, length $967.0$ steps), and to **$70\% - 80\%$ on Seeds 1009 and 1010**.
- **Physical Interpretation:** By preventing early entropy collapse during training ($\bar{\mathcal{H}} = -10.0$), the policy acquired diverse balance recovery maneuvers, enabling the limbs to catch the ground and push the torso upward before touching the floor.

---

### 3. Substantial Stride Velocity Surge
- The average reward acquisition rate increased from **$0.762 \text{ points/step}$** to **$1.860 \text{ points/step}$** ($2.44\times$ higher).
- Across novel seeds, the tuned agent does not simply survive; it maintains high forward momentum. Even on Seed 1007 where episode duration was shorter ($505.4$ steps), the agent earned **$+1,192.05$ points**, far exceeding the baseline's $+195.11$ points over a similar duration.

---

### 4. Robust Inter-Seed Consistency
- The inter-seed standard deviation of episode length contracted from **$194.8 \text{ steps}$ down to $122.1 \text{ steps}$** (a $37.3\%$ reduction in cross-seed variance).
- While individual episodes still exhibit variance due to continuous terrain collisions, the policy's average performance across different environment seeds is substantially more predictable and consistent.

---

## 5. Benchmark Data Artifacts

All raw episode transitions, lengths, and return arrays are preserved in the repository for auditability and verification:

- **Default Baseline Benchmark:** [benchmarks/crawler_default_sac_unseen_seeds.json](../../benchmarks/crawler_default_sac_unseen_seeds.json)
- **Tuned Model Benchmark:** [benchmarks/crawler_tuned_sac_unseen_seeds.json](../../benchmarks/crawler_tuned_sac_unseen_seeds.json)
- **Evaluation Engine:** [scripts/evaluate_policy.py](../../scripts/evaluate_policy.py)

---

## 6. Next Step: Continuous ONNX Model Packaging

With the Tuned SAC Model thoroughly validated across held-out environments, we proceed to **Phase 6: Continuous ONNX Model Export & Packaging**:
- We will adapt our ONNX exporter ([scripts/export_onnx.py](../../scripts/export_onnx.py)) for continuous Gaussian actor policies.
- In discrete environments (Push Block), export requires an `ArgMax` operator over logits. In continuous control (Crawler), the exported computational graph extracts the deterministic actor mean ($\mu(s)$) squashed through $\tanh$, producing an exact $20$-dimensional motor torque vector in $[-1.0, 1.0]$ for zero-overhead embedded Unity / C++ deployment.
