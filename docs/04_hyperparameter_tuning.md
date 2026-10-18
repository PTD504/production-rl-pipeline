# Hyperparameter Tuning: Playbook and Strategy

## Overview

In our previous steps ([03_baseline_experiments.md](03_baseline_experiments.md) and our reward analysis in [03b_reward_shaping.md](03b_reward_shaping.md)), we evaluated our baseline PPO model across 3 independent random seeds. The results were remarkably solid: the agent learned to push the block into the goal consistently, reaching a mean reward of **4.72** with a 100% task success rate.

Normally, in a standard Reinforcement Learning (RL) project, this is the exact moment where you would jump into **Hyperparameter Tuning** (also called Hyperparameter Optimization or HPO). 

Even though our baseline already solved the Push Block task well enough that we don't need to run a heavy tuning sweep here, we still want to document this step clearly. Think of this guide as our practical **Tuning Playbook**: what hyperparameter tuning actually involves in RL, which knobs move the needle the most, how to organize a sweep properly, and why we decided to freeze our baseline model as-is.

---

## 1. Why Tuning RL is Different from Supervised Learning

If you come from standard deep learning (like computer vision or NLP), hyperparameter tuning usually means searching for parameters that minimize loss on a fixed validation dataset.

In RL, tuning is quite different and noticeably trickier:

1. **Data depends on the policy:** When you change a hyperparameter (like the learning rate or entropy bonus), you change how the agent acts. That changes the transitions it collects, which in turn changes the training data itself.
2. **Seed sensitivity:** RL algorithms have high variance. An agent might perform great on seed 42 simply because it got lucky during random exploration, not because your learning rate was superior. Any serious tuning effort must account for multi-seed averaging.
3. **Fragile balance:** Tweak a parameter too far, and the policy might collapse completely into a degenerate behavior (e.g., spinning in circles or freezing to avoid negative step penalties).

---

## 2. The PPO Tuning Playbook: Key Knobs to Care About

If you are using PPO and your baseline is underperforming, here is the prioritized list of hyperparameters worth tuning, ordered by their typical real-world impact:

| Hyperparameter | Typical Range | Impact | What it Controls & Practical Advice |
| :--- | :--- | :---: | :--- |
| **Learning Rate (`learning_rate`)** | `1e-5` to `1e-3` | **Highest** | Start around `3e-4`. If training is unstable or policy updates fluctuate wildly, try reducing to `1e-4` or using a linear decay schedule. |
| **Rollout Length (`n_steps`)** | `64` to `2048` | **High** | How many steps each parallel environment collects before running an update. Shorter lengths mean faster update cycles, while longer rollouts provide a more diverse batch of experiences. |
| **Batch Size (`batch_size`)** | `32` to `512` | **High** | Mini-batch size for SGD updates. Must divide `n_steps * n_envs`. Larger batches stabilize gradients; smaller batches can escape local minima faster. |
| **Entropy Coefficient (`ent_coef`)** | `0.0` to `0.05` | **High** | Pushes the agent to keep exploring random actions. If your agent converges prematurely to a suboptimal strategy, bump this up (e.g., from `0.0` to `0.01` or `0.02`). |
| **Clip Range (`clip_range`)** | `0.1` to `0.3` | **Medium** | The PPO surrogate clipping threshold. `0.2` is a great default. Lower values (`0.1`) make updates safer and more conservative; higher values (`0.3`) allow bolder updates. |
| **Discount Factor (`gamma`)** | `0.95` to `0.999` | **Medium** | How much the agent cares about distant rewards versus immediate ones. For long-horizon tasks, keep it high (`0.99` - `0.995`). |
| **GAE Parameter (`gae_lambda`)** | `0.90` to `0.98` | **Medium** | Balances bias vs. variance in Generalized Advantage Estimation. `0.95` works well across most benchmarks. |
| **Network Architecture** | `[64, 64]` to `[256, 256]` | **Medium** | For state-based inputs (like our 210-dim vector), small MLPs (2 layers with 64 or 128 hidden units) are almost always fast and sufficient. |

---

## 3. How We Would Run HPO If Needed

When you do need to run an automated sweep, here is the standard workflow we recommend:

```
[Define Search Space] ──► [Sampler (Optuna TPE)] ──► [Train Trial (2-3 seeds)] ──► [Pruning (ASHA / Median)] ──► [Pick Best Config]
```

1. **Framework:** Use **Optuna** or **Ray Tune**. They integrate smoothly with PyTorch and Stable-Baselines3.
2. **Search Strategy:** Use Bayesian Optimization (such as Tree-structured Parzen Estimator, or **TPE**). It is far more sample-efficient than pure random search or grid search because it learns from past trials to suggest better parameter combinations.
3. **Early Pruning (Saving Compute):** Use a pruner like **MedianPruner** or **Hyperband / ASHA**. If a trial is clearly failing to gain rewards after the first 20% of its timesteps, terminate it early so GPU resources go toward promising runs.
4. **Multi-Seed Objective:** Evaluate every candidate configuration on at least 2 or 3 distinct random seeds. Optimize for the **mean evaluation reward** while penalizing high variance across seeds to avoid picking brittle configurations.

---

## 4. Why We Are Skipping the Tuning Sweep for Push Block

In engineering, knowing **when to stop** is just as important as knowing how to tune. Based on our Phase 03 baseline results, we made the deliberate decision to **freeze the baseline model and skip the tuning sweep**.

Here is why:

### 1. We Already Hit the Practical Theoretical Ceiling
The Push Block environment awards `+5.0` points for landing the block in the goal zone, minus a `-0.001` penalty for every elapsed step. Since solving the physics task takes at least 30 to 80 steps, the realistic physical maximum reward is roughly:

$$\text{Max Reward} \approx 5.0 - (50 \times 0.001) = 4.95$$

Our baseline achieved an average evaluation reward of **4.722** (with Seed 3 hitting **4.971**). The agent already solves the task directly and quickly—there is virtually no performance gap left to close.

### 2. High Stability and 100% Success Rate
All 3 baseline seeds converged reliably without policy collapse, divergence, or getting stuck in bad habits. The variance between seeds was minimal, proving that the default settings are already in a sweet spot.

### 3. Compute Efficiency
Running a thorough Optuna sweep with 30 to 50 trials across multiple seeds would burn hours of CPU and GPU time. Spending compute to chase a 0.1 reward improvement on an already solved task provides near-zero return on investment.

---

## 5. Next Step

Now that our policy configuration is validated and frozen, we are ready to move on to the next phase:

**[Phase 05: Deep Evaluation and Robustness Testing]**

Before we package and export this model for production deployment, we need to stress-test it: evaluate it across a wider batch of unseen seeds, inspect edge cases (such as awkward block angles or obstacles), and verify that our policy is genuinely robust rather than just lucky.
