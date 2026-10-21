# Algorithm Selection: Choosing the Engine for 20-DOF Continuous Locomotion

## Introduction

In **[01_mdp_modeling.md](01_mdp_modeling.md)**, our automated MDP profiler established the core characteristics of the Crawler environment:
- **Action Space ($\mathcal{A}$):** Continuous 20-dimensional motor torque vector (`Box(-1.0, 1.0, (20,), float32)`).
- **Observation Space ($\mathcal{S}$):** 158 continuous body sensors (`Box(-17.5, 39.6, (158,), float32)`).
- **Reward Signal ($\mathcal{R}$):** Dense continuous locomotion reward (forward velocity, upright posture, and $-1.0$ fall penalty).
- **Simulation Concurrency:** 10 parallel arenas in a single Unity process, delivering $\approx 1,200$ transitions/sec.

In Case Study 1 (Push Block), our discrete action space led us directly to PPO over DQN. 

Now, with **20 continuous motors to coordinate simultaneously**, we evaluate candidate RL algorithms from first principles to select the best engine for high-dimensional quadruped locomotion.

---

## Step 1: Immediate Elimination of the DQN Family

The first filter is immediate: **DQN and its variants (Double DQN, Dueling DQN, Rainbow) are fundamentally incompatible with continuous action spaces.**

In Q-learning, the Bellman optimality update requires computing:

$$y_t = r_t + \gamma \max_{a' \in \mathcal{A}} Q(s_{t+1}, a')$$

- In discrete environments (like Push Block), finding $\max_{a'} Q(s, a')$ is trivial: evaluate the Q-network across 7 scalar outputs and pick the index with the highest value.
- In Crawler, $\mathcal{A} \in [-1.0, 1.0]^{20}$. Finding the global maximum of a deep non-linear Q-function over a continuous 20-dimensional volume requires solving an intractable, non-convex continuous optimization problem at every single environment transition.

**Verdict:** All discrete Q-learning algorithms are eliminated immediately.

---

## Step 2: On-Policy vs. Off-Policy in Continuous Control

With discrete methods off the table, we look at the two major paradigms for continuous control:

### 1. On-Policy Methods (A2C, PPO)
- **A2C (Advantage Actor-Critic):** Evaluates a 1-step gradient update and discards data immediately. Highly sample-inefficient and prone to catastrophic policy collapse on high-dimensional continuous physics. **Eliminated.**
- **PPO (Proximal Policy Optimization):** Can parameterize a Gaussian policy $\pi(a|s) \sim \mathcal{N}(\mu(s), \sigma^2)$ and uses clipped surrogate objectives to run multi-epoch mini-batches safely. While PPO remains a strong, stable baseline when simulator throughput is very high, on-policy algorithms cannot reuse past transitions once the policy updates.

### 2. Off-Policy Methods (DDPG, TD3, SAC)
- In continuous physical control (robotics, multi-joint locomotion), **sample efficiency** is paramount. Off-policy algorithms store transitions in a persistent Experience Replay Buffer ($\mathcal{D}$), reusing past experiences hundreds of times to update both Actor and Critic networks.

This leaves our primary continuous off-policy contenders: **DDPG**, **TD3**, and **SAC**.

---

## Step 3: Eliminating DDPG (Deep Deterministic Policy Gradient)

DDPG was the foundational pioneer of continuous actor-critic methods, but in complex 20-dimensional locomotion, it suffers from two fatal flaws:

1. **Catastrophic Overestimation Bias:** DDPG uses a single Critic network $Q_\phi(s, a)$. Maximizing over a single critic systematically overestimates state-action values. Over thousands of updates, this overestimation compounds through the Bellman backup until the Q-values diverge to unrealistic positive numbers, corrupting the policy.
2. **Hyperparameter Fragility:** With 20 coupled motor joints, DDPG is notoriously brittle to learning rates, target update rates ($\tau$), and action noise parameters. A single suboptimal update frequently collapses the walking gait into uncontrollable spinning or freezing.

**Verdict:** DDPG is eliminated in favor of modern second-generation off-policy architectures.

---

## Step 4: The Final Showdown — TD3 vs. SAC

This leaves the two premier state-of-the-art continuous control algorithms: **TD3 (Twin Delayed DDPG)** and **SAC (Soft Actor-Critic)**.

Both algorithms resolve DDPG's overestimation bias by using **Clipped Double-Q Learning** (two independent Critic networks $Q_{\phi_1}, Q_{\phi_2}$, taking $y_t = r_t + \gamma \min_{i=1,2} Q_{\phi_i}(s', a')$).

However, their exploration mechanics and policy formulations differ profoundly:

```
┌──────────────────────────────────────────────────────────┐
│                   CONTINUOUS SHOWDOWN                    │
├────────────────────────────┬─────────────────────────────┤
│   TD3 (Deterministic)      │     SAC (Maximum Entropy)   │
├────────────────────────────┼─────────────────────────────┤
│ a = μ(s) + ε (Manual Noise)│ a ~ π(·|s) (Stochastic)     │
│ Explores via fixed noise   │ Explores via Policy Entropy │
│ Prone to clumsy gaits      │ Discovers coordinated gaits │
│ Fixed noise schedule       │ Automatic α Temperature     │
└────────────────────────────┴─────────────────────────────┘
```

### 1. The Quadruped Locomotion Challenge: Gaits, Not Just Balance

It is important to understand the nature of Crawler:
- Unlike bipedal agents (such as ML-Agents' native **Walker** environment) that constantly threaten to tip over on two feet, Crawler is a **4-legged quadruped**. Because it has a broad polygon of support (4 contact points on the ground), keeping basic static balance without falling is relatively straightforward.
- The real challenge in Crawler is **locomotion coordination**: orchestrating 20 motorized joints across 4 limbs into a fluid, efficient stride cycle that maximizes forward speed.

### 2. Why TD3 Struggles with 20-DOF Coordination
- **Deterministic Policy + Blind Manual Noise:** TD3 learns a deterministic policy $a = \mu_\theta(s)$ and injects exploration purely through external additive noise (e.g., Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$).
- In a 20-dimensional action space, adding uniform random noise across all 20 joints produces crude, jerky limb twitches. It does not help discover coordinated multi-joint movements (e.g., "swing hip while flexing knee and planting ankle").
- **Sub-Optimal Gait Trapping:** Because static stability is easy on 4 legs, TD3 quickly discovers safe, "lazy" gaits—such as dragging its rear legs while clumsily paddling with its front legs. Since this avoids the $-1.0$ fall penalty, TD3's deterministic gradients easily get trapped in this sluggish local minimum.

### 3. Why SAC Excels: The Maximum Entropy RL Framework
SAC optimizes a dual objective: **maximize expected return while simultaneously maximizing policy entropy**:

$$J(\pi) = \sum_{t} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}\left(\pi(\cdot | s_t)\right) \right]$$

Where $\mathcal{H}(\pi(\cdot | s_t)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a | s_t)]$ is the Shannon entropy, and $\alpha$ is the temperature parameter.

This design gives SAC decisive advantages on Crawler:

1. **Exploration without Blind Noise:** SAC's policy is inherently stochastic ($\pi_\theta(a|s)$ outputs both a mean $\mu_\theta(s)$ and a variance $\sigma_\theta(s)$, squashed through $\tanh$). The policy explores by assigning non-zero probability to diverse limb positions rather than relying on an external noise generator.
2. **Multimodal Gait Discovery:** By actively penalizing low entropy, SAC is discouraged from prematurely settling on a lazy, single-mode gait (like crawling on 2 knees). It naturally explores multiple gait patterns, discovering symmetrical, high-velocity running strides.
3. **Automatic Temperature Tuning ($\alpha$):** Rather than forcing engineers to manually tune exploration decay schedules, SAC automatically adjusts $\alpha$ via gradient descent to match a target entropy budget:
   $$\bar{\mathcal{H}} = -\text{dim}(\mathcal{A}) = -20$$
   When the agent encounters unfamiliar body orientations, $\alpha$ increases to stimulate exploration. Once a fast, stable gait is established, $\alpha$ automatically contracts to exploit the optimal running motion.

---

## 5. Summary Decision Matrix

| Algorithm | Paradigm | Continuous Support | Overestimation Mitigation | Exploration Strategy | Verdict for Crawler (20-DOF) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **DQN / Rainbow** | Off-policy | None (Discrete only) | Double Q | $\epsilon$-greedy | **Eliminated:** Incompatible with continuous $\mathbb{R}^{20}$. |
| **A2C** | On-policy | Gaussian Head | None | Entropy bonus | **Eliminated:** High variance, poor sample efficiency. |
| **DDPG** | Off-policy | Deterministic Actor | None (Single Critic) | Additive OU / Gaussian noise | **Eliminated:** Divergent Q-values, high hyperparameter fragility. |
| **PPO** | On-policy | Gaussian Head | Baseline Subtraction | Entropy bonus + Clipping | **Strong Baseline:** High stability, but lower sample reuse than off-policy. |
| **TD3** | Off-policy | Deterministic Actor | Clipped Double-Q | Fixed additive Gaussian noise | **Viable, but Limited:** Prone to sub-optimal local gaits in 20-DOF. |
| **SAC** | Off-policy | **Stochastic Squashed Gaussian** | **Clipped Double-Q** | **Maximum Entropy ($\alpha$ Auto-tuned)** | **WINNER:** Gold standard for multi-limb continuous robotics. |

---

## Conclusion & Architecture Choice

For high-dimensional 20-DOF quadruped locomotion in Crawler:
- **Primary Algorithm:** **Soft Actor-Critic (SAC)** is selected as our core off-policy continuous control engine, leveraging maximum entropy exploration and dual Q-critics to learn high-speed coordinated gaits without manual noise tuning.
- **Reference / Alternative Engine:** **PPO (Continuous Gaussian Policy)** remains available as our battle-tested on-policy baseline, allowing direct empirical comparison between on-policy and off-policy scaling in Phase 3.

We proceed to **[03_baseline_experiments.md](03_baseline_experiments.md)** to configure hyperparameters, establish training pipelines, and evaluate empirical performance.
