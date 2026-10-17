# MDP Modeling: Understanding the Push Block Environment

## Introduction

Once we confirm that our problem is a great fit for Reinforcement Learning (RL), the textbook next step is to model it as a Markov Decision Process (MDP) defined by the tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$.

Usually, if you were building an RL problem from scratch, you'd spend days defining your own state space, designing the action interfaces, and tuning reward math. But here, we're using **Push Block**—an established 3D physics environment built on Unity ML-Agents.

So instead of reinventing the wheel, our job here is to **reverse-engineer and deeply understand the environment's MDP properties**. Knowing the exact dimensions, ranges, and reward dynamics is what allows us to pick the right algorithm, design effective neural network architectures, and avoid silly debugging headaches later.

---

## 1. Inspecting the Raw Environment

Let's start by running our automated environment profiler script (`inspect_env.py`) directly against the compiled Unity binary:

```bash
uv run python scripts/inspect_env.py --config configs/envs/push_block.yaml --steps 500
```

Here is the empirical telemetry report we got:

```text
ENVIRONMENT SPECIFICATION:
  - Backend               : unity
  - Parallel Arenas (N)   : 32
  - Observation Space (S) : Box(-inf, inf, (210,), float32)
    * Shape               : (210,)
    * Dtype               : float32
  - Action Space (A)      : Discrete(7)
    * Type                : Discrete(7)

• SAMPLING TRAJECTORIES VIA RANDOM POLICY...
----------------------------------------------------------------------
EMPIRICAL TELEMETRY (16,000 transitions across 32 arenas):
  - Profiling Wall Time   : 2.48 s
  - Sampling Throughput   : 6,450.1 transitions/sec (FPS)
  - Observed State Range  : [0.0000, 1.0000]
    * State Integrity     : NaN=False, Inf=False (Healthy: True)
  - Step Reward Range     : [-0.00100, -0.00100]
    * Unique Reward Values: [-0.001]
  - Goals Reached (Raw)   : 0
  - Terminated Episodes   : 0 (Increase --steps to observe full episode terminations)
```

At a glance, three big takeaways jump right out:
1. **Observation Space ($\mathcal{S}$)** is a 210-dimensional continuous vector (`Box(210,)`) with values neatly resting inside $[0.0, 1.0]$.
2. **Action Space ($\mathcal{A}$)** is a single discrete branch with 7 possible moves (`Discrete(7)`).
3. **Reward Signal ($\mathcal{R}$)** gives a constant step penalty of `-0.001` per step, and random exploration achieved **0 goals** across 16,000 transitions.

Let's break down each of these components in detail.

---

## 2. Observation Space ($\mathcal{S}$)

### What does the `(210,)` vector actually represent?

You might wonder: *this is a 3D visual game, so why isn't the observation an RGB camera feed?*

Rendering raw camera pixels (e.g., $84 \times 84 \times 3$) and passing them through heavy CNNs is computationally expensive and significantly slows down sample collection. Instead, Unity ML-Agents equips the Push Block agent with **Ray Perception Sensors** (lidar-like raycasts), which gives high spatial precision at lightning speed (~6,400+ FPS).

Here is the math behind the 210 dimensions:

1. **Ray Perception Fan:**
   - The agent casts rays out into the environment covering a spherical/horizontal field of view.
   - In Push Block, there are **14 raycast directions**.
2. **Detectable Tags & Distance:**
   - Each ray checks what type of object it strikes among 3 possible detectable tags: `wall`, `goal`, or `block`.
   - Along with one-hot encoded hit indicators and normalized ray hit distances, each ray produces **5 feature dimensions** ($3 \text{ object tags} + 1 \text{ distance} + 1 \text{ hit flag}$ or sub-features), resulting in:
     $$\text{Base Features per Step} = 14 \times 5 = 70 \text{ floats}$$
3. **Observation Stacking (Temporal History):**
   - A single static snapshot doesn't tell the agent whether an object or the agent itself is moving, how fast it's sliding, or in which direction.
   - To make the environment strictly Markovian without forcing us to use recurrent models (like LSTM or GRU), ML-Agents uses **stacked vector observations** with a history size of **3 consecutive frames**:
     $$\text{Total Observation Dimension} = 70 \times 3 = \mathbf{210}$$

### Key Properties & Good News for Training
- **Bounded Range:** The empirical telemetry confirmed an observed range of `[0.0, 1.0]`. All ray distances and tags are already pre-normalized by the Unity engine.
- **Data Integrity:** `NaN=False, Inf=False`. There are no corrupted values or sensor failures.
- **Model Efficiency:** Because input vectors are just 210 floats, we can use a lightweight Multi-Layer Perceptron (MLP) policy (e.g. `[256, 256]` or `[128, 128]`), making both training and deployment super fast!

---

## 3. Action Space ($\mathcal{A}$)

The action space is modeled as `Discrete(7)`, meaning at every decision step the agent chooses exactly one integer index from $\{0, 1, 2, 3, 4, 5, 6\}$.

Here is how each action maps to the physical actuators of the agent:

| Action Index | Physical Command | Description |
| :---: | :--- | :--- |
| **0** | **No-op** | Do nothing (coasting / inertia). |
| **1** | **Move Forward** | Apply forward thrust along the agent's facing direction. |
| **2** | **Move Backward** | Apply reverse thrust. |
| **3** | **Strafe Left** | Shift body laterally to the left. |
| **4** | **Strafe Right** | Shift body laterally to the right. |
| **5** | **Rotate Clockwise** | Turn right around the vertical axis ($Y$-axis). |
| **6** | **Rotate Counter-Clockwise** | Turn left around the vertical axis ($Y$-axis). |

### Practical Implications
- **Physics-driven Interaction:** The agent does not teleport. Moving or rotating translates to applying forces and torques via Unity's PhysX engine. To push the block, the agent must maneuver behind it, face the goal direction, and continuously push forward against the block's mass and friction.
- **Action Selection Strategy:** Since the space is discrete, standard policy gradient methods can output simple categorical distributions. We won't need to deal with continuous action clamping, tanh squashing, or variance vanishing tricks.

---

## 4. Default Reward Function ($\mathcal{R}$) & Episode Horizon

In the default configuration of Push Block (latest version), the reward signal is structured as follows:

$$R_t = \begin{cases} 
+5.0 & \text{if the movable block successfully enters/touches the goal zone (Terminal Success)} \\
-0.001 & \text{step penalty applied on every single decision step}
\end{cases}$$

### Why this design?

1. **Urgency & Efficiency (Step Penalty of `-0.001`):**
   - The environment maximum episode horizon is $T_{\max} = 1,000$ steps.
   - If the agent just wanders around or stands still, it bleeds reward at a rate of $-0.001$ per step, accumulating a cumulative penalty of $-1.0$ if it times out without reaching the goal.
   - This negative pressure pushes the policy to solve the task using the shortest, most direct trajectory possible.
2. **Sparse Terminal Success (`+5.0`):**
   - The primary objective reward is strictly given when the block actually hits the target goal.
   - There is **no intermediate guidance** (no hand-crafted reward for "walking toward the block" or "bumping into the block").

### The Exploration Challenge: Why Random Policy Gets 0 Goals

Look back at our telemetry report:
> `Goals Reached (Raw): 0` (out of 16,000 steps across 32 arenas)

Pushing a block into a goal requires a precise sequential chain of actions:
1. Locate the block using raycasts.
2. Navigate around the block so that the block is positioned directly between the agent and the goal.
3. Push the block straight ahead into the target zone without pushing it into a stuck corner.

The probability of a purely uniform random policy executing this specific sequence by pure chance is astronomically small. 

This tells us right away that **exploration will be the key bottleneck**. When choosing our algorithm and hyperparameters, we will need:
- Strong entropy regularization to prevent the policy from collapsing prematurely into a stationary or wall-hugging local minimum.
- High sample throughput (leveraging all 32 arenas in parallel) so that rare accidental successes can be experienced and reinforced early.
- Value function normalization / baseline estimation to handle the contrast between small step penalties and large terminal bonuses.

---

## Summary of the MDP Specification

| MDP Component | Definition in Push Block |
| :--- | :--- |
| **State / Observation Space ($\mathcal{S}$)** | $\mathbb{R}^{210}$ — 14 Raycast sensors $\times$ 5 attributes $\times$ 3 stacked history frames, normalized in $[0, 1]$. |
| **Action Space ($\mathcal{A}$)** | $\text{Discrete}(7)$ — 4 directional translations, 2 rotations, and 1 no-op. |
| **Reward Function ($\mathcal{R}$)** | $r_t = -0.001$ per step; $r_{\text{terminal}} = +5.0$ upon pushing the block into the goal. |
| **Episode Horizon ($T$)** | Up to 1,000 steps per episode; terminates immediately on goal completion or timeout. |
| **Parallelization** | 32 concurrent arenas inside a single Unity engine process ($\approx 6,000+$ FPS). |

With the MDP mechanics crystal clear, we are ready to move on to **[02_algorithm_selection.md](02_algorithm_selection.md)** where we'll compare candidate algorithms (PPO, DQN, SAC, etc.) and choose the best one for this exact setup!
