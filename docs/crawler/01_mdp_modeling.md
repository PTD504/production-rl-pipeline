# MDP Modeling: Understanding the Crawler Environment

## Introduction

In Case Study 1, we built and validated our production RL pipeline on **Push Block**, where the agent acted like a little forklift choosing from 7 discrete buttons (`Discrete(7)`) to push a box into a goal zone.

Now, to test whether our pipeline is truly universal and scalable, Case Study 2 moves to a fundamentally different challenge: **Crawler** from Unity ML-Agents.

Instead of navigating a room, Crawler is a **dynamic physical creature** (think of a mechanical spider or a 4-legged table with bendable knees). Its goal is simple to understand but notoriously hard to solve: **learn how to walk and run forward without collapsing**.

Before jumping into training, we follow Phase 0 of our pipeline: **inspecting and understanding the environment's Markov Decision Process (MDP)** tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ using our automated profiler script.

---

## 1. Automated Environment Profiling

Let's run `inspect_env.py` directly against the Crawler scene inside our compiled Unity binary:

```bash
uv run python scripts/inspect_env.py --config configs/envs/crawler.yaml --steps 500
```

### Empirical Telemetry Report

```text
======================================================================
MDP AUDIT REPORT: Crawler
======================================================================
• ENVIRONMENT SPECIFICATION:
  - Backend               : unity
  - Parallel Arenas (N)   : 10
  - Observation Space (S) : Box(-inf, inf, (158,), float32)
    * Shape               : (158,)
    * Dtype               : float32
  - Action Space (A)      : Box(-1.0, 1.0, (20,), float32)
    * Type                : Continuous(shape=(20,), low=[-1.0 ... -1.0], high=[1.0 ... 1.0])

• SAMPLING TRAJECTORIES VIA RANDOM POLICY...
----------------------------------------------------------------------
EMPIRICAL TELEMETRY (5,000 transitions across 10 arenas):
  - Profiling Wall Time   : 4.21 s
  - Sampling Throughput   : 1,188.5 transitions/sec (FPS)
  - Observed State Range  : [-17.4963, 39.6460]
    * State Integrity     : NaN=False, Inf=False (Healthy: True)
  - Step Reward Range     : [-1.00000, 1.37530]
  - Terminated Episodes   : 253
  - Episode Return (Mean) : -0.809 ± 0.929 (Range: [-1.000, 10.042])
  - Episode Length (Mean) : 19.1 ± 12.4 steps (Range: [7, 83] steps)
======================================================================
```

### Three Quick Takeaways from the Data

1. **Continuous Actions ($\mathcal{A}$):** Instead of picking 1 discrete button out of 7, the policy must control **20 motor pedals simultaneously**, sending a float between `-1.0` and `1.0` to each joint.
2. **Dense Sensor Feed ($\mathcal{S}$):** The agent receives a 158-dimensional vector containing body balance, joint angles, speeds, and foot-ground contact flags.
3. **Random Flailing Fails Instantly:** Under a random policy, the creature collapses in an average of **just 19.1 steps** (less than half a second of simulation). Gravity pulls its belly into the floor, triggering early termination with a `-1.0` penalty.

---

## 2. Observation Space ($\mathcal{S}$)

The observation space is a 158-dimensional vector: `Box(-inf, inf, (158,), float32)`.

### What do the 158 numbers actually tell the agent?

Like Push Block, Crawler does not use an RGB camera. Processing camera frames is unnecessarily slow. Instead, the environment gives the agent direct digital "gauges" of its own body:

1. **Body Orientation & Balance (~14 numbers):**
   - Is the main body tilting forward, backward, or sideways?
   - How fast is the body sliding ($v_x, v_y, v_z$) and spinning ($\omega_x, \omega_y, \omega_z$)?
2. **Target Direction (~4 numbers):**
   - A unit vector pointing in the direction the creature is supposed to move.
3. **Leg Joint Angles & Speeds (~132 numbers):**
   - The creature has 4 legs. Each leg has multiple hinges (hips, knees, ankles).
   - For every single joint, the agent knows its current angle and how fast it is rotating.
4. **Foot Contact Sensors (~8 numbers):**
   - Simple binary/contact flags: Is the front-left foot touching the ground? Is the back-right foot in the air?

### Why Observation Normalization (`VecNormalize`) is Crucial Here

Notice the raw numbers from our telemetry:
$$\min(s) = -17.5, \quad \max(s) = 39.6$$

This is a classic pitfall in continuous control:
- A foot contact flag is just `0.0` or `1.0`.
- A joint angle might be small radians like `0.35`.
- A spinning joint velocity can spike up to `39.6`.

If you feed these numbers raw into a neural network, the big numbers (`39.6`) will dominate the gradients, while the critical small numbers (like whether a foot is touching the ground) will be completely ignored.

By wrapping the environment in **`VecNormalize`**, we dynamically scale every single feature to zero-mean and unit-variance:

$$z = \text{clamp}\left(\frac{x - \mu}{\sigma}, -10.0, 10.0\right)$$

This ensures all 158 body gauges have equal influence on the policy's decisions.

---

## 3. Action Space ($\mathcal{A}$)

The action space is `Box(-1.0, 1.0, (20,), float32)`.

At every decision step, the policy outputs 20 numbers between `-1.0` and `+1.0`:

$$a_t = [a_1, a_2, \dots, a_{20}]^\top \in [-1.0, 1.0]^{20}$$

### What does each action do?

Think of the 20 numbers as **20 motorized pedals** distributed across the 4 limbs:

| Limb | Motor Count | Movements Controlled |
| :--- | :---: | :--- |
| **Front Left Leg** | 5 | Hip swing (forward/backward), hip twist, knee bend, ankle flex. |
| **Front Right Leg** | 5 | Symmetrical controls for the front right leg. |
| **Back Left Leg** | 5 | Hip swing, knee bend, ankle push for propulsion. |
| **Back Right Leg** | 5 | Symmetrical controls for the back right leg. |
| **Total Actuators** | **20** | **Full 4-legged coordinated movement.** |

- Sending `+1.0` means applying maximum forward/positive torque to that motor.
- Sending `-1.0` means applying maximum reverse torque.
- Sending `0.0` leaves the joint relaxed (no active motor torque).

### The Mathematical Shift: Categorical vs. Gaussian Policy

In Push Block, our Actor network used a **Softmax layer** to choose one discrete action:
$$\pi(a|s) = \text{Softmax}(\text{logits}) \implies \text{Select action } 0, 1, \dots, 6$$

In Crawler, our Actor network must predict a **continuous mean vector $\mu \in \mathbb{R}^{20}$**:
$$\pi(a|s) \sim \mathcal{N}\left(\mu(s), \sigma^2\right)$$

During training, actions are sampled around the mean $\mu(s)$ with noise $\sigma$ for exploration. During production deployment, we simply take the deterministic mean action $a = \mu(s)$ without any noise.

---

## 4. Reward Dynamics ($\mathcal{R}$) & The Challenge

In Push Block, reward was **sparse**: the agent wandered around with a tiny step penalty of `-0.001` until it eventually shoved the block into the goal (`+5.0`).

In Crawler, reward is **dense and continuous**:

$$r_t = r_{\text{forward\_speed}} + r_{\text{upright\_posture}} - r_{\text{fall\_penalty}}$$

### What gives points?
1. **Forward Velocity ($+v_{\text{forward}}$):** The faster and more steadily the creature moves in the target direction, the higher the reward (up to $\approx +1.37$ per step).
2. **Upright Posture:** Points for keeping the torso off the ground and oriented properly.
3. **Fall Penalty ($-1.0$ and Episode Reset):** If the creature loses balance and its belly/torso hits the floor, the episode terminates immediately.

### The Learning Bottleneck: Coordination, Not Exploration

In Push Block, the main bottleneck was **exploration** (the agent didn't know where to push the box).

In Crawler, the bottleneck is **motor coordination (gait learning)**:
- Random flailing produces quick collapses (19 steps).
- The agent must discover a stable periodic rhythm (gait): lifting leg A while planting leg B, pushing off the ground, and balancing its center of mass.
- Once it learns not to fall, the forward speed reward naturally pulls the policy toward running.

---

## 5. Summary: Push Block vs. Crawler

Here is a side-by-side comparison of the two case studies in our repository:

| Aspect | Case Study 1: Push Block | Case Study 2: Crawler | Key Takeaway |
| :--- | :--- | :--- | :--- |
| **Problem Type** | Navigation & Object Manipulation | 4-Legged Locomotion (Robotics) | Tests discrete vs. continuous control. |
| **Observation ($\mathcal{S}$)** | 210 numbers (14 Raycast sensors $\times$ 3 frames) | 158 numbers (body tilt, velocities, joint angles) | Lidar raycasts vs. Internal body sensors. |
| **Action ($\mathcal{A}$)** | `Discrete(7)` | `Continuous Box(20)` in $[-1.0, 1.0]$ | 7 buttons vs. 20 motor pedals. |
| **Policy Type** | Categorical (Softmax over 7 logits) | Gaussian (Mean $\mu \in \mathbb{R}^{20}$ + learned variance) | Discrete ArgMax vs. Continuous motor signals. |
| **Reward Signal** | Sparse ($-0.001$ per step, $+5.0$ at goal) | Dense (Progress speed, posture bonus, $-1.0$ on fall) | Finding a goal vs. Learning a walking gait. |
| **Random Policy** | Survived all 1,000 steps (achieved 0 goals) | Collapsed in 19 steps (fell on belly) | Need balance before moving forward. |
| **Simulation Speed** | 32 parallel arenas ($\approx 6,000$ FPS) | 10 parallel arenas ($\approx 1,200$ FPS) | Heavier physics calculation per frame. |

---

## What's Next?

Now that the MDP structure is clear and intuitive, we move to **[02_algorithm_selection.md](02_algorithm_selection.md)** to decide between candidate algorithms (PPO vs. SAC/TD3) for training this 20-DOF continuous locomotion agent.
