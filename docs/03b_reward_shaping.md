# Reward Shaping: Concepts, Pitfalls, and Practical Guidelines

## Overview

When training a Reinforcement Learning agent, there is a common saying among practitioners:

> *"If an agent isn't learning, most of the time the issue is the reward function, not the neural network."*

Right after running a baseline ([03_baseline_experiments.md](03_baseline_experiments.md)), the first practical question to ask is: **Did the agent actually learn the intended task?**

If the answer is no, before spending days tuning hyperparameters or swapping algorithms, the first lever you look at is **Reward Shaping**.

This document explains what reward shaping is at a high level, the major traps to avoid, the core principles of designing safe rewards, and why our Push Block baseline worked without any custom shaping.

---

## 1. What is Reward Shaping?

In RL, the reward signal defines what the agent strives to achieve. These signals typically come in two forms:

- **Sparse Rewards:** The agent only receives a reward when it fully succeeds (e.g., `+1.0` for reaching the goal, `0.0` for everything else). While this cleanly describes the true objective, it can be extremely difficult to learn. In complex environments, an agent wandering randomly might take millions of steps before stumbling upon the goal by chance.
- **Dense Rewards:** The agent receives continuous feedback throughout the episode (like playing the childhood game of *"warmer / colder"* as you approach a hidden object).

**Reward Shaping** simply means adding intermediate guidance signals to the environment's reward to help the agent learn faster and explore more effectively.

---

## 2. The Biggest Trap: Reward Hacking

While adding extra rewards sounds straightforward, it comes with a notorious danger called **Reward Hacking** (or reward gaming).

Reinforcement learning agents do not understand human common sense or intent—they are purely numerical optimizers. If there is any loophole in your reward design that allows the agent to collect points without genuinely solving the problem, the agent will inevitably discover and exploit it.

### Well-Known Examples:
- **The Racing Boat Loop:** In a boat racing video game, developers awarded small points for hitting intermediate targets along the track. Instead of racing to the finish line, the agent learned to turn around, crash repeatedly, and do endless donuts in a small lagoon because the targets respawned, racking up infinite points without ever completing the race.
- **The Hesitant Robot Arm:** When training a robotic arm to pick up an object, developers rewarded the arm for getting closer to the object. The robot quickly learned to hover its hand directly above the object forever. Picking it up was risky (it might drop it and lose points), whereas hovering generated a steady stream of proximity points safely.

**Golden Rule:** Always ask yourself: *"Can the agent maximize this intermediate score without actually solving the true task?"*

---

## 3. High-Level Principles for Safe Shaping

If you do need to shape rewards in your own projects, keep these core principles in mind:

### 1. Reward Progress, Not States
Never reward the agent simply for *being* in a good location, because that encourages stalling. Instead, reward the agent for *making forward progress* relative to its previous step. 

### 2. The "Conservation" Intuition (Potential-Based Shaping)
A sound way to think about safe shaping is conservation: any reward bonus the agent gains by moving closer to the goal must be surrendered if it retreats. This ensures the agent cannot loop back and forth to harvest free points. In literature, this concept is formalized as **Potential-Based Reward Shaping (PBRS)**, which mathematically guarantees that the optimal policy remains unchanged.

### 3. Consider Curriculum Learning
If shaping the reward feels too risky or unnatural, another powerful high-level approach is **Curriculum Learning**: start the agent in an easy scenario (e.g., placing the block right next to the goal) so it experiences success early, and gradually increase the difficulty as its competency grows.

---

## 4. Engineering Practice: Keep Shaping Modular

From a software engineering standpoint, **never hardcode custom reward logic into the core environment simulation**.

Instead, wrap the environment using an external decorator (such as a Gymnasium `RewardWrapper`). This keeps your original environment code clean, preserves the true benchmark metrics, and makes it trivial to turn shaping on or off across different experiments.

---

## 5. Our Case: Why Push Block Did Not Need Shaping

In our Push Block baseline, we deliberately avoided adding custom reward shaping because the default environment already provided the right ingredients:

1. **A Small Step Penalty (`-0.001`):** Gently nudges the agent to solve the task quickly rather than wasting time.
2. **A Clear Terminal Reward (`+5.0`):** Dominates the small step penalties and provides an unambiguous end goal.
3. **Massively Parallel Exploration (32 Arenas):** Running 32 arenas simultaneously provided enough diverse random exploration that agents collided with the block and scored within the first 50,000 steps without needing artificial hints.

Because the baseline converged smoothly and achieved near-maximum theoretical return on its own, leaving the reward unshaped was the cleanest, most robust engineering choice.
