# Continuous Model Export & Production Packaging: Crawler ONNX Deployment

## Overview & Objectives

In the previous phases, we trained and validated a continuous Soft Actor-Critic (SAC) policy for the 20-DOF Crawler robot across 10 unseen environment seeds ([05_multi_seed_evaluation.md](05_multi_seed_evaluation.md)), achieving an average return of $+1,430.99$ and a $75.0\%$ survival rate.

Inheriting the export methodology established in Case Study 1 ([05_model_export_onnx.md](../05_model_export_onnx.md)), Phase 6 focuses on packaging the continuous Crawler policy into a standalone **ONNX** binary (`models/crawler_policy.onnx`). 

This document highlights the continuous-control adaptations, empirical parity verification, latency benchmarks, and deployment steps.

---

## 1. Continuous Control Export Architecture

The export pipeline ([scripts/export_onnx.py](../../scripts/export_onnx.py)) incorporates two key adaptations for continuous robotic locomotion:

### 1.1 Fused Normalization for 158 Sensors

The 158 continuous observation values (torso velocities, quaternions, limb orientations, joint angles, angular velocities, and raycast contacts) were normalized during training using running statistics.

Rather than requiring downstream clients to manage normalization logic, the exporter extracts the frozen statistics ($\mu \in \mathbb{R}^{158}, \sigma^2 \in \mathbb{R}^{158}$) from `final_vec_normalize.pkl` and bakes them directly as constant tensor operations at the entrance of the ONNX graph:

$$z = \text{clamp}\left(\frac{x - \mu}{\sqrt{\sigma^2 + 10^{-8}}}, -10.0, 10.0\right)$$

Clients supply raw, unnormalized sensor readings directly from the robot or simulation; normalization runs entirely within the optimized ONNX runtime.

### 1.2 Deterministic Torque Mapping (No ArgMax)

Unlike discrete tasks that require an `ArgMax` operator over categorical action logits, continuous quadruped control requires real-valued joint torques in $[-1.0, 1.0]^{20}$.

- **Stochastic Sampling Removed:** The policy wrapper (`StandaloneContinuousPolicy`) extracts the Gaussian mean head $\mu(s)$ and bounds it deterministically using hyperbolic tangent ($\tanh$):
  $$a_{\text{prod}} = \tanh(\mu(s)) \in [-1.0, 1.0]^{20}$$
- **Direct Output:** The ONNX model outputs a single tensor `continuous_action` (shape `[batch_size, 20]`, `float32`).

### 1.3 Checkpoint Trimming & Size Reduction

During training, the SAC checkpoint carries dual Critic networks ($Q_1, Q_2$), target Critic networks ($\bar{Q}_1, \bar{Q}_2$), log standard deviation parameters, and Adam optimizer momentums.

For production inference, all Critic branches and optimizer buffers are discarded:

| Metric | Source Checkpoint (`best_model.zip`) | Exported Model (`crawler_policy.onnx`) | Change |
| :--- | :---: | :---: | :---: |
| **File Format** | Zip (PyTorch weights + optimizer states) | Self-Contained ONNX Binary | Standalone |
| **Contents** | Actor, 2x Critics, Targets, Adam States | Fused Normalization + Actor MLP | Critics & Optimizers stripped |
| **Disk Size** | **4.9 MB** (4,930 KB) | **439 KB** (438.88 KB) | **-91.2% footprint reduction** |

---

## 2. Numerical Parity Verification

To verify that the ONNX model reproduces PyTorch inference outputs down to floating-point precision, we evaluated both runtimes on 100 synthetic observation states spanning $[-1.0, 1.0]$:

```bash
uv run python scripts/export_onnx.py \
  --model experiments/outputs/tuned_crawler_sac/seed_42/best_model/best_model.zip \
  --output models/crawler_policy.onnx
```

### Verification Results

```text
[VERIFICATION] Running Numerical Parity Check...
  - Test Samples Evaluated : 100
  - Mean Absolute Error (MAE): 9.75792389e-08
  - Maximum Absolute Error   : 6.13927841e-06
[PASS] Numerical parity verified successfully (tolerance < 1e-05).
```

- **Mean Absolute Error (MAE):** $9.76 \times 10^{-8}$
- **Maximum Absolute Error:** $6.14 \times 10^{-6}$ (safely within the $1.0 \times 10^{-5}$ precision tolerance).

---

## 3. Inference Latency & Physics Budget Benchmark

In continuous robotics and simulation, the Crawler physics engine runs at **50 Hz** ($20.0\text{ ms}$ per physics tick). Real-time control requires inference to complete in a fraction of this window.

We measured 1,000 single-sample inferences (`batch_size = 1`) on a single CPU core using ONNX Runtime:

| Metric | Measured Result | Allowed Budget (50 Hz Physics) | Overhead |
| :--- | :---: | :---: | :---: |
| **Mean Latency** | **0.026 ms** ($26\ \mu\text{s}$) | $< 1.000\text{ ms}$ | **0.13%** of physics tick |
| **Median (P50)** | **0.027 ms** ($27\ \mu\text{s}$) | $< 1.000\text{ ms}$ | Negligible |
| **P90 Latency** | **0.028 ms** ($28\ \mu\text{s}$) | $< 1.500\text{ ms}$ | Consistent |
| **P99 Latency** | **0.034 ms** ($34\ \mu\text{s}$) | $< 5.000\text{ ms}$ | Zero latency spikes |
| **Throughput** | **~38,700 inf/sec** | $> 1,000\text{ inf/sec}$ | $38\times$ target throughput |

At $26\ \mu\text{s}$ per decision, inference consumes only $0.13\%$ of each $20\text{ ms}$ physics cycle, leaving over $99.8\%$ of compute headroom for physics, inverse kinematics, and graphics rendering.

---

## 4. Deployment & Client Usage

### 4.1 Python / Edge Integration

```python
import numpy as np
import onnxruntime as ort

# 1. Initialize session once at startup
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.intra_op_num_threads = 1
session = ort.InferenceSession("models/crawler_policy.onnx", opts, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# 2. Ingest raw 158-dimensional sensor vector directly (normalization is fused in the graph)
raw_sensors = np.random.uniform(-1.0, 1.0, size=(1, 158)).astype(np.float32)

# 3. Compute continuous joint torques
motor_torques = session.run(None, {input_name: raw_sensors})[0][0]  # shape: (20,), range: [-1.0, 1.0]
```

### 4.2 Unity Sentis / Barracuda C# Integration

1. Place `models/crawler_policy.onnx` into the Unity project's `Assets/` directory.
2. Select the Crawler Agent in the scene hierarchy.
3. In **Behavior Parameters**:
   - **Vector Observation Space Size:** `158`
   - **Vector Action Space:** `Continuous`, Size: `20`
   - **Model:** Drag and drop `crawler_policy.onnx`
   - **Behavior Type:** `Inference Only`
4. The Unity Sentis engine executes the ONNX graph directly in C# without Python.

### 4.3 Real-Time 3D Desktop Validation

To visually verify the exported ONNX model running live across 10 Crawler arenas without PyTorch:

```bash
uv run python scripts/visualize.py \
  --config configs/experiments/tuned_crawler_sac.yaml \
  --onnx models/crawler_policy.onnx \
  --time-scale 1.0
```

---

## 5. Summary

| Milestone Step | Status | Evidence |
| :--- | :---: | :--- |
| **Model Size** | Complete | Stripped critics and optimizers, reducing checkpoint from **4.9 MB to 439 KB** (-91.2%). |
| **Preprocessing** | Complete | Fused 158-sensor running normalization directly into the graph. |
| **Parity** | Complete | Verified across 100 test states with MAE $9.76 \times 10^{-8}$ and max error $6.14 \times 10^{-6}$. |
| **Latency** | Complete | **$0.026\text{ ms}$** mean latency on single CPU core ($0.13\%$ of 50 Hz physics budget). |
| **Deployment** | Complete | Validated via standalone ONNX Runtime and ready for Unity Sentis. |
