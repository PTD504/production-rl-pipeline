"""ONNX Export and Production Packaging Script (export_onnx.py).

Exports a trained Stable-Baselines3 PPO policy into a standalone, production-ready
ONNX model with fused observation normalization (VecNormalize mean & std).
Verifies numerical parity against PyTorch and benchmarks inference latency.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained RL policy to production ONNX format")
    parser.add_argument(
        "--model",
        type=str,
        default="experiments/outputs/default_push_block_ppo/seed_42/best_model/best_model.zip",
        help="Path to trained model .zip file.",
    )
    parser.add_argument(
        "--vec-normalize",
        type=str,
        default=None,
        help="Path to final_vec_normalize.pkl (default: auto-detected).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/push_block_policy.onnx",
        help="Output path for exported ONNX model (default: models/push_block_policy.onnx).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX operator set version (default: 18).",
    )
    parser.add_argument(
        "--no-fuse-norm",
        action="store_true",
        help="Disable fusing VecNormalize into the ONNX graph (model will expect pre-normalized inputs).",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=1000,
        help="Number of inference steps for latency benchmark (default: 1000).",
    )
    return parser.parse_args()


def find_vec_normalize_path(model_path: Path) -> Optional[Path]:
    """Search for vec_normalize.pkl in parent directories of the model."""
    candidates = [
        model_path.parent / "final_vec_normalize.pkl",
        model_path.parent.parent / "final_vec_normalize.pkl",
        model_path.parent / "vec_normalize.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


class StandalonePolicy(nn.Module):
    """Production wrapper fusing observation normalization and policy network.

    If mean and variance statistics are provided, they are registered as persistent
    buffers so that the normalization transformation:
        z = clamp((x - mean) / std, -clip_obs, clip_obs)
    is embedded directly into the ONNX computational graph. This makes the model
    fully self-contained for downstream clients (Unity Sentis, C#, C++, Microservices).
    """

    def __init__(
        self,
        policy: nn.Module,
        mean: Optional[np.ndarray] = None,
        var: Optional[np.ndarray] = None,
        epsilon: float = 1e-8,
        clip_obs: float = 10.0,
    ):
        super().__init__()
        if mean is not None and var is not None:
            self.register_buffer("obs_mean", torch.tensor(mean, dtype=torch.float32).unsqueeze(0))
            self.register_buffer("obs_std", torch.sqrt(torch.tensor(var, dtype=torch.float32) + epsilon).unsqueeze(0))
            self.clip_obs = float(clip_obs)
            self.has_normalization = True
        else:
            self.has_normalization = False

        # Extract only Actor network components (discard Critic/Value network)
        self.features_extractor = policy.features_extractor
        self.mlp_policy_net = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs: Observation tensor (shape: [batch_size, obs_dim]).

        Returns:
            discrete_action: Chosen action index with highest logit (shape: [batch_size]).
            action_logits: Unnormalized logits for all actions (shape: [batch_size, num_actions]).
        """
        if self.has_normalization:
            obs = torch.clamp((obs - self.obs_mean) / self.obs_std, -self.clip_obs, self.clip_obs)

        features = self.features_extractor(obs)
        latent_pi = self.mlp_policy_net(features)
        action_logits = self.action_net(latent_pi)
        discrete_action = torch.argmax(action_logits, dim=-1)
        return discrete_action, action_logits


def verify_numerical_parity(
    torch_model: nn.Module,
    onnx_path: Path,
    obs_dim: int,
    num_samples: int = 100,
    tolerance: float = 1e-5,
) -> bool:
    """Verifies that PyTorch and ONNX Runtime produce identical outputs."""
    print("\n[VERIFICATION] Running Numerical Parity Check...")

    # Load ONNX Runtime session
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    # Generate synthetic observations spanning diverse ranges
    test_obs = np.random.uniform(low=0.0, high=1.0, size=(num_samples, obs_dim)).astype(np.float32)

    # 1. PyTorch inference
    torch_model.eval()
    with torch.no_grad():
        pt_actions, pt_logits = torch_model(torch.from_numpy(test_obs))
        pt_actions_np = pt_actions.numpy()
        pt_logits_np = pt_logits.numpy()

    # 2. ONNX Runtime inference
    ort_inputs = {"obs": test_obs}
    ort_actions_np, ort_logits_np = session.run(None, ort_inputs)

    # Compare logits and action predictions
    max_logit_diff = float(np.max(np.abs(pt_logits_np - ort_logits_np)))
    action_match_rate = float(np.mean(pt_actions_np == ort_actions_np)) * 100.0

    print(f"  - Test Samples Evaluated : {num_samples}")
    print(f"  - Maximum Logit Difference: {max_logit_diff:.8e}")
    print(f"  - Action Match Agreement : {action_match_rate:.2f}%")

    if max_logit_diff > tolerance:
        print(f"[FAIL] Numerical parity check failed: logit difference {max_logit_diff} exceeds tolerance {tolerance}.")
        return False

    if action_match_rate < 100.0:
        print(f"[FAIL] Discrete action mismatch between PyTorch and ONNX Runtime ({action_match_rate}%).")
        return False

    print(f"[PASS] Numerical parity verified successfully (tolerance < {tolerance}).")
    return True


def benchmark_inference_latency(
    onnx_path: Path,
    obs_dim: int,
    benchmark_runs: int = 1000,
    warmup_runs: int = 50,
) -> None:
    """Measures single-sample inference latency percentiles on CPU."""
    print(f"\n[BENCHMARK] Measuring Single-Sample Inference Latency ({benchmark_runs} runs)...")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    single_obs = np.random.uniform(0.0, 1.0, size=(1, obs_dim)).astype(np.float32)
    inputs = {"obs": single_obs}

    # Warmup
    for _ in range(warmup_runs):
        session.run(None, inputs)

    # Timed runs
    latencies = []
    for _ in range(benchmark_runs):
        t0 = time.perf_counter()
        session.run(None, inputs)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # Convert to milliseconds

    latencies_arr = np.array(latencies)
    mean_lat = np.mean(latencies_arr)
    p50_lat = np.percentile(latencies_arr, 50)
    p90_lat = np.percentile(latencies_arr, 90)
    p95_lat = np.percentile(latencies_arr, 95)
    p99_lat = np.percentile(latencies_arr, 99)
    throughput = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    print(f"  - Mean Latency : {mean_lat:.3f} ms")
    print(f"  - P50 Latency  : {p50_lat:.3f} ms")
    print(f"  - P90 Latency  : {p90_lat:.3f} ms")
    print(f"  - P95 Latency  : {p95_lat:.3f} ms")
    print(f"  - P99 Latency  : {p99_lat:.3f} ms")
    print(f"  - Throughput   : {throughput:,.1f} inferences/sec (single CPU core)")


def main():
    args = parse_args()
    model_path = Path(args.model).resolve()
    if not model_path.exists():
        fallback = model_path.parent.parent / "final_model.zip"
        if fallback.exists():
            model_path = fallback
        else:
            raise FileNotFoundError(f"Model file not found at: {model_path}")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vec_norm_path = Path(args.vec_normalize).resolve() if args.vec_normalize else find_vec_normalize_path(model_path)

    print("=" * 70)
    print("PRODUCTION RL PIPELINE: ONNX MODEL EXPORT")
    print("=" * 70)
    print(f"  - Source Model        : {model_path}")
    print(f"  - Normalization File  : {vec_norm_path if vec_norm_path else 'None'}")
    print(f"  - Fuse Normalization  : {'Enabled (Self-Contained Graph)' if not args.no_fuse_norm else 'Disabled'}")
    print(f"  - Target Output Path  : {output_path}")
    print(f"  - ONNX Opset Version  : {args.opset}")
    print("=" * 70)

    # 1. Load PyTorch model
    print("\n[1/4] Loading trained PyTorch policy...")
    model = PPO.load(str(model_path), device="cpu")
    policy = model.policy

    # Extract observation and action dimensions
    obs_dim = int(policy.observation_space.shape[0])
    num_actions = int(policy.action_space.n)

    # 2. Extract Normalization Statistics (if applicable)
    mean = None
    var = None
    epsilon = 1e-8
    clip_obs = 10.0

    if not args.no_fuse_norm and vec_norm_path and vec_norm_path.exists():
        import pickle

        with open(vec_norm_path, "rb") as f:
            vn = pickle.load(f)
        mean = vn.obs_rms.mean
        var = vn.obs_rms.var
        epsilon = getattr(vn, "epsilon", 1e-8)
        clip_obs = getattr(vn, "clip_obs", 10.0)
        print(f"  [INFO] Extracted running mean (dim={mean.shape[0]}) and variance from: {vec_norm_path.name}")
    elif args.no_fuse_norm:
        print("  [INFO] Normalization fusion skipped by user flag.")
    else:
        print("  [WARN] No VecNormalize file found; exporting raw policy network.")

    # 3. Construct Standalone Policy Wrapper
    print("\n[2/4] Assembling standalone production policy graph...")
    export_model = StandalonePolicy(
        policy=policy,
        mean=mean,
        var=var,
        epsilon=epsilon,
        clip_obs=clip_obs,
    )
    export_model.eval()

    # 4. Export to ONNX
    print(f"\n[3/4] Exporting to ONNX format (opset {args.opset})...")
    dummy_input = torch.zeros(1, obs_dim, dtype=torch.float32)

    torch.onnx.export(
        export_model,
        dummy_input,
        str(output_path),
        input_names=["obs"],
        output_names=["discrete_action", "action_logits"],
        dynamic_axes={
            "obs": {0: "batch_size"},
            "discrete_action": {0: "batch_size"},
            "action_logits": {0: "batch_size"},
        },
        opset_version=args.opset,
    )

    # Validate ONNX model integrity
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    file_size_kb = output_path.stat().st_size / 1024.0

    print(f"  [SUCCESS] ONNX model exported and validated successfully!")
    print(f"  - File Path   : {output_path}")
    print(f"  - File Size   : {file_size_kb:.2f} KB (Optimized, Critic stripped)")
    print(f"  - Input Node  : 'obs' (shape: [batch_size, {obs_dim}], dtype: float32)")
    print(f"  - Output Node 1: 'discrete_action' (shape: [batch_size], dtype: int64)")
    print(f"  - Output Node 2: 'action_logits' (shape: [batch_size, {num_actions}], dtype: float32)")

    # 5. Verification & Benchmark
    print("\n[4/4] Production Readiness Verification...")
    parity_ok = verify_numerical_parity(
        torch_model=export_model,
        onnx_path=output_path,
        obs_dim=obs_dim,
        num_samples=100,
        tolerance=1e-5,
    )

    if not parity_ok:
        sys.exit(1)

    if args.benchmark_runs > 0:
        benchmark_inference_latency(
            onnx_path=output_path,
            obs_dim=obs_dim,
            benchmark_runs=args.benchmark_runs,
        )

    print("\n" + "=" * 70)
    print("ONNX EXPORT AND VERIFICATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()
