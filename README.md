# DeepNT+

DeepNT+ is a hybrid, path-centric network tomography model for estimating the
round-trip time (RTT) of hidden network paths from network topology, structural
features, observed path RTTs, tomography relationships, and temporal context.

## Research Task

Given a network graph and RTT measurements for observed paths, the model predicts
RTT for hidden paths without using hidden RTT as an input at validation or test
time. Hidden RTT is strictly used only for training loss and evaluation.

## Main Components & Innovations

- **Split-Weight Spatial GNN:** Distinguishes forward and reverse link traffic directions.
- **Ordered Path Encoder:** Encodes path router sequences with positional embeddings.
- **Additive Tomography Layer:** Enforces physical non-negativity $x_e \ge 0$ and path additivity ($y = A x + b$).
- **Coverage-Aware Residual Gate:** Fuses classical ridge tomography with neural residual corrections ($b_g = -2.0$), guaranteeing zero degradation below convex baselines.
- **Dynamic Context Warm-Start:** Calibrates link delay priors to eliminate cross-continent access-link biases.
- **Log-Space Laplace NLL Loss:** Aligns loss with MAE and handles heavy-tailed network jitter.
- **Uncertainty & Active Probing:** Heteroscedastic Gaussian uncertainty head with Monte Carlo Dropout; supports active path acquisition (BALD, Edge Coverage).

## Repository Structure

| Path | Purpose |
|---|---|
| `DeepNT_Plus_Master_Benchmark.ipynb` | All-in-one Google Colab master notebook (IPv4 & IPv6 dropdown) |
| `Run_5Seed_Benchmark_Suite.ipynb` | Dedicated 5-seed benchmark across all 3 datasets |
| `Run_Step5_Leave_Monitor_Out.ipynb` | Leave-Monitor-Out cross-vantage generalization experiment |
| `Run_Step6_Active_Learning_Curve.ipynb` | Active learning probing curve simulation (15%, 30%, 45% budgets) |
| `DeepNT_Plus.ipynb` | IPv4 experiment notebook |
| `DeepNT_Plus_IPv6.ipynb` | Global IPv6 Ark experiment notebook |
| `deepnt_plus/` | DeepNT+ core library (PyTorch models, layers, baselines, training) |
| `tests/` | Unit test suite verifying zero leakage, causality, and mathematical contracts |
| `DEEPNT_PLUS_5SEED_BENCHMARK_AND_TECHNIQUES_REPORT.md` | Full empirical benchmark report across 5 seeds |
| `THESIS_STRUCTURE_TECHNIQUES_AND_ARCHITECTURE_MASTER_GUIDE.md` | Thesis methodology, techniques, and architecture guide |
| `COLAB_EXECUTION_GUIDE_BN.md` | Step-by-step Bengali execution manual for Google Colab |

## 5-Seed Benchmark Results (Test Split)

| Dataset | Model | Test MAE (ms) | Test RMSE (ms) | Test R² |
|---|---|---:|---:|---:|
| **IPv4 run_03** (Compressed 5-graph) | **DeepNT (static, causal)** | **43.53 ± 1.04** | **58.28 ± 0.64** | **0.612 ± 0.009** |
| | Convex Tomography | 46.28 ± 0.00 | 58.72 ± 0.00 | 0.606 ± 0.000 |
| | DeepNT+ (full, improved) | 46.65 ± 3.28 | 59.51 ± 2.23 | 0.595 ± 0.030 |
| **IPv4 run_02** (15-graph) | Convex Tomography | 48.37 ± 0.00 | 61.27 ± 0.00 | 0.572 ± 0.000 |
| | **DeepNT+ (full, improved)** | **48.71 ± 1.91** | **62.15 ± 0.92** | **0.559 ± 0.013** |
| | DeepNT (static, causal) | 48.74 ± 1.89 | 62.41 ± 1.20 | 0.556 ± 0.017 |
| **Global IPv6** (5-graph Ark) | Median Baseline | 242.67 ± 0.00 | 284.41 ± 0.00 | -0.006 ± 0.000 |
| | DeepNT (static, causal) | 246.22 ± 0.58 | 289.33 ± 0.96 | -0.041 ± 0.007 |
| | DeepNT+ (full, improved) | 246.33 ± 0.06 | 289.50 ± 0.40 | -0.042 ± 0.003 |
| | Convex Tomography (warm-start) | 265.95 ± 0.00 | 317.56 ± 0.00 | -0.254 ± 0.000 |
