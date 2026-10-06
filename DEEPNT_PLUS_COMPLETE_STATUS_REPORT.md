# DeepNT+ Complete Status Report

Date: 2026-10-01

## Executive Summary

DeepNT+ is a hybrid, path-centric network tomography model. It predicts the RTT of hidden network paths from topology, structural node features, observed path RTTs, tomography relationships, and optional temporal context.

The codebase is implemented and the synthetic test suite passes. The recorded real-data experiment also completed, but its result should be treated as provisional because the saved `run_02` output is monitor-confounded: validation and test contain only `dac-bd`, while training contains all monitors.

The current recorded test ranking is:

| Model | Test MAE (ms) |
|---|---:|
| Classical convex tomography | 59.6 |
| DeepNT static | 65.7 |
| DeepNT+ full | 74.1 |
| DeepNT static causal | 74.2 |
| GraphSAGE | 91.1 |
| GAT | 90.3 |
| Median | 123.5 |

The simpler convex and static models currently outperform the full temporal model.

## Research Goal

The model addresses this task:

> Given a network graph and RTT measurements for observed paths, estimate RTT for hidden paths without using the hidden RTT as model input.

The hidden RTT is used only by the training loss and evaluation metrics. At validation and test time, the model receives the fixed observed-path mask and predicts the unobserved paths.

## Data Construction

The authoritative current data path is `Dataset_Build.ipynb`.

Its processing stages are:

1. Read CAIDA Ark traceroute RTT data.
2. Detect whether RTT values are in microseconds or milliseconds.
3. Convert RTT to milliseconds.
4. Keep only traceroutes with contiguous hops and a responding destination.
5. Map hop IP addresses to real ITDK nodes.
6. Require every consecutive hop pair to exist as a real ITDK link.
7. Record every rejected traceroute and its reason.
8. Partition usable paths into 15 graph-level splits.
9. Mark approximately 30% of each graph's paths as observed.
10. Store true hidden RTT only in `path_ground_truth.csv`.
11. Build structural node features, edge features, tomography matrices, and temporal events.
12. Fit normalization statistics using training graphs only.

Recorded real-data summary:

- Raw traceroutes: 51,072.
- Usable paths: 1,326.
- Usable fraction: approximately 2.6%.
- Graphs: 15.
- Observed paths: 396.
- Hidden paths: 930.
- Edges: 6,598.
- Temporal events: 396.
- Packet-loss labels: unavailable.
- RTT outliers in the recorded run: 0.

The strict real-ITDK policy is scientifically defensible, but it discards most raw traceroutes. The permissive alternative in `Graph_from_RTT.ipynb` keeps more data by allowing anonymous/private inferred nodes. That alternative is not the current model path.

## DeepNT+ Architecture

The implementation is in `deepnt_plus/model.py`, `layers.py`, and `tgn.py`.

### 1. Structural GNN

The default split-weight graph convolution computes a separate self transformation and neighbor transformation:

$$
h'_v = W_s h_v + W_n \operatorname{mean}_{u \in N(v)} h_u
$$

The input includes structural node features and, when enabled, a summary of observed context measurements at each node.

GraphSAGE and GAT are implemented as comparison models.

### 2. Path Encoder

Each path's ordered node embeddings are processed with:

- Positional embeddings.
- Transformer self-attention.
- Masked mean pooling.

This produces a path-level representation.

### 3. Tomography Layer

The GNN predicts a non-negative latent delay for each edge. Path delay is computed from the path-edge incidence matrix:

$$
\hat{y}_{\text{additive}} = A x + b
$$

Because the target is end-to-end round-trip time, this is an approximation rather than an exact one-way additive model. The intercept and residual head absorb non-additive effects.

### 4. Convex Refinement

For observed context paths, a closed-form ridge update refines the learned edge prior:

$$
\min_x \|A_c x - y_c\|^2 + \alpha \|x - x_0\|^2
$$

The refined estimate is constrained to be non-negative.

### 5. Learned Fusion

A learned gate chooses the contribution of the convex estimate versus the neural estimate for each target path. The gate is initialized to favor the convex estimate.

### 6. Temporal Branch

The temporal branch contains:

- TGN-style per-node memory.
- GRU updates for observed path events.
- Time-gap encodings.
- A recent-history GRU.
- A Transformer over related path measurements.

The dataset has very few repeated source-destination pairs, so history is formed from recent related paths, especially paths with overlapping edges. Causal evaluation divides target paths into time blocks and allows only earlier context measurements.

### 7. Uncertainty

The uncertainty head estimates aleatoric variance in normalized log-RTT space. MC-dropout estimates epistemic variance. Calibration metrics include NLL, sharpness, interval coverage, and regression ECE.

### 8. Active Learning

Active learning is simulated. The system reveals real held-back RTT measurements and adds them to the context. It does not generate new network probes.

Implemented strategies:

- Random.
- Predictive variance.
- BALD.
- Edge coverage.
- Longest path.

## Training and Evaluation Protocol

Training is episodic:

- Each episode samples context and target paths.
- Training uses random context masks.
- Validation and test use fixed observed/unobserved masks.
- RTT outliers are excluded from context, loss, and metrics.
- Validation MAE selects the checkpoint.
- Test graphs are evaluated after model selection.
- Multiple random seeds are summarized with mean and standard deviation.

The loss combines normalized RTT prediction loss, tomography consistency loss, and optional packet-loss loss. No packet-loss head is trained in the current dataset because labels are unavailable.

## Baselines and Ablations

The experiment runner compares:

- Median.
- Hop-count regression.
- Linear regression.
- Random forest.
- MLP.
- Classical convex tomography.
- GraphSAGE.
- GAT.
- DeepNT static.
- DeepNT static causal.
- DeepNT+ full.

Ablations remove:

- Temporal branch.
- Convex fusion.
- Tomography layer.
- Uncertainty head.
- Context features.

The active implementation is in `deepnt_plus/run.py`, `baselines.py`, and `active.py`.

## Recorded Real-Data Results

### Test split

| Model | MAE mean (ms) | RMSE mean (ms) | R2 mean |
|---|---:|---:|---:|
| Classical convex tomography | 59.555 | 200.634 | 0.047 |
| DeepNT static | 65.676 | 202.726 | 0.027 |
| DeepNT+ full | 74.105 | 207.220 | -0.017 |
| DeepNT static causal | 74.207 | 206.976 | -0.014 |
| DeepNT+ without temporal branch | 77.002 | 208.367 | -0.028 |
| DeepNT+ uncertainty ablation | 80.625 | 209.520 | -0.040 |
| DeepNT+ context-feature ablation | 81.112 | 210.199 | -0.046 |
| DeepNT+ convex-fusion ablation | 84.376 | 210.702 | -0.052 |
| GAT | 90.341 | 211.899 | -0.063 |
| GraphSAGE | 91.102 | 211.774 | -0.062 |
| DeepNT+ tomography ablation | 105.303 | 220.136 | -0.148 |
| Hop-count linear | 118.831 | 229.948 | -0.252 |
| Median | 123.536 | 232.314 | -0.278 |
| Linear regression | 125.621 | 234.207 | -0.299 |
| Random forest | 138.088 | 240.348 | -0.368 |
| MLP | 225.620 | 307.109 | -1.233 |

### Validation split

The validation results were generally better than test results. DeepNT static achieved approximately 44.0 ms MAE, classical convex tomography approximately 46.0 ms, and DeepNT+ full approximately 47.1 ms.

### Uncertainty

The recorded uncertainty results show that the model produces usable but imperfect calibration. The causal static model had the lowest recorded NLL and ECE among the reported variants, while the full DeepNT+ model did not provide the best calibration.

## Important Data-Split Finding

The intended design says every monitor should be represented across train, validation, and test. However, the executed `DeepNT_Plus.ipynb` output for `run_02` reports:

| Split | abz2-uk | acc-gh | dac-bd |
|---|---:|---:|---:|
| train | 431 | 417 | 38 |
| val | 0 | 0 | 176 |
| test | 0 | 0 | 264 |

This means the recorded validation and test results are effectively evaluating on the `dac-bd` vantage point while training contains all monitors. The result may therefore reflect a monitor/vantage shift rather than only graph generalization.

The package already supports an explicit holdout mode with `load_dataset(processed_dir, holdout_monitor=...)` in `deepnt_plus/data.py`. That mode should be used for a clearly labeled leave-monitor-out experiment.

## Other Graph Pipelines

### `Graph_rep.ipynb`

This is a strict graph-to-RTT connection and audit pipeline. It requires a confirmed graph-ID to ITDK-ID mapping and intentionally stops if the mapping cannot be recovered. Its recorded output confirms that no mapping file was available, so it should not be used as the main data path unless the mapping is recovered.

### `Graph_from_RTT.ipynb`

This is an alternative permissive pipeline. It constructs graphs directly from RTT traceroutes and allows unresolved or silent hops to become scoped inferred nodes. It avoids the graph-ID mapping problem but changes the data assumptions and should be treated as a separate experiment.

### Current authoritative path

The current model experiments use:

`Dataset_Build.ipynb` -> `deepnt_plus.data` -> `DeepNT_Plus.ipynb`

## Validation Status

The command below was executed successfully:

```text
python tests/test_deepnt_plus.py
```

All listed synthetic tests passed, including:

- Data loading.
- Train/test split construction.
- Monitor holdout split logic.
- Robust metrics.
- GAT attention behavior.
- Closed-form ridge equivalence.
- No target RTT leakage.
- Temporal causality.
- All model variants and empty-context behavior.
- Training improvement over the median baseline.
- Outlier exclusion.
- Calibration metrics.
- Active-learning execution.
- Complete experiment output generation.

## Outstanding Work

No executable `TODO` or `FIXME` items were found in the source package. The remaining work is experimental and methodological.

### Required next experiment: verified real-data split

- Rebuild or verify the processed dataset used by `run_02`.
- Confirm monitor counts per graph before training.
- Confirm that the intended train/validation/test monitor distribution is actually present.
- Rerun all variants with the verified split.
- Save the new run under a new output directory instead of overwriting `run_02`.

### Required generalization experiment: leave-monitor-out

- Run `load_dataset(..., holdout_monitor='dac-bd')`.
- Train only on non-holdout monitor graphs.
- Use the holdout monitor as test.
- Report train, validation, and test monitor identities explicitly.
- Compare against classical tomography and static DeepNT.

### Required pipeline decision

Choose one of the following and state it explicitly in the thesis:

1. Strict real-ITDK pipeline: fewer paths, no inferred topology.
2. RTT-grounded permissive pipeline: more paths, scoped inferred nodes.

Do not combine results from the two pipelines without labeling them as separate datasets.

### Model interpretation

The current results suggest:

- Tomography is important.
- Convex refinement is useful.
- Static DeepNT may be more reliable than the temporal branch on this dataset.
- The temporal branch may be limited by sparse repeated endpoint histories and the small number of observed events.
- More complexity does not currently translate into lower test MAE.

These are observations, not final claims, until the split is corrected and rerun.

## Final Current State

The repository is a working research prototype with:

- A reproducible data builder.
- A complete hybrid DeepNT+ implementation.
- Baselines and ablations.
- Causal evaluation.
- Uncertainty estimation.
- Simulated active learning.
- Synthetic tests covering the important contracts.
- A completed but methodologically questionable real-data run.

The next defensible milestone is a new real-data run with verified monitor balancing or an explicitly labeled leave-monitor-out protocol. Until that run is complete, `run_02` should be treated as an exploratory result rather than the final thesis result.
