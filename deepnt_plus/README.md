# deepnt_plus

Hybrid path-centric GNN for network tomography (thesis model DeepNT+), written in plain PyTorch. It reads the
dataset built by `Dataset_Build.ipynb` and is driven by `DeepNT_Plus.ipynb`.

## Run

```
python tests/test_deepnt_plus.py        # 11 tests on synthetic data (leakage, causality, shapes, learning, ...)
```
Then open `DeepNT_Plus.ipynb`. In Colab copy this folder to `My Drive/Thesis/DeepNT_Output/code/deepnt_plus`.
Requirements: torch, numpy, pandas, scipy, scikit-learn, matplotlib.

## Thesis component -> code

| Thesis component | Where | Notes |
|---|---|---|
| Split-weight spatial GNN | `layers.SplitWeightConv`, `model.DeepNTPlus` | `h' = W_s h + W_n mean(h_N)` |
| Path encoder | `layers.PathEncoder` | positional encoding + self-attention over the ordered nodes of a path |
| Tomography relation `y ~ A x` | `model.DeepNTPlus` | GNN edge latents `x >= 0`, path RTT `= A x + b` |
| Convex fusion | `model.DeepNTPlus` | ridge tomography `min ||Ax-y||^2 + a||x-x0||^2` on the context paths, closed form, `x0` = GNN prior; `y_hat = lam*y_convex + (1-lam)*y_gnn`, `lam` a learned per-path gate warm-started high |
| TGN memory, time encoding, GRU | `tgn.TemporalBranch`, `layers.TimeEncoder` | per-node memory updated by every observed path (event), GRU update |
| GRU + Transformer over history | `tgn.TemporalBranch` | last K related measurements, GRU then Transformer with a query token |
| Uncertainty | `model.var_head`, `train.compute_loss`, `train.predict_frame` | heteroscedastic Gaussian NLL, MC-dropout epistemic variance |
| Active learning | `active.py` | variance, BALD (Gaussian), edge-coverage, longest-path, random |
| Baselines | `baselines.py`, `run.VARIANTS` | median, hop-count, LR, RF, MLP, classical convex tomography, GraphSAGE, GAT |

## Protocol (spec sections 7, 8, 17)

- **Graph-level split**: train `graph_01-10`, validation `11-12`, test `13-15`. Normalisation is fitted on train graphs only.
- **Task**: the model sees the RTT of the *context* paths and predicts the RTT of the *target* paths of the same graph.
  Validation/test: context = the fixed `observed` mask, targets = the hidden paths. Training: random context masks every epoch.
- **No leakage**: `DeepNTPlus.forward` receives `y_ctx` only and never reads `g.y`; `tests/test_no_target_leakage` changes every
  hidden RTT and asserts the predictions are bit-identical. Hidden RTT lives only in `path_ground_truth.csv`.
- **Causal mode** (temporal branch, spec section 17): targets are cut into 6 time blocks and each block only sees context measured
  strictly before it. The temporal branch is only allowed in this mode. `tests/test_causality_*` perturbs a late measurement and
  asserts earlier blocks do not change. `DeepNT (static, causal)` is the fair non-temporal comparison for the full model.
- **Outliers** (`quality_flag` `rtt_outlier*`) are excluded from context, training loss and metrics.
- **Model selection** on validation MAE only; test graphs are scored once, over several seeds.

## Deliberate deviations from the thesis text (state these in the thesis)

1. **Data is CAIDA Ark traceroutes, not RIPE Atlas.** The thesis document names RIPE Atlas; the dataset specification and everything
   built here use CAIDA Ark. The measurements cover about two days.
2. **The "path history" is built from related paths, not the same path.** Almost no (source, destination) pair repeats in the data, so a
   path has no history of its own. The history window is the most recent measurements of paths that share edges with the target.
3. **TGN events are whole-path measurements** that update the memory of every node on the path (not only the two endpoints), because
   destination routers are almost never shared between paths.
4. **GraphSAGE and the split-weight layer are the same function class** (`W[h || mean]` versus `W_s h + W_n mean`). The GraphSAGE baseline
   differs from DeepNT+ by lacking the tomography layer, convex fusion, temporal branch and uncertainty head, not by the aggregator.
5. **RTT is round-trip**, so `y ~ A x` is an approximation (reverse path and last hop are not additive); `b` and the residual head absorb it.
6. **Packet loss is not modelled**: the traceroute file has one probe per hop and no probe counts, so there is no defensible label. A loss
   head exists (`ModelConfig.has_loss`) and trains only where labels exist; today none do.
7. **Active learning is simulated** by revealing real measurements of held-back paths. It measures the value of a schedule, it does not
   run new probes.

## Files written by `run.run_all` (`experiments/deepnt/<run>/`)

`predictions.csv` (graph_id, path_id, source, destination, observed, y_true, y_pred, split, model, seed, uncertainty),
`metrics_summary.csv`, `history.csv`, `config.json` (all hyper-parameters, seeds, graph split, normalisation, protocol),
`active_learning.csv`, `deepnt_static.pt`, `deepnt_plus_full.pt`, and figures from `report.make_all`.
