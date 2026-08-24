"""End-to-end benchmark: load config + dataset, HPO, test eval, LOO importance, save results."""

from __future__ import annotations

import argparse
import warnings

from sliceheads.config import load_config
from sliceheads.explain import compute_importance_dataset
from sliceheads.hpo import run_benchmark
from sliceheads.results import save_results
from sliceheads.store import EmbedStore, validate_h5

warnings.filterwarnings("ignore")


def _print_hpo_table(head_name: str, all_results: list, best_params: dict, selection_metric: str) -> None:
    print(f"\n  {head_name} — {len(all_results)} combination(s) tried:")
    for r in sorted(all_results, key=lambda x: -(x["score"] or -999)):
        score = r.get("score")
        err = r.get("error", "")
        score_str = f"{score:.4f}" if score is not None else f"ERROR: {err}"
        marker = " <-- best" if r["params"] == best_params else ""
        params_str = "  ".join(f"{k}={v}" for k, v in r["params"].items())
        print(f"    {selection_metric}={score_str}  |  {params_str}{marker}")


def main(config_path: str, results_dir: str) -> None:
    print(f"Loading config: {config_path}")
    cfg = load_config(config_path)
    h5_path = cfg["data"]["h5_path"]
    selection_metric = cfg.get("training", {}).get("selection_metric", "roc_auc")

    print(f"Validating dataset: {h5_path}")
    validate_h5(h5_path)
    store = EmbedStore(h5_path)

    mean_embedding = store.mean_embedding()
    if mean_embedding is None:
        print("  mean_embedding missing — computing from train split …")
        mean_embedding = store.compute_and_write_mean_embedding()
    print(f"  mean_embedding shape: {mean_embedding.shape}")

    print("Loading splits …")
    emb_train, y_train, ids_train = store.load_split("train")
    emb_val,   y_val,   ids_val   = store.load_split("val")
    emb_test,  y_test,  ids_test  = store.load_split("test")
    print(f"  train={len(emb_train)}  val={len(emb_val)}  test={len(emb_test)}")

    print("\nRunning HPO + benchmark …")
    bench = run_benchmark(
        cfg,
        emb_train, y_train,
        emb_val,   y_val,
        emb_test,  y_test,
        ids_test,
    )

    # ── HPO detail ──────────────────────────────────────────────────────────
    print("\n=== HPO results (val ROC-AUC, sorted best→worst) ===")
    for head_name, all_results in bench["hpo_detail_per_head"].items():
        best_params = bench["hyperparameters_per_head"].get(head_name, {})
        _print_hpo_table(head_name, all_results, best_params, selection_metric)

    # ── Best params ──────────────────────────────────────────────────────────
    print("\n=== Best hyperparameters selected ===")
    for head_name, params in bench["hyperparameters_per_head"].items():
        params_str = "  ".join(f"{k}={v}" for k, v in params.items())
        print(f"  {head_name:30s}  {params_str}")

    # ── Test metrics ─────────────────────────────────────────────────────────
    print("\n=== Test-set metrics ===")
    header = f"  {'Head':<30s}  {'ROC-AUC':>8}  {'95% CI':>15}  {'PR-AUC':>8}  {'BAcc':>7}  {'Sens':>6}  {'Spec':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for head_name, m in bench["metrics_per_head"].items():
        ci = f"[{m['roc_auc_ci_lower']:.3f}, {m['roc_auc_ci_upper']:.3f}]"
        print(
            f"  {head_name:<30s}  {m['roc_auc']:>8.3f}  {ci:>15}  "
            f"{m['pr_auc']:>8.3f}  {m['balanced_accuracy']:>7.3f}  "
            f"{m['sensitivity']:>6.3f}  {m['specificity']:>6.3f}"
        )

    # ── LOO importance ───────────────────────────────────────────────────────
    print("\nComputing LOO slice importance …")
    importance_per_head: dict = {}
    for head_name, best_head in bench["best_heads_per_head"].items():
        print(f"  {head_name} …")
        imp = compute_importance_dataset(
            best_head,
            emb_test,
            ids_test,
            mean_embedding,
        )
        importance_per_head[head_name] = imp

    # ── Save ─────────────────────────────────────────────────────────────────
    sample_meta = {}
    import h5py
    with h5py.File(h5_path, "r") as f:
        for sid in ids_test:
            sample_meta[sid] = {"patient_id": str(f[sid].attrs.get("patient_id", ""))}

    print(f"\nSaving results to {results_dir}/ …")
    save_results(
        results_dir,
        metrics_per_head=bench["metrics_per_head"],
        predictions_per_head=bench["predictions_per_head"],
        hyperparameters_per_head=bench["hyperparameters_per_head"],
        importance_per_head=importance_per_head,
        config=cfg,
        sample_metadata=sample_meta,
    )
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiment.yaml")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    main(args.config, args.results)
