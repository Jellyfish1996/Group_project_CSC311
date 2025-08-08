"""
bagged_ensemble.py (no-kNN)
===========================

Bagged/blended ensemble over:
  • ALS (matrix_factorization.als)
  • Extended ALS with (theta, beta, bias) (matrix_factorization.als_with_modification)
  • IRT 1PL (item_response.irt)

The ensemble:
  1) Trains ALS, Extended ALS, and IRT across B different seeds (bagging via randomness).
  2) For each bag, blends models by validation-accuracy weights.
  3) Averages blended predictions across bags.

Plots:
  • Boxplot and scatter of per-bag validation/test accuracies (variance & generalization).
  • Cumulative mean test accuracy vs. number of bags (stability).
  • Average model weights bar chart (contribution).
  • Calibration (reliability) curve on validation data.
  • Generalization gap boxplot (val − test) per base model and the ensemble.

Assumptions:
  • User and item IDs are 0..(n_users-1) and 0..(n_items-1).
"""

from __future__ import annotations

import os
import json
from typing import Dict, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

# --- your modules ---
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)

from matrix_factorization import als, als_with_modification
from item_response import irt as irt_train


# -------------------------- small helpers --------------------------

def rng_seed(seed: int):
    """Set numpy RNG for reproducibility inside each bag."""
    np.random.seed(seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / (1.0 + np.exp(x))


def clip01(M: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(M, eps, 1.0 - eps)


def dataset_dims(*splits) -> Tuple[int, int]:
    n_users = max(max(s["user_id"]) for s in splits) + 1
    n_items = max(max(s["question_id"]) for s in splits) + 1
    return int(n_users), int(n_items)


def probs_from_irt(theta: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Dense probability matrix P[u, i] = sigmoid(theta_u - beta_i)."""
    return clip01(sigmoid(theta.reshape(-1, 1) - beta.reshape(1, -1)))


def blend_by_val_accuracy(preds: Dict[str, np.ndarray], val_data: dict, eps: float = 1e-9):
    """
    Compute accuracy on validation for each model and return:
      blended_matrix, acc_dict, weight_dict
    """
    names = list(preds.keys())
    accs = np.array([max(sparse_matrix_evaluate(val_data, preds[n]), eps) for n in names])
    weights = accs / accs.sum()
    blend = sum(w * preds[nm] for w, nm in zip(weights, names))
    return clip01(blend), dict(zip(names, accs)), dict(zip(names, weights))


def reliability_curve(pred_matrix: np.ndarray, data: dict, nbins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute (bin_centers, empirical_accuracy) on validation pairs contained in `data`.
    """
    u = np.array(data["user_id"], dtype=int)
    q = np.array(data["question_id"], dtype=int)
    y = np.array(data["is_correct"], dtype=int)
    p = pred_matrix[u, q]

    bins = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.digitize(p, bins) - 1
    accs, centers = [], []
    for b in range(nbins):
        mask = (idx == b)
        if mask.any():
            accs.append(float(y[mask].mean()))
            centers.append(0.5 * (bins[b] + bins[b + 1]))
    return np.array(centers), np.array(accs)


# -------------------------- core ensemble --------------------------

"""
bagged_ensemble.py (no-kNN) — corrected figures/plots
"""


import os
import json
from typing import Dict, Tuple, List

import numpy as np
import matplotlib as mpl

import matplotlib.pyplot as plt

from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)
from matrix_factorization import als, als_with_modification
from item_response import irt as irt_train


# -------------------------- small helpers --------------------------

def rng_seed(seed: int):
    np.random.seed(seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / (1.0 + np.exp(x))


def clip01(M: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return np.clip(M, eps, 1.0 - eps)


def dataset_dims(*splits) -> Tuple[int, int]:
    n_users = max(max(s["user_id"]) for s in splits) + 1
    n_items = max(max(s["question_id"]) for s in splits) + 1
    return int(n_users), int(n_items)


def probs_from_irt(theta: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return clip01(sigmoid(theta.reshape(-1, 1) - beta.reshape(1, -1)))


def blend_by_val_accuracy(preds: Dict[str, np.ndarray], val_data: dict, eps: float = 1e-9):
    """Return (blended_matrix, acc_dict, weight_dict) with accuracy-proportional weights."""
    names = list(preds.keys())
    accs = np.array([max(sparse_matrix_evaluate(val_data, preds[n]), eps) for n in names], dtype=float)
    weights = accs / accs.sum()
    blend = sum(w * preds[nm] for w, nm in zip(weights, names))
    return clip01(blend), dict(zip(names, accs.tolist())), dict(zip(names, weights.tolist()))


def reliability_curve(pred_matrix: np.ndarray, data: dict, nbins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """(bin_centers, empirical_accuracy) over validation points in `data`."""
    u = np.asarray(data["user_id"], dtype=int)
    q = np.asarray(data["question_id"], dtype=int)
    y = np.asarray(data["is_correct"], dtype=int)
    p = pred_matrix[u, q]

    bins = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.digitize(p, bins) - 1
    accs, centers = [], []
    for b in range(nbins):
        mask = (idx == b)
        if mask.any():
            accs.append(float(y[mask].mean()))
            centers.append(0.5 * (bins[b] + bins[b + 1]))
    return np.array(centers), np.array(accs)


# -------------------------- core ensemble --------------------------

def build_bagged_ensemble(
    B: int = 10,
    # ALS hyperparams
    als_k: int = 50, als_lr: float = 0.3, als_iters: int = 8000,
    # Extended ALS hyperparams
    ext_k: int = 50, ext_lr: float = 0.3, ext_iters: int = 8000,
    # IRT hyperparams
    irt_lr: float = 1e-3, irt_iters: int = 500,
    seed: int = 2025,
    save_prefix: str = "bagged",
    save_dir: str = "figs",
) -> Dict[str, object]:
    """
    Train a bagged/blended ensemble and produce plots & JSON summary.

    Returns dict:
      {'val_scores','test_scores','avg_weights','models_accs_per_bag','final_val','final_test'}
    """
    # --- load data ---
    train = load_train_csv("./data")
    val   = load_valid_csv("./data")
    test  = load_public_test_csv("./data")
    _ = load_train_sparse("./data").toarray()  # not used, but ensures consistency

    n_users, n_items = dataset_dims(train, val, test)

    # --- storage ---
    bag_val: List[float] = []
    bag_test: List[float] = []
    bag_weights: List[Dict[str, float]] = []
    bag_model_accs: List[Dict[str, float]] = []
    blended_val_running = None
    blended_test_running = None

    model_names = ["ALS", "ExtALS", "IRT"]
    model_val_accs = {m: [] for m in model_names}
    model_test_accs = {m: [] for m in model_names}

    # --- per-bag training ---
    for b in range(1, B + 1):
        rng_seed(seed + b)

        # ALS (plain)
        M_als = clip01(als(train, k=als_k, lr=als_lr, num_iteration=als_iters))

        # Extended ALS (theta,beta,bias)
        theta0 = np.zeros(n_users)
        beta0  = np.zeros(n_items)
        M_ext = clip01(
            als_with_modification(train, k=ext_k, lr=ext_lr, num_iteration=ext_iters,
                                  theta=theta0, beta=beta0)
        )

        # IRT (1PL)
        theta, beta, _val_acc_lst, *_ = irt_train(train, val, irt_lr, irt_iters)
        M_irt = probs_from_irt(theta, beta)

        preds = {"ALS": M_als, "ExtALS": M_ext, "IRT": M_irt}

        # Per-model accuracies on val/test
        for nm, mat in preds.items():
            model_val_accs[nm].append(float(sparse_matrix_evaluate(val, mat)))
            model_test_accs[nm].append(float(sparse_matrix_evaluate(test, mat)))

        # Accuracy-weighted blend (by validation)
        blend_val, accs_val, weights = blend_by_val_accuracy(preds, val)
        blend_test = sum(weights[name] * preds[name] for name in preds)

        v = float(sparse_matrix_evaluate(val, blend_val))
        t = float(sparse_matrix_evaluate(test, blend_test))

        bag_val.append(v)
        bag_test.append(t)
        bag_weights.append(weights)
        bag_model_accs.append(accs_val)

        # Running average of blended predictions for a final stable matrix
        if blended_val_running is None:
            blended_val_running = blend_val
            blended_test_running = blend_test
        else:
            alpha = 1.0 / b
            blended_val_running  = (1 - alpha) * blended_val_running  + alpha * blend_val
            blended_test_running = (1 - alpha) * blended_test_running + alpha * blend_test

        print(f"[Bag {b:02d}/{B}] val={v:.4f}  test={t:.4f}  weights={weights}")

    # Final blended evaluation
    final_val  = float(sparse_matrix_evaluate(val,  blended_val_running))
    final_test = float(sparse_matrix_evaluate(test, blended_test_running))

    # Average weights
    w_names = list(bag_weights[0].keys())
    avg_weights = {m: float(np.mean([w[m] for w in bag_weights])) for m in w_names}

    # Save summary + raw arrays (for exact reproducibility in the paper)
    summary = {
        "B": B,
        "val_scores": bag_val,
        "test_scores": bag_test,
        "avg_weights": avg_weights,
        "models_accs_per_bag": bag_model_accs,
        "final_val": final_val,
        "final_test": final_test,
        "als_params": {"k": als_k, "lr": als_lr, "iters": als_iters},
        "ext_params": {"k": ext_k, "lr": ext_lr, "iters": ext_iters},
        "irt_params": {"lr": irt_lr, "iters": irt_iters},
    }
    with open(f"{save_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- plots ---
    os.makedirs(save_dir, exist_ok=True)

    bag_val_np  = np.asarray(bag_val, dtype=float)
    bag_test_np = np.asarray(bag_test, dtype=float)

    # 1) Accuracy distribution per bag (validation & test)
    plt.figure(figsize=(6.5, 4.2))
    plt.boxplot([bag_val_np, bag_test_np], labels=["Val", "Test"])
    plt.ylabel("Accuracy")
    plt.title("Per-bag accuracy distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bag_accuracy_boxplot.png"), dpi=150)
    plt.close()

    # 1b) Generalization gap (val - test) per base model and ensemble
    plt.figure(figsize=(7.6, 4.6))
    labels = ["ALS", "ExtALS", "IRT", "Ensemble"]
    gap_series = [
        np.asarray(model_val_accs["ALS"])    - np.asarray(model_test_accs["ALS"]),
        np.asarray(model_val_accs["ExtALS"]) - np.asarray(model_test_accs["ExtALS"]),
        np.asarray(model_val_accs["IRT"])    - np.asarray(model_test_accs["IRT"]),
        bag_val_np - bag_test_np,
    ]
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.boxplot(gap_series, labels=labels, showfliers=True)
    plt.ylabel("Validation − Test accuracy")
    plt.title("Generalization gap by model (val − test)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_minus_test_gap_per_model.png"), dpi=150)
    plt.close()

    # 2) Scatter: Val vs Test per bag (generalization)
    plt.figure(figsize=(5.8, 5.2))
    plt.scatter(bag_val_np, bag_test_np, s=40)
    both = np.concatenate([bag_val_np, bag_test_np])
    lo, hi = float(both.min()) - 0.01, float(both.max()) + 0.01
    plt.plot([lo, hi], [lo, hi], "--", linewidth=1)
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.xlabel("Validation accuracy")
    plt.ylabel("Test accuracy")
    plt.title("Generalization: per-bag Val vs Test")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bag_val_vs_test_scatter.png"), dpi=150)
    plt.close()

    # 3) Cumulative mean test accuracy vs number of bags (stability)
    cum_mean = np.cumsum(bag_test_np) / (np.arange(B) + 1)
    plt.figure(figsize=(6.8, 4.2))
    plt.plot(np.arange(1, B + 1), cum_mean, marker="o")
    plt.xlabel("Number of bags")
    plt.ylabel("Cumulative mean test accuracy")
    plt.title("Stability of ensemble as bags increase")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cumulative_mean_test_accuracy.png"), dpi=150)
    plt.close()

    # 4) Average model weights
    plt.figure(figsize=(6.8, 4.2))
    names = list(avg_weights.keys())
    vals = [avg_weights[n] for n in names]
    plt.bar(names, vals)
    plt.ylabel("Average validation weight")
    plt.title("Average blend weights across bags")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "avg_model_weights.png"), dpi=150)
    plt.close()

    # 5) Calibration on validation (reliability curve)
    centers, accs = reliability_curve(blended_val_running, val, nbins=10)
    plt.figure(figsize=(5.8, 5.2))
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.plot(centers, accs, marker="o", label="Ensemble (val)")
    plt.xlabel("Predicted probability")
    plt.ylabel("Empirical accuracy")
    plt.title("Calibration (validation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "calibration_validation.png"), dpi=150)
    plt.close()

    print("\n=== Bagged Ensemble Summary ===")
    print(f"Val mean±std:  {bag_val_np.mean():.4f} ± {bag_val_np.std():.4f}")
    print(f"Test mean±std: {bag_test_np.mean():.4f} ± {bag_test_np.std():.4f}")
    print(f"Final blended  Val={final_val:.4f}  Test={final_test:.4f}")
    print("Average model weights:", avg_weights)
    print("Saved: " + os.path.abspath(save_dir) + "/* and JSON summary -> " + f"{save_prefix}_summary.json")

    return summary



# -------------------------- script entry --------------------------

if __name__ == "__main__":
    build_bagged_ensemble(
        B=50,
        als_k=28, als_lr=0.027, als_iters=140000,
        ext_k=22, ext_lr=0.031, ext_iters=140000,
        irt_lr=1e-3, irt_iters=500,
        seed=2025,
        save_prefix="bagged"
    )
