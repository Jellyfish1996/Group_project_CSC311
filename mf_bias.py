"""mf_bias.py – Matrix Factorisation with bias terms (multi‑subject aware) + Hyper‑parameter grid search
==============================================================================================
This standalone script
  • handles questions tagged with **multiple subjects**,
  • tolerates both `date_of_birth` and the typo `data_of_birth`, and
  • runs an exhaustive grid‑search over (k, lr, λ, batch_size) without any PyTorch
    dependency – everything is NumPy only.

It prints training progress every 2 epochs and finally reports the best
configuration.
"""

import ast, csv, os, argparse
from typing import Dict, List
import numpy as np
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    sparse_matrix_evaluate,
)

# ------------------------- metadata loaders ------------------------- #

def load_question_meta(root_dir: str = "./data") -> Dict[int, List[int]]:
    """Map question_id → list[int] subject IDs."""
    path = os.path.join(root_dir, "question_meta.csv")
    out: Dict[int, List[int]] = {}
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            qid = int(row["question_id"])
            raw = row["subject_id"].strip()
            subjs = list(map(int, ast.literal_eval(raw))) if raw.startswith("[") else [int(raw)]
            out[qid] = subjs
    return out


def load_student_meta(root_dir: str = "./data"):
    """Return dicts gender_by_u, age_by_u, premium_by_u."""
    path = os.path.join(root_dir, "student_meta.csv")
    g_by_u, age_by_u, prem_by_u = {}, {}, {}
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            uid = int(row["user_id"])
            g_by_u[uid] = int(row.get("gender", 0) or 0)
            dob = row.get("date_of_birth") or row.get("data_of_birth") or ""
            age_by_u[uid] = 2025 - int(dob[:4]) if dob[:4].isdigit() else -1
            try:
                prem_by_u[uid] = int(float(row.get("premium_pupil", 0) or 0))
            except ValueError:
                prem_by_u[uid] = 0
    return g_by_u, age_by_u, prem_by_u

# ------------------------ model helper ------------------------------ #

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

# ---------------------- training function --------------------------- #

def train_mf_bias(train, val, q_meta, stu_meta, *, k, lr, lam, epochs, batch):
    """Return full prob‑matrix and validation accuracy for given hyper‑params."""
    num_u = 1 + max(train["user_id"])
    num_q = 1 + max(train["question_id"])
    num_g = 3  # 0,1,2
    num_s = 1 + max(s for subs in q_meta.values() for s in subs)

    rng = np.random.default_rng(0)
    P = rng.normal(scale=0.1, size=(num_u, k))
    Q = rng.normal(scale=0.1, size=(num_q, k))
    b_u = np.zeros(num_u)
    d_q = np.zeros(num_q)
    g_vec = np.zeros(num_g)
    s_vec = np.zeros(num_s)
    mu = np.mean(train["is_correct"])

    u_ids = np.array(train["user_id"])
    q_ids = np.array(train["question_id"])
    r_vals = np.array(train["is_correct"])

    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(r_vals))
        for start in range(0, len(idx), batch):
            batch_idx = idx[start:start + batch]
            for u, q, r in zip(u_ids[batch_idx], q_ids[batch_idx], r_vals[batch_idx]):
                gender = stu_meta[0].get(u, 0)
                subjs = q_meta[q]
                subj_bias = s_vec[subjs].mean()
                pred = sigmoid(mu + b_u[u] + d_q[q] + g_vec[gender] + subj_bias + P[u] @ Q[q])
                err = r - pred
                # scalar updates
                b_u[u] += lr * (err - lam * b_u[u])
                d_q[q] += lr * (err - lam * d_q[q])
                g_vec[gender] += lr * (err - lam * g_vec[gender])
                share = lr * err / len(subjs)
                s_vec[subjs] += share - lr * lam * s_vec[subjs]
                # latent factors
                P_u_copy = P[u].copy()
                P[u] += lr * (err * Q[q] - lam * P[u])
                Q[q] += lr * (err * P_u_copy - lam * Q[q])
        if ep % 2 == 0:
            val_acc = sparse_matrix_evaluate(val, build_full_matrix(P, Q, mu, b_u, d_q, g_vec, s_vec, q_meta, stu_meta))
            print(f"Epoch {ep:02d} val acc = {val_acc:.4f}")
    full_mat = build_full_matrix(P, Q, mu, b_u, d_q, g_vec, s_vec, q_meta, stu_meta)
    val_acc = sparse_matrix_evaluate(val, full_mat)
    return val_acc, full_mat


def build_full_matrix(P, Q, mu, b_u, d_q, g_vec, s_vec, q_meta, stu_meta):
    num_u, num_q = P.shape[0], Q.shape[0]
    mat = np.empty((num_u, num_q), dtype=np.float32)
    for u in range(num_u):
        g = stu_meta[0].get(u, 0)
        row_core = mu + b_u[u] + g_vec[g] + d_q + P[u] @ Q.T  # broadcast
        # subject bias per column (mean over tags)
        subj_bias = np.array([s_vec[q_meta[q]].mean() for q in range(num_q)])
        mat[u] = sigmoid(row_core + subj_bias)
    return mat

# ------------------------ grid search ------------------------------- #

def grid_search(data_dir="./data"):
    train = load_train_csv(data_dir)
    val   = load_valid_csv(data_dir)
    test  = load_public_test_csv(data_dir)
    q_meta = load_question_meta(data_dir)
    stu_meta = load_student_meta(data_dir)

    grid = {
        "k":          [16, 32, 48],
        "lr":         [0.02, 0.01],
        "lam":        [1e-4, 3e-4, 1e-3],
        "batch_size": [4096, 8192, 16384],
    }
    best = {"acc": -1.0}
    for k in grid["k"]:
        for lr in grid["lr"]:
            for lam in grid["lam"]:
                for bs in grid["batch_size"]:
                    print(f"→ k={k} lr={lr} λ={lam} bs={bs}")
                    val_acc, full_mat = train_mf_bias(train, val, q_meta, stu_meta,
                                                        k=k, lr=lr, lam=lam,
                                                        epochs=10, batch=bs)
                    print(f"   val={val_acc:.4f}")
                    if val_acc > best["acc"]:
                        best.update(dict(acc=val_acc, k=k, lr=lr, lam=lam, batch=bs,
                                          test=sparse_matrix_evaluate(test, full_mat)))
    print("\nBest config:", best)


if __name__ == "__main__":
    grid_search()
