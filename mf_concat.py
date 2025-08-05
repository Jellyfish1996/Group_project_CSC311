"""mf_concat.py – Latent MF with feature‑concatenation (Approach B)
==================================================================
Implements
    \hat r_{uq} = σ( [p_u ; e_u]^⊤ [q_q ; f_q] )
where
  • p_u, q_q ∈ ℝ^k are standard user/item latent factors,
  • e_u  is a learned embedding of the **user features**  (gender, age‑bucket, premium),
  • f_q  is a learned embedding of the **item features**  (subject IDs – possibly multiple; we use the mean of their embeddings).

All parameters are trained together by mini‑batch SGD on binary cross‑entropy.
The implementation is pure‑NumPy to stay consistent with utils.py.
"""
import os, csv, ast, argparse, math, random, time
from typing import Dict, List
import numpy as np
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    sparse_matrix_evaluate,
    evaluate,
)

# ------------------------------------------------------------------
# Metadata loaders (re‑use logic from mf_bias.py)
# ------------------------------------------------------------------

def load_question_meta(root_dir="./data") -> Dict[int, List[int]]:
    path = os.path.join(root_dir, "question_meta.csv")
    out: Dict[int, List[int]] = {}
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            qid = int(row["question_id"])
            raw = row["subject_id"].strip()
            out[qid] = list(map(int, ast.literal_eval(raw))) if raw.startswith("[") else [int(raw)]
    return out


def load_student_meta(root_dir="./data"):
    """Return dicts: gender, age_bucket, premium."""
    path = os.path.join(root_dir, "student_meta.csv")
    gender, age_b, prem = {}, {}, {}
    with open(path, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            uid = int(row["user_id"])
            gender[uid] = int(row.get("gender", 0) or 0)  # 0,1,2
            dob = row.get("date_of_birth") or row.get("data_of_birth") or ""
            if len(dob) >= 4 and dob[:4].isdigit():
                age = 2025 - int(dob[:4])
            else:
                age = -1
            # bucket: <8,8‑10,11‑13,14‑16,17‑18,unknown(-1)) => 0‑5
            if age < 0:
                age_b[uid] = 5
            elif age < 8:
                age_b[uid] = 0
            elif age < 11:
                age_b[uid] = 1
            elif age < 14:
                age_b[uid] = 2
            elif age < 17:
                age_b[uid] = 3
            else:
                age_b[uid] = 4
            raw = row.get("premium_pupil", "")
            try:
                prem[uid] = int(float(raw)) if raw != "" else 0
            except ValueError:
                prem[uid] = 0
    return gender, age_b, prem

# ------------------------------------------------------------------
# Helper: feature index mapping
# ------------------------------------------------------------------
class FeatureIndex:
    """Assign a unique index to every categorical feature value."""
    def __init__(self, n_user, n_item, n_gender=3, n_age=6, n_premium=2, n_subject=250):
        self.off_user = 0
        self.off_item = self.off_user + n_user
        self.off_gender = self.off_item + n_item
        self.off_age = self.off_gender + n_gender
        self.off_prem = self.off_age + n_age
        self.off_subj = self.off_prem + n_premium
        self.n_total = self.off_subj + n_subject

    def user(self, u):
        return self.off_user + u
    def item(self, q):
        return self.off_item + q
    def gender(self, g):
        return self.off_gender + g
    def age(self, a):
        return self.off_age + a
    def prem(self, p):
        return self.off_prem + p
    def subj(self, s):
        return self.off_subj + s

# ------------------------------------------------------------------
# Factorisation model with concat
# ------------------------------------------------------------------
class FMConcat:
    def __init__(self, n_feat, k, rng):
        self.k = k
        self.W = rng.normal(scale=0.01, size=(n_feat, k))  # latent factors per feature
        self.linear = np.zeros(n_feat)                     # linear term (optional)
        self.bias = 0.0

    def predict(self, feats: List[int]):
        """Binary prob for one example given list of active feature indices."""
        vecs = self.W[feats]           # (m,k)
        sums = vecs.sum(axis=0)        # (k,)
        inter = 0.5*(sums @ sums - (vecs*vecs).sum())  # FM trick
        lin   = self.linear[feats].sum()
        logit = self.bias + lin + inter
        return 1/(1+np.exp(-logit))

    # SGD update for a single example
    def update(self, feats, y, lr, lam):
        vecs = self.W[feats]
        sums = vecs.sum(axis=0)
        inter = 0.5*(sums @ sums - (vecs*vecs).sum())
        lin   = self.linear[feats].sum()
        logit = self.bias + lin + inter
        p = 1/(1+np.exp(-logit))
        grad = (y - p)                  # BCE derivative w.r.t. logit (since y∈{0,1})
        # update bias & linear weights
        self.bias += lr*grad
        self.linear[feats] += lr*(grad - lam*self.linear[feats])
        # update latent factors (FM equation)
        for f in feats:
            v_f = self.W[f]
            self.W[f] += lr*(grad*(sums - v_f) - lam*v_f)

# ------------------------------------------------------------------
def build_feature_indices(train,q_meta,stu_meta):
    n_user = 1 + max(train["user_id"])
    n_item = 1 + max(train["question_id"])
    n_subj = 1 + max(s for subs in q_meta.values() for s in subs)
    return FeatureIndex(n_user,n_item,n_subject=n_subj)


def make_feat_list(u,q,fi,q_meta,stu_meta):
    feats=[fi.user(u),fi.item(q)]
    gender,ageb,prem=stu_meta
    feats.append(fi.gender(gender.get(u,0)))
    feats.append(fi.age(ageb.get(u,5)))
    feats.append(fi.prem(prem.get(u,0)))
    feats.extend([fi.subj(s) for s in q_meta.get(q,[ ])])
    return feats

# ------------------------------------------------------------------
# Training & evaluation
# ------------------------------------------------------------------

def train_one_epoch(model, data, fi, q_meta, stu_meta, lr, lam, batch, rng):
    idx = rng.permutation(len(data["user_id"]))
    for start in range(0,len(idx),batch):
        for j in idx[start:start+batch]:
            u = data["user_id"][j]
            q = data["question_id"][j]
            y = data["is_correct"][j]
            feats = make_feat_list(u,q,fi,q_meta,stu_meta)
            model.update(feats,y,lr,lam)


def evaluate_split(data, model, fi, q_meta, stu_meta):
    preds=[model.predict(make_feat_list(u,q,fi,q_meta,stu_meta))
           for u,q in zip(data["user_id"],data["question_id"])]
    return evaluate(data,preds)

# ------------------------------------------------------------------
# Main / CLI
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--patience", type=int, default=10)

    # comma-separated lists for a quick grid search
    ap.add_argument("--k_grid",   default="20,40,80")         # latent dims
    ap.add_argument("--lr_grid",  default="0.005,0.01,0.02")  # learning rates
    ap.add_argument("--lam_grid", default="1e-6,1e-5,1e-4")   # L2 penalties
    args = ap.parse_args()

    # ----- parse grids -------------------------------------------------
    k_grid   = [int(x)   for x in args.k_grid.split(",")   if x]
    lr_grid  = [float(x) for x in args.lr_grid.split(",")  if x]
    lam_grid = [float(x) for x in args.lam_grid.split(",") if x]

    # ----- load data & metadata once -----------------------------------
    train = load_train_csv(args.data)
    val   = load_valid_csv(args.data)
    test  = load_public_test_csv(args.data)
    q_meta   = load_question_meta(args.data)
    stu_meta = load_student_meta(args.data)
    fi = build_feature_indices(train, q_meta, stu_meta)

    # ----- grid search -------------------------------------------------
    rng_master = np.random.default_rng(0)
    best_overall = -1.0
    best_cfg     = None
    best_state   = None
    results      = []           # (val_acc, k, lr, lam)

    for k_best in k_grid:
        for lr in lr_grid:
            for lam in lam_grid:
                # fresh model & RNG for every run
                rng   = np.random.default_rng(rng_master.integers(1<<32))
                model = FMConcat(fi.n_total, k_best, rng)

                best_val = -1.0
                pat      = args.patience

                for ep in range(1, args.epochs + 1):
                    train_one_epoch(model, train, fi, q_meta, stu_meta,
                                    lr, lam, args.batch, rng)
                    val_acc = evaluate_split(val, model, fi, q_meta, stu_meta)

                    if val_acc > best_val + 1e-4:
                        best_val  = val_acc
                        best_ep   = ep
                        best_copy = (model.W.copy(),
                                     model.linear.copy(),
                                     model.bias)
                        pat = args.patience
                    else:
                        pat -= 1
                        if pat == 0:
                            break  # early stop

                # restore model to its per-run best
                model.W, model.linear, model.bias = best_copy
                results.append((best_val, k_best, lr, lam))

                if best_val > best_overall:
                    best_overall = best_val
                    best_cfg     = (k_best, lr, lam)
                    best_state   = best_copy

                print(f"[k={k_best:>3}, lr={lr:>6}, lam={lam:>7}] "
                      f"best-val={best_val:.4f} @ epoch {best_ep:02d}")

    # ----- evaluate the winner -----------------------------------------
    k_opt, lr_opt, lam_opt = best_cfg  # unpack for readability
    print("\n=== TOP CONFIGURATION ===")
    print(f"k={k_opt}, lr={lr_opt}, lam={lam_opt}, val_acc={best_overall:.4f}")

    # reload the best-overall weights
    model_best = FMConcat(fi.n_total, k_opt, np.random.default_rng(42))
    model_best.W, model_best.linear, model_best.bias = best_state

    test_acc = evaluate_split(test, model_best, fi, q_meta, stu_meta)
    print(f"Test accuracy = {test_acc:.4f}")

    # (Optional) show the whole leaderboard, sorted by validation score
    results.sort(key=lambda t: t[0], reverse=True)
    print("\n=== Leaderboard (top-5) ===")
    for rank, (val_acc, k_val, lr_val, lam_val) in enumerate(results[:5], 1):
        print(f"{rank:2d}. val={val_acc:.4f} | k={k_val:<3} lr={lr_val:<6} lam={lam_val}")

if __name__ == "__main__":
    main()