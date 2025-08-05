"""fm.py — Second‑order Factorisation Machine for CSC311 Eedi slice
================================================================
This is a **pure‑NumPy, 2‑way Factorisation Machine** that supports the extra
metadata provided in the project:

* user‑id               (one‑hot)
* question‑id           (one‑hot)
* gender                (0 = unknown, 1 = female, 2 = male)
* age bucket            (<8,8‑10,11‑13,14‑16,17‑18,unknown)
* premium‑pupil flag    (0/1)
* subject ids           (one‑hot for each tag; questions may have multiple)

The FM prediction follows Rendle (2010):

    ŷ(x) = w₀ + Σ w_j x_j  +  Σ_{j<k} ⟨v_j, v_k⟩ x_j x_k .

We train with mini‑batch SGD on binary cross‑entropy plus ℓ₂ regularisation.
A small grid‑search over (k, lr) is available via `--grid`.
"""

from __future__ import annotations
import os, csv, ast, argparse, random, math
from typing import Dict, List, Tuple
import numpy as np
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    evaluate,                  # accuracy helper uses flat preds
)

# ----------------------------- meta loaders ----------------------------- #

def load_question_meta(root: str = "./data") -> Dict[int, List[int]]:
    """Return mapping *question_id → [subject ids]* (handles multi‑subject)."""
    path = os.path.join(root, "question_meta.csv")
    meta: Dict[int, List[int]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            qid = int(row["question_id"])
            raw = row["subject_id"].strip()
            if raw.startswith("["):                      # list in string
                meta[qid] = list(map(int, ast.literal_eval(raw)))
            elif raw:
                meta[qid] = [int(raw)]
            else:
                meta[qid] = []
    return meta


def load_student_meta(root: str = "./data") -> Tuple[Dict[int,int], Dict[int,int], Dict[int,int]]:
    """Return three dicts: gender, age‑bucket, premium flag keyed by user_id."""
    path = os.path.join(root, "student_meta.csv")
    gender, age_bkt, premium = {}, {}, {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            u = int(row["user_id"])
            gender[u] = int(row.get("gender", 0) or 0)
            dob = row.get("date_of_birth") or row.get("data_of_birth") or ""
            if len(dob) >= 4 and dob[:4].isdigit():
                age = 2025 - int(dob[:4])
            else:
                age = -1
            # bucket index
            if age < 0:
                age_bkt[u] = 5          # unknown
            elif age < 8:
                age_bkt[u] = 0
            elif age < 11:
                age_bkt[u] = 1
            elif age < 14:
                age_bkt[u] = 2
            elif age < 17:
                age_bkt[u] = 3
            else:
                age_bkt[u] = 4
            raw_p = row.get("premium_pupil", "")
            try:
                premium[u] = int(float(raw_p)) if raw_p else 0
            except ValueError:
                premium[u] = 0
    return gender, age_bkt, premium

# --------------------------- feature indexing --------------------------- #

class FieldIndex:
    """Assign a contiguous slot range to each categorical field."""
    def __init__(self):
        self.offset: Dict[str, int] = {}
        self.dim: Dict[str, int] = {}
        self.size: int = 0

    def add(self, name: str, cardinality: int):
        self.offset[name] = self.size
        self.dim[name] = cardinality
        self.size += cardinality

    def index(self, name: str, value: int) -> int:
        return self.offset[name] + value


def build_feature_index(train, q_meta) -> FieldIndex:
    fi = FieldIndex()
    fi.add("uid", 1 + max(train["user_id"]))
    fi.add("iid", 1 + max(train["question_id"]))
    fi.add("gender", 3)
    fi.add("age", 6)
    fi.add("premium", 2)
    num_subj = 1 + max(s for subs in q_meta.values() for s in subs)
    fi.add("subject", num_subj)
    return fi

# --------------------------- FM model class ---------------------------- #

class FM:
    def __init__(self, num_feat: int, k: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w0 = 0.0
        self.w = np.zeros(num_feat)
        self.V = rng.normal(scale=0.1, size=(num_feat, k))

    def predict(self, idxs: List[int]) -> float:
        """Return sigmoid( score )."""
        # linear part
        lin = self.w0 + self.w[idxs].sum()
        # pairwise part via efficient trick
        V_sum = self.V[idxs].sum(axis=0)
        pair = 0.5 * ((V_sum**2).sum() - (self.V[idxs]**2).sum())
        score = lin + pair
        return 1.0 / (1.0 + math.exp(-score))

    def sgd_update(self, idxs: List[int], y: int, lr: float, lam: float):
        p = self.predict(idxs)
        grad = p - y  # derivative of BCE wrt score
        # update linear weights
        self.w0 -= lr * grad
        self.w[idxs] -= lr * (grad + lam * self.w[idxs])
        # update latent factors
        V_sum = self.V[idxs].sum(axis=0)
        for j in idxs:
            self.V[j] -= lr * (grad * (V_sum - self.V[j]) + lam * self.V[j])

# ------------------------- feature construction ------------------------ #

def build_feature_vector(u: int, q: int, fi: FieldIndex, q_meta, stu_meta) -> List[int]:
    g = stu_meta[0].get(u, 0)
    age = stu_meta[1].get(u, 5)
    prem = stu_meta[2].get(u, 0)
    idxs = [
        fi.index("uid", u),
        fi.index("iid", q),
        fi.index("gender", g),
        fi.index("age", age),
        fi.index("premium", prem),
    ]
    idxs.extend(fi.index("subject", s) for s in q_meta.get(q, []))
    return idxs

# ------------------------------ training -------------------------------- #

def train_fm(train, val, test, fi, q_meta, stu_meta, k=32, lr=0.05, lam=1e-4,
             epochs=20, batch=8192, seed: int = 0):
    rng = np.random.default_rng(seed)
    fm = FM(fi.size, k, seed)
    n = len(train["user_id"])

    def accuracy(split):
        preds = [fm.predict(build_feature_vector(u, q, fi, q_meta, stu_meta))
                  for u, q in zip(split["user_id"], split["question_id"])]
        return evaluate(split, preds)

    best_val, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        order = rng.permutation(n)
        # mini‑batch SGD
        for b in range(0, n, batch):
            for j in order[b: b + batch]:
                u = train["user_id"][j]
                q = train["question_id"][j]
                r = train["is_correct"][j]
                idxs = build_feature_vector(u, q, fi, q_meta, stu_meta)
                fm.sgd_update(idxs, r, lr, lam)
        if ep % 2 == 0:
            val_acc = accuracy(val)
            print(f"Epoch {ep:02d} val acc = {val_acc:.4f}")
            if val_acc > best_val + 1e-4:
                best_val, best_state = val_acc, (fm.w0, fm.w.copy(), fm.V.copy())

    # load best params
    if best_state is not None:
        fm.w0, fm.w, fm.V = best_state
    return fm, best_val, accuracy(test)

# ---------------------------- main / grid ------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./data")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--lam", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default= 2048)
    parser.add_argument("--grid", action="store_true", help="run small grid search over k, lr")
    args = parser.parse_args()

    train = load_train_csv(args.data)
    val   = load_valid_csv(args.data)
    test  = load_public_test_csv(args.data)
    q_meta = load_question_meta(args.data)
    stu_meta = load_student_meta(args.data)
    fi = build_feature_index(train, q_meta)

    if args.grid:
        # ---------- single run ------------------------------
        fm, vacc, tacc = train_fm(
            train, val, test, fi, q_meta, stu_meta,
            k=args.k,
            lr=args.lr,
            lam=args.lam,
            epochs=args.epochs,
            batch=args.batch,
        )
        print("Validation accuracy =", round(vacc, 4))
        print("Test accuracy        =", round(tacc, 4))
    else:
        # ---------- tiny grid search -------------------------
        grid_k   = [16, 32, 48, 64]
        grid_lr  = [0.05]
        grid_lam = [1e-4]
        best = {"val": -1.0}
        for k in grid_k:
            for lr in grid_lr:
                for lam in grid_lam:
                    print(f">>> k={k} lr={lr:.2f} lam={lam:.0e}")
                    _, vacc, tacc = train_fm(
                        train, val, test, fi, q_meta, stu_meta,
                        k=k, lr=lr, lam=lam,
                        epochs=args.epochs,
                        batch=args.batch,
                    )
                    print(f"val={vacc:.4f} test={tacc:.4f}")
                    if vacc > best["val"]:
                        best.update(dict(k=k, lr=lr, lam=lam, val=vacc, test=tacc))
        print("Best configuration")
        print("------------------")
        print(
            f"k={best['k']} lr={best['lr']:.2f} lam={best['lam']:.0e} "
            f"val={best['val']:.4f} test={best['test']:.4f}"
        )

if __name__ == "__main__":
    main()
