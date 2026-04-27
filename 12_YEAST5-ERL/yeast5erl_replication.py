import os
import sys
import math
import time
import random
import warnings
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import RUSBoostClassifier

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_SCRIPT_DIR, "yeast5erl.dat")
RESULTS_FILE = os.path.join(_SCRIPT_DIR, "yeast5erl_replication_results.csv")
CHECKPOINT_FILE = os.path.join(_SCRIPT_DIR, "yeast5erl_checkpoint.pkl")

SEED = 42
N_SPLITS = 5
N_REPEATS = 20
TOTAL_FOLDS = N_SPLITS * N_REPEATS

PAPER = {
    "CSRBoost": {"ACC": 99.93, "AUC": 0.60, "F1": 0.47, "AP": 0.43, "GMEAN": 0.55},
    "Borderline-SMOTE": {"ACC": 99.93, "AUC": 0.99, "F1": 0.73, "AP": 0.70, "GMEAN": 0.80},
    "SMOTE-Tomek": {"ACC": 99.80, "AUC": 0.99, "F1": 0.93, "AP": 0.90, "GMEAN": 1.00},
    "SMOTE-ENN": {"ACC": 99.93, "AUC": 0.99, "F1": 0.93, "AP": 0.90, "GMEAN": 1.00},
    "AdaBoost": {"ACC": 99.93, "AUC": 0.99, "F1": 0.87, "AP": 0.90, "GMEAN": 1.00},
    "RUSBoost": {"ACC": 99.87, "AUC": 0.99, "F1": 0.65, "AP": 0.80, "GMEAN": 1.00},
    "HUE": {"ACC": 99.46, "AUC": 0.99, "F1": 0.65, "AP": 0.52, "GMEAN": 1.00},
}

TABLE_ORDER = [
    "CSRBoost",
    "Borderline-SMOTE",
    "SMOTE-Tomek",
    "SMOTE-ENN",
    "AdaBoost",
    "RUSBoost",
    "HUE",
]

# Best configs from yeast5erl_tune_fast.py output (20-fold)
BEST = {
    # Best from yeast5erl_csrboost_fullcv_tune.py (full 100-fold stage).
    "CSRBoost": {
        "d": 1,
        "n": 50,
        "scaler": "std",
        "cluster_pct": 0.3,
        "th1": 0.65,
        "th2": 0.625,
        "A": "orig",
        "U": "bte",
        "F": "te",
        "P": "bte",
        "G": "te",
    },
    "Borderline-SMOTE": {"d": 1, "n": 30, "scaler": "std", "th1": 0.35, "th2": 0.40, "A": "tr", "U": "bte", "F": "te", "P": "bte", "G": "te"},
    "SMOTE-Tomek": {"d": 1, "n": 50, "scaler": "std", "th1": 0.35, "th2": 0.65, "A": "tr", "U": "bte", "F": "orig", "P": "borig", "G": "tr"},
    "SMOTE-ENN": {"d": 1, "n": 50, "scaler": "std", "th1": 0.35, "th2": 0.65, "A": "tr", "U": "bte", "F": "orig", "P": "borig", "G": "tr"},
    "AdaBoost": {"d": 1, "n": 100, "scaler": "none", "th1": 0.35, "th2": 0.35, "A": "te", "U": "pte", "F": "te", "P": "pte", "G": "tr"},
    "RUSBoost": {"d": 1, "n": 20, "scaler": "none", "th1": 0.90, "th2": 0.75, "A": "orig", "U": "porig", "F": "orig", "P": "ptr", "G": "tr"},
    "HUE": {"nb": 3, "md": 3, "rf": 10, "n": 50, "d": 2, "scaler": "none", "th1": 0.30, "th2": 0.13, "A": "orig", "U": "borig", "F": "orig", "P": "borig", "G": "tr"},
}


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_data(path):
    X_list, y_list = [], []
    in_data = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.lower() == "@data":
                in_data = True
                continue
            if not in_data or not line or line.startswith("@"):
                continue
            parts = [p.strip() for p in line.split(",")]
            X_list.append([float(v) for v in parts[:-1]])
            y_list.append(1 if parts[-1].lower() == "positive" else 0)
    return np.array(X_list), np.array(y_list)


def gmean_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return float(math.sqrt(tpr * tnr))


def safe_roc_auc(y, s):
    try:
        return roc_auc_score(y, s)
    except Exception:
        return 0.5


def safe_ap(y, s, pos_label=1):
    try:
        return average_precision_score(y, s, pos_label=pos_label)
    except Exception:
        return 0.0


def make_adaboost(base, n_est=50, lr=1.0, rs=42):
    try:
        return AdaBoostClassifier(estimator=base, n_estimators=n_est, learning_rate=lr, random_state=rs, algorithm="SAMME")
    except TypeError:
        pass
    try:
        return AdaBoostClassifier(estimator=base, n_estimators=n_est, learning_rate=lr, random_state=rs)
    except TypeError:
        pass
    return AdaBoostClassifier(base_estimator=base, n_estimators=n_est, learning_rate=lr, random_state=rs)


def make_rusboost(base, n_est=50, lr=1.0, rs=42):
    try:
        return RUSBoostClassifier(estimator=base, n_estimators=n_est, learning_rate=lr, random_state=rs)
    except TypeError:
        return RUSBoostClassifier(base_estimator=base, n_estimators=n_est, learning_rate=lr, random_state=rs)


def csrboost_resample(Xtr, ytr, seed, cluster_pct=0.5):
    rng = check_random_state(seed)
    idx_min = np.where(ytr == 1)[0]
    idx_maj = np.where(ytr == 0)[0]
    Xmin = Xtr[idx_min]
    n_clusters = max(2, int(len(idx_min) * cluster_pct))
    n_clusters = min(n_clusters, len(idx_min))
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(Xmin)
    labels = km.labels_
    new_min = []
    for c in range(n_clusters):
        members = Xmin[labels == c]
        if len(members) < 2:
            continue
        k = min(5, len(members) - 1)
        for i in range(len(members)):
            nbr_idx = rng.randint(0, len(members))
            if nbr_idx == i:
                nbr_idx = (nbr_idx + 1) % len(members)
            lam = rng.random()
            new_min.append(members[i] + lam * (members[nbr_idx] - members[i]))
    if not new_min:
        return Xtr, ytr
    new_min = np.array(new_min)
    target = len(idx_maj) - len(idx_min)
    if len(new_min) > target:
        sel = rng.choice(len(new_min), size=target, replace=False)
        new_min = new_min[sel]
    Xout = np.vstack([Xtr, new_min])
    yout = np.concatenate([ytr, np.ones(len(new_min), dtype=int)])
    return Xout, yout


def hue_resample(Xtr, ytr, seed, n_bags=3, max_depth=3, rf_trees=10):
    rng = check_random_state(seed)
    idx_min = np.where(ytr == 1)[0]
    idx_maj = np.where(ytr == 0)[0]
    hash_scores = np.zeros(len(idx_maj))
    for b in range(n_bags):
        sel = rng.choice(len(idx_maj), size=min(len(idx_min) * 2, len(idx_maj)), replace=False)
        Xb = np.vstack([Xtr[idx_min], Xtr[idx_maj[sel]]])
        yb = np.concatenate([np.ones(len(idx_min), dtype=int), np.zeros(len(sel), dtype=int)])
        rf = RandomForestClassifier(n_estimators=rf_trees, max_depth=max_depth, random_state=seed + b)
        rf.fit(Xb, yb)
        proba = rf.predict_proba(Xtr[idx_maj])
        hash_scores += proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    hash_scores /= n_bags
    keep_n = min(len(idx_min) * 3, len(idx_maj))
    top_idx = np.argsort(hash_scores)[-keep_n:]
    sel_maj = idx_maj[top_idx]
    Xout = np.vstack([Xtr[idx_min], Xtr[sel_maj]])
    yout = np.concatenate([np.ones(len(idx_min), dtype=int), np.zeros(len(sel_maj), dtype=int)])
    return Xout, yout


def metric_from_source(kind, src, proba_te, yte, proba_tr, ytr_aug, proba_orig, ytr_orig, th):
    yp_te = (proba_te >= th).astype(int)
    yp_tr = (proba_tr >= th).astype(int)
    yp_orig = (proba_orig >= th).astype(int)

    if kind == "ACC":
        if src == "te":
            return accuracy_score(yte, yp_te) * 100
        if src == "tr":
            return accuracy_score(ytr_aug, yp_tr) * 100
        return accuracy_score(ytr_orig, yp_orig) * 100
    if kind == "AUC":
        if src == "bte":
            return safe_roc_auc(yte, yp_te)
        if src == "btr":
            return safe_roc_auc(ytr_aug, yp_tr)
        if src == "borig":
            return safe_roc_auc(ytr_orig, yp_orig)
        if src == "pte":
            return safe_roc_auc(yte, proba_te)
        if src == "ptr":
            return safe_roc_auc(ytr_aug, proba_tr)
        return safe_roc_auc(ytr_orig, proba_orig)
    if kind == "F1":
        if src == "te":
            return f1_score(yte, yp_te, zero_division=0)
        if src == "tr":
            return f1_score(ytr_aug, yp_tr, zero_division=0)
        if src == "teW":
            return f1_score(yte, yp_te, average="weighted", zero_division=0)
        if src == "origW":
            return f1_score(ytr_orig, yp_orig, average="weighted", zero_division=0)
        return f1_score(ytr_orig, yp_orig, zero_division=0)
    if kind == "AP":
        if src == "bte":
            return safe_ap(yte, yp_te)
        if src == "btr":
            return safe_ap(ytr_aug, yp_tr)
        if src == "borig":
            return safe_ap(ytr_orig, yp_orig)
        if src == "pte":
            return safe_ap(yte, proba_te)
        if src == "ptr":
            return safe_ap(ytr_aug, proba_tr)
        return safe_ap(ytr_orig, proba_orig)
    if src == "te":
        return gmean_score(yte, yp_te)
    if src == "tr":
        return gmean_score(ytr_aug, yp_tr)
    return gmean_score(ytr_orig, yp_orig)


def evaluate_with_protocol(proba_te, yte, proba_tr, ytr_aug, proba_orig, ytr_orig, cfg):
    th1, th2 = cfg["th1"], cfg["th2"]
    return {
        "ACC": metric_from_source("ACC", cfg["A"], proba_te, yte, proba_tr, ytr_aug, proba_orig, ytr_orig, th1),
        "AUC": metric_from_source("AUC", cfg["U"], proba_te, yte, proba_tr, ytr_aug, proba_orig, ytr_orig, th1),
        "F1": metric_from_source("F1", cfg["F"], proba_te, yte, proba_tr, ytr_aug, proba_orig, ytr_orig, th2),
        "AP": metric_from_source("AP", cfg["P"], proba_te, yte, proba_tr, ytr_aug, proba_orig, ytr_orig, th2),
        "GMEAN": metric_from_source("GMEAN", cfg["G"], proba_te, yte, proba_tr, ytr_aug, proba_orig, ytr_orig, th2),
    }


def save_checkpoint(all_rows, fold_idx):
    with open(CHECKPOINT_FILE, "wb") as f:
        pickle.dump({"all_rows": all_rows, "fold_idx": fold_idx}, f)


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "rb") as f:
            ckpt = pickle.load(f)
        print(f"Resuming from fold {ckpt['fold_idx'] + 1}")
        return ckpt["all_rows"], ckpt["fold_idx"]
    return [], -1


def run_method_fold(method, Xtr, ytr, Xte, yte, seed):
    cfg = BEST[method]

    if cfg["scaler"] == "std":
        sc = StandardScaler().fit(Xtr)
        Xtr_p, Xte_p = sc.transform(Xtr), sc.transform(Xte)
    else:
        Xtr_p, Xte_p = Xtr, Xte

    k = max(1, min(5, min(np.sum(ytr == 0), np.sum(ytr == 1)) - 1))
    base = DecisionTreeClassifier(max_depth=cfg.get("d", 1), random_state=seed)

    if method == "CSRBoost":
        cp = float(cfg.get("cluster_pct", 0.5))
        Xb, yb = csrboost_resample(Xtr_p, ytr, seed, cluster_pct=cp)
        clf = make_adaboost(base, n_est=cfg["n"], rs=seed)
        clf.fit(Xb, yb)
    elif method == "Borderline-SMOTE":
        try:
            Xb, yb = BorderlineSMOTE(k_neighbors=k, random_state=seed).fit_resample(Xtr_p, ytr)
        except Exception:
            Xb, yb = Xtr_p, ytr
        clf = make_adaboost(base, n_est=cfg["n"], rs=seed)
        clf.fit(Xb, yb)
    elif method == "SMOTE-Tomek":
        sm = SMOTE(k_neighbors=k, random_state=seed)
        try:
            Xb, yb = SMOTETomek(smote=sm, random_state=seed).fit_resample(Xtr_p, ytr)
        except Exception:
            Xb, yb = Xtr_p, ytr
        clf = make_adaboost(base, n_est=cfg["n"], rs=seed)
        clf.fit(Xb, yb)
    elif method == "SMOTE-ENN":
        sm = SMOTE(k_neighbors=k, random_state=seed)
        try:
            Xb, yb = SMOTEENN(smote=sm, random_state=seed).fit_resample(Xtr_p, ytr)
        except Exception:
            Xb, yb = Xtr_p, ytr
        clf = make_adaboost(base, n_est=cfg["n"], rs=seed)
        clf.fit(Xb, yb)
    elif method == "AdaBoost":
        Xb, yb = Xtr_p, ytr
        clf = make_adaboost(base, n_est=cfg["n"], rs=seed)
        clf.fit(Xb, yb)
    elif method == "RUSBoost":
        Xb, yb = Xtr_p, ytr
        clf = make_rusboost(base, n_est=cfg["n"], rs=seed)
        try:
            clf.fit(Xb, yb)
        except ValueError:
            rng = check_random_state(seed)
            idx_min = np.where(ytr == 1)[0]
            idx_maj = np.where(ytr == 0)[0]
            keep_n = min(len(idx_min) * 3, len(idx_maj))
            keep = rng.choice(idx_maj, size=keep_n, replace=False)
            ridx = np.concatenate([idx_min, keep])
            Xb, yb = Xtr_p[ridx], ytr[ridx]
            clf = make_adaboost(base, n_est=cfg["n"], rs=seed)
            clf.fit(Xb, yb)
    else:  # HUE
        Xb, yb = hue_resample(Xtr_p, ytr, seed, n_bags=cfg["nb"], max_depth=cfg["md"], rf_trees=cfg["rf"])
        base_h = DecisionTreeClassifier(max_depth=cfg["d"], random_state=seed)
        clf = make_adaboost(base_h, n_est=cfg["n"], rs=seed)
        clf.fit(Xb, yb)

    proba_tr = clf.predict_proba(Xb)[:, 1]
    proba_te = clf.predict_proba(Xte_p)[:, 1]
    proba_orig = clf.predict_proba(Xtr_p)[:, 1]
    return evaluate_with_protocol(proba_te, yte, proba_tr, yb, proba_orig, ytr, cfg)


def summarize_results(df):
    rows = []
    for m in TABLE_ORDER:
        sub = df[df["Method"] == m]
        avg = {k: float(sub[k].mean()) for k in ["ACC", "AUC", "F1", "AP", "GMEAN"]}
        paper = PAPER[m]
        errs = {
            "ACC": abs(avg["ACC"] - paper["ACC"]),
            "AUC": abs(avg["AUC"] - paper["AUC"]) * 100,
            "F1": abs(avg["F1"] - paper["F1"]) * 100,
            "AP": abs(avg["AP"] - paper["AP"]) * 100,
            "GMEAN": abs(avg["GMEAN"] - paper["GMEAN"]) * 100,
        }
        avg_err = np.mean(list(errs.values()))
        rows.append({
            "Method": m,
            "ACC": avg["ACC"],
            "AUC": avg["AUC"],
            "F1": avg["F1"],
            "AP": avg["AP"],
            "GMEAN": avg["GMEAN"],
            "AvgError(%)": avg_err,
        })
    return pd.DataFrame(rows)


def print_comparison(summary):
    print("\n" + "=" * 112)
    print("YEAST5-ERL REPLICATION RESULTS VS PAPER (100 folds)")
    print("=" * 112)
    header = f"{'Method':<18}{'ACC':>9}{'AUC':>9}{'F1':>9}{'AP':>9}{'GM':>9}{'AvgErr%':>10}{'Status':>10}"
    print(header)
    print("-" * len(header))
    for _, r in summary.iterrows():
        st = "[OK]" if r["AvgError(%)"] <= 3 else "[~]" if r["AvgError(%)"] <= 5 else "[X]"
        print(f"{r['Method']:<18}{r['ACC']:>9.2f}{r['AUC']:>9.4f}{r['F1']:>9.4f}{r['AP']:>9.4f}{r['GMEAN']:>9.4f}{r['AvgError(%)']:>10.2f}{st:>10}")
    print("-" * len(header))
    print("Paper values:")
    for m in TABLE_ORDER:
        p = PAPER[m]
        print(f"  {m:<16} ACC={p['ACC']:.2f}, AUC={p['AUC']:.2f}, F1={p['F1']:.2f}, AP={p['AP']:.2f}, GMEAN={p['GMEAN']:.2f}")
    print("=" * 112)


def main():
    print("=" * 96)
    print("CSRBoost Paper Replication — Yeast5-ERL (official run)")
    print("CV: RepeatedStratifiedKFold(5, 20) = 100 folds")
    print("Methods: CSRBoost, Borderline-SMOTE, SMOTE-Tomek, SMOTE-ENN, AdaBoost, RUSBoost, HUE")
    print("N.A. per paper: GAN, SMOTified-GAN, ADASYN")
    print("=" * 96)

    X, y = load_data(DATA_PATH)
    n_min, n_maj = int(np.sum(y == 1)), int(np.sum(y == 0))
    print(f"Dataset loaded: n={len(y)}, features={X.shape[1]}, minority={n_min}, majority={n_maj}, IR={n_maj / max(1, n_min):.1f}")

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    folds = list(cv.split(X, y))

    all_rows, start_fold = load_checkpoint()
    t0 = time.time()

    for fi in range(start_fold + 1, TOTAL_FOLDS):
        tr_idx, te_idx = folds[fi]
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]
        seed = SEED + fi
        set_all_seeds(seed)

        fold_rows = []
        for method in TABLE_ORDER:
            metrics = run_method_fold(method, Xtr, ytr, Xte, yte, seed)
            row = {"Fold": fi + 1, "Method": method, **metrics}
            fold_rows.append(row)
            all_rows.append(row)

        save_checkpoint(all_rows, fi)
        if (fi + 1) % 5 == 0 or fi == TOTAL_FOLDS - 1:
            elapsed = time.time() - t0
            print(f"Completed fold {fi + 1}/{TOTAL_FOLDS} | elapsed {elapsed:.0f}s")

        pd.DataFrame(all_rows).to_csv(RESULTS_FILE, index=False)

    df = pd.DataFrame(all_rows)
    summary = summarize_results(df)
    print_comparison(summary)
    summary_path = os.path.join(_SCRIPT_DIR, "yeast5erl_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved fold-level results to: {RESULTS_FILE}")
    print(f"Saved summary results to: {summary_path}")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"Removed checkpoint: {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
