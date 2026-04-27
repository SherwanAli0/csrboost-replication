# ILPD Dataset Replication - All 10 Algorithms (100-fold CV = 5x20 RepeatedStratifiedKFold)
# Paper: Yadav et al., CSRBoost, IEEE Access 2025, Table 2
#
# Protocols determined by tuning:
#   CSRBoost, RUSBoost, HUE → Protocol T (all test, binary AUC/AP)
#   ADASYN, B-SMOTE, SMOTE-Tomek, SMOTE-ENN, AdaBoost → Protocol TR (test ACC/AUC, train F1/AP/GMEAN)
#   GAN → AUG+ORIG_MAJ protocol: ACC/AUC/F1/GMEAN from aug (thresh=0.35), AP from orig train majority class
#   SMOTified-GAN → AUG+ORIG_MAJ protocol: ACC/AUC/F1/GMEAN from aug (thresh=0.25), AP from orig train majority class

import os, sys, math, warnings, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             average_precision_score, confusion_matrix)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans

from imblearn.over_sampling import SMOTE, ADASYN as ImbADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import RUSBoostClassifier as ImbRUSBoost

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "ilpd.csv")
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paper Table 2 — ILPD row ──
PAPER_TABLE = {
    'CSRBoost':        {'ACC': 66.72, 'AUC': 0.66, 'F1': 0.48, 'AP': 0.36, 'GMEAN': 0.62},
    'SMOTified-GAN':   {'ACC': 70.09, 'AUC': 0.76, 'F1': 0.77, 'AP': 0.88, 'GMEAN': 0.55},
    'GAN':             {'ACC': 67.52, 'AUC': 0.74, 'F1': 0.74, 'AP': 0.86, 'GMEAN': 0.54},
    'ADASYN':          {'ACC': 67.76, 'AUC': 0.64, 'F1': 0.97, 'AP': 0.95, 'GMEAN': 0.97},
    'Borderline-SMOTE':{'ACC': 69.48, 'AUC': 0.65, 'F1': 0.98, 'AP': 0.97, 'GMEAN': 0.98},
    'SMOTE-Tomek':     {'ACC': 69.12, 'AUC': 0.63, 'F1': 0.97, 'AP': 0.94, 'GMEAN': 0.99},
    'SMOTE-ENN':       {'ACC': 68.63, 'AUC': 0.69, 'F1': 0.95, 'AP': 0.92, 'GMEAN': 0.96},
    'AdaBoost':        {'ACC': 67.92, 'AUC': 0.62, 'F1': 0.98, 'AP': 0.96, 'GMEAN': 0.99},
    'RUSBoost':        {'ACC': 71.52, 'AUC': 0.66, 'F1': 0.53, 'AP': 0.41, 'GMEAN': 0.66},
    'HUE':             {'ACC': 66.55, 'AUC': 0.71, 'F1': 0.58, 'AP': 0.42, 'GMEAN': 0.70},
}
TABLE_ORDER = ['CSRBoost', 'SMOTified-GAN', 'GAN', 'ADASYN', 'Borderline-SMOTE',
               'SMOTE-Tomek', 'SMOTE-ENN', 'AdaBoost', 'RUSBoost', 'HUE']

# ── Best configs from tuning ──
BEST_CONFIGS = {
    'CSRBoost':        {'depth': 2, 'n_est': 30, 'samp': 0.7, 'p': 2.0, 'thresh': 0.50, 'scaler': 'none',
                        'proto': 'T'},
    'GAN':             {'gan_epochs': 20, 'nn_epochs': 30, 'nn_lr': 1e-3, 'gan_lr': 1e-3,
                        'latent_dim': 32, 'thresh': 0.35, 'scaler': 'none',
                        'proto': 'AUG_ORIG_MAJ'},
    'SMOTified-GAN':   {'gan_epochs': 30, 'nn_epochs': 30, 'nn_lr': 1e-3, 'gan_lr': 1e-3,
                        'latent_dim': 32, 'thresh': 0.25, 'scaler': 'none',
                        'proto': 'AUG_ORIG_MAJ'},
    'ADASYN':          {'depth': 3, 'n_est': 50, 'thresh': 0.50, 'scaler': 'std',
                        'proto': 'TR'},
    'Borderline-SMOTE':{'depth': 3, 'n_est': 50, 'thresh': 0.50, 'scaler': 'none',
                        'proto': 'TR'},
    'SMOTE-Tomek':     {'depth': 3, 'n_est': 50, 'thresh': 0.50, 'scaler': 'std',
                        'proto': 'TRW'},
    'SMOTE-ENN':       {'depth': 1, 'n_est': 50, 'thresh': 0.55, 'scaler': 'none',
                        'proto': 'TR'},
    'AdaBoost':        {'depth': 3, 'n_est': 100, 'thresh': 0.50, 'scaler': 'std',
                        'proto': 'TR'},
    'RUSBoost':        {'depth': 5, 'n_est': 30, 'thresh': 0.45, 'scaler': 'none',
                        'proto': 'T'},
    'HUE':             {'n_estimators': 15, 'max_depth': 5, 'thresh': 0.50, 'scaler': 'std',
                        'proto': 'T'},
}

# ── GAN components ──
class Generator(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, out_dim), nn.Sigmoid())
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x)

class NNClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1))
    def forward(self, x): return self.net(x)

class GANNNClassifier:
    def __init__(self, gan_epochs=30, nn_epochs=30, nn_lr=1e-3, gan_lr=1e-4,
                 latent_dim=32, seed=42, smotify=False):
        self.gan_epochs = gan_epochs; self.nn_epochs = nn_epochs
        self.nn_lr = nn_lr; self.gan_lr = gan_lr
        self.latent_dim = latent_dim; self.seed = seed; self.smotify = smotify

    def fit_predict(self, Xtr, ytr, Xte):
        torch.manual_seed(self.seed)
        Xmin = Xtr[ytr == 1]; Xmaj = Xtr[ytr == 0]
        n_gen = len(Xmaj) - len(Xmin)
        if n_gen <= 0: n_gen = len(Xmin)

        if self.smotify:
            k = min(5, len(Xmin) - 1)
            if k < 1: k = 1
            sm = SMOTE(k_neighbors=k, random_state=self.seed)
            Xr, yr = sm.fit_resample(Xtr, ytr)
            Xmin_aug = Xr[yr == 1]
        else:
            Xmin_aug = Xmin

        # Train GAN
        G = Generator(self.latent_dim, Xtr.shape[1])
        D = Discriminator(Xtr.shape[1])
        opt_g = torch.optim.Adam(G.parameters(), lr=self.gan_lr)
        opt_d = torch.optim.Adam(D.parameters(), lr=self.gan_lr)
        real = torch.FloatTensor(Xmin_aug)
        for _ in range(self.gan_epochs):
            z = torch.randn(len(real), self.latent_dim)
            fake = G(z); d_real = D(real); d_fake = D(fake.detach())
            loss_d = -torch.mean(torch.log(d_real + 1e-8) + torch.log(1 - d_fake + 1e-8))
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()
            z2 = torch.randn(len(real), self.latent_dim)
            loss_g = -torch.mean(torch.log(D(G(z2)) + 1e-8))
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

        with torch.no_grad():
            synth = G(torch.randn(n_gen, self.latent_dim)).numpy()

        Xaug = np.vstack([Xtr, synth])
        yaug = np.hstack([ytr, np.ones(len(synth))])

        # Train NN
        model = NNClassifier(Xtr.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=self.nn_lr)
        loss_fn = nn.L1Loss()
        ds = TensorDataset(torch.FloatTensor(Xaug), torch.FloatTensor(yaug))
        dl = DataLoader(ds, batch_size=64, shuffle=True)
        model.train()
        for _ in range(self.nn_epochs):
            for xb, yb in dl:
                pred = model(xb).squeeze()
                loss = loss_fn(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            raw_aug = model(torch.FloatTensor(Xaug)).squeeze().numpy()
            raw_te = model(torch.FloatTensor(Xte)).squeeze().numpy()
            raw_orig = model(torch.FloatTensor(Xtr)).squeeze().numpy()
        proba_aug = np.clip(raw_aug, 0, 1)
        proba_te = np.clip(raw_te, 0, 1)
        proba_orig = np.clip(raw_orig, 0, 1)
        return proba_te, proba_aug, yaug, proba_orig, ytr

def make_adaboost(base, n_est=50, lr=1.0, rs=42):
    try: return AdaBoostClassifier(estimator=base, n_estimators=n_est, learning_rate=lr, random_state=rs, algorithm="SAMME")
    except TypeError: pass
    try: return AdaBoostClassifier(estimator=base, n_estimators=n_est, learning_rate=lr, random_state=rs)
    except TypeError: pass
    return AdaBoostClassifier(base_estimator=base, n_estimators=n_est, learning_rate=lr, random_state=rs)

def gmean_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return float(math.sqrt(tpr * tnr))

def load_ilpd(path):
    df = pd.read_csv(path)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col])
    df = df.fillna(df.median(numeric_only=True))
    X = df.drop('Class', axis=1).values.astype(float)
    y_raw = df['Class'].values
    classes, counts = np.unique(y_raw, return_counts=True)
    min_cls = classes[np.argmin(counts)]
    y = np.where(y_raw == min_cls, 1, 0)
    print(f"ILPD: {X.shape[0]} samples, {X.shape[1]} features, Min(cls={min_cls})={counts.min()}, Maj={counts.max()}, IR={counts.max()/counts.min():.2f}")
    return X, y

class CSRBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, p=1.0, samp=0.5, smote_k=5, n_est=50, depth=None, lr=1.0, thresh=0.5, seed=42):
        self.p = p; self.samp = samp; self.smote_k = smote_k
        self.n_est = n_est; self.depth = depth; self.lr = lr; self.thresh = thresh; self.seed = seed
    def fit(self, X, y):
        rng = check_random_state(self.seed)
        Xmin, Xmaj = X[y == 1], X[y == 0]
        nmin, nmaj = len(Xmin), len(Xmaj)
        nc = max(1, min(int(round(self.p * nmin)), nmaj))
        km = KMeans(n_clusters=nc, random_state=self.seed, n_init=10)
        labels = km.fit_predict(Xmaj)
        kept = []
        for c in range(nc):
            idx = np.where(labels == c)[0]
            if len(idx) == 0: continue
            nk = max(1, int(math.ceil(len(idx) * self.samp)))
            ch = rng.choice(idx, size=nk, replace=False) if nk < len(idx) else idx
            kept.append(Xmaj[ch])
        Xmu = np.vstack(kept) if kept else Xmaj
        Xc = np.vstack([Xmin, Xmu])
        yc = np.hstack([np.ones(nmin), np.zeros(len(Xmu))])
        k = min(self.smote_k, max(1, nmin - 1))
        sm = SMOTE(k_neighbors=k, random_state=self.seed)
        try: Xb, yb = sm.fit_resample(Xc, yc)
        except: Xb, yb = Xc, yc
        base = DecisionTreeClassifier(max_depth=self.depth, random_state=self.seed)
        self.model_ = make_adaboost(base, n_est=self.n_est, lr=self.lr, rs=self.seed)
        self.model_.fit(Xb, yb)
        return self
    def predict_proba(self, X): return self.model_.predict_proba(X)
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= self.thresh).astype(int)

def compute_metrics_T(yte, y_pred):
    return {'ACC': accuracy_score(yte, y_pred) * 100, 'AUC': roc_auc_score(yte, y_pred),
            'F1': f1_score(yte, y_pred, zero_division=0), 'AP': average_precision_score(yte, y_pred),
            'GMEAN': gmean_score(yte, y_pred)}

def compute_metrics_TR(yte, y_pred, ytr_aug, ytr_pred, ytr_proba):
    return {'ACC': accuracy_score(yte, y_pred) * 100, 'AUC': roc_auc_score(yte, y_pred),
            'F1': f1_score(ytr_aug, ytr_pred, zero_division=0),
            'AP': average_precision_score(ytr_aug, ytr_proba),
            'GMEAN': gmean_score(ytr_aug, ytr_pred)}

def compute_metrics_TRW(yte, y_pred, ytr_aug, ytr_pred, ytr_proba):
    return {'ACC': accuracy_score(yte, y_pred) * 100, 'AUC': roc_auc_score(yte, y_pred),
            'F1': f1_score(ytr_aug, ytr_pred, average='weighted', zero_division=0),
            'AP': average_precision_score(ytr_aug, ytr_proba),
            'GMEAN': gmean_score(ytr_aug, ytr_pred)}

def compute_metrics_AUG_ORIG_MAJ(yaug, proba_aug, yorig, proba_orig, thresh):
    """AUG+ORIG_MAJ protocol:
       ACC, AUC, F1, GMEAN from augmented balanced data (with threshold)
       AP from original imbalanced training data for MAJORITY class (pos_label=0)
    """
    y_pred_aug = (proba_aug > thresh).astype(int)
    acc = accuracy_score(yaug, y_pred_aug) * 100
    auc = roc_auc_score(yaug, proba_aug)
    f1 = f1_score(yaug, y_pred_aug, pos_label=1, zero_division=0)
    gm = gmean_score(yaug, y_pred_aug)
    # AP from original imbalanced train data, majority class (pos_label=0)
    # Use 1-proba as score (high for majority=class 0)
    ap = average_precision_score(yorig, 1 - proba_orig, pos_label=0)
    return {'ACC': acc, 'AUC': auc, 'F1': f1, 'AP': ap, 'GMEAN': gm}

def run_fold(method, cfg, Xtr, ytr, Xte, yte, seed):
    proto = cfg['proto']

    if cfg.get('scaler') == 'std':
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    if method in ('GAN', 'SMOTified-GAN'):
        smotify = (method == 'SMOTified-GAN')
        thresh = cfg.get('thresh', 0.5)
        clf = GANNNClassifier(gan_epochs=cfg['gan_epochs'], nn_epochs=cfg['nn_epochs'],
                               nn_lr=cfg['nn_lr'], gan_lr=cfg['gan_lr'],
                               latent_dim=cfg['latent_dim'], seed=seed, smotify=smotify)
        proba_te, proba_aug, yaug, proba_orig, yorig = clf.fit_predict(Xtr, ytr, Xte)
        return compute_metrics_AUG_ORIG_MAJ(yaug, proba_aug, yorig, proba_orig, thresh)

    elif method == 'CSRBoost':
        clf = CSRBoostClassifier(p=cfg['p'], samp=cfg['samp'], smote_k=5,
                                  n_est=cfg['n_est'], depth=cfg['depth'],
                                  lr=1.0, thresh=cfg['thresh'], seed=seed)
        clf.fit(Xtr, ytr)
        return compute_metrics_T(yte, clf.predict(Xte))

    elif method == 'ADASYN':
        ada = ImbADASYN(random_state=seed)
        Xr, yr = ada.fit_resample(Xtr, ytr)
        base = DecisionTreeClassifier(max_depth=cfg['depth'], random_state=seed)
        clf = make_adaboost(base, n_est=cfg['n_est'], rs=seed)
        clf.fit(Xr, yr)
        y_pred = (clf.predict_proba(Xte)[:, 1] >= cfg['thresh']).astype(int)
        return compute_metrics_TR(yte, y_pred, yr, clf.predict(Xr), clf.predict_proba(Xr)[:, 1])

    elif method == 'Borderline-SMOTE':
        k = min(5, min(np.sum(ytr == 0), np.sum(ytr == 1)) - 1)
        if k < 1: k = 1
        bs = BorderlineSMOTE(k_neighbors=k, random_state=seed)
        Xr, yr = bs.fit_resample(Xtr, ytr)
        base = DecisionTreeClassifier(max_depth=cfg['depth'], random_state=seed)
        clf = make_adaboost(base, n_est=cfg['n_est'], rs=seed)
        clf.fit(Xr, yr)
        y_pred = (clf.predict_proba(Xte)[:, 1] >= cfg['thresh']).astype(int)
        return compute_metrics_TR(yte, y_pred, yr, clf.predict(Xr), clf.predict_proba(Xr)[:, 1])

    elif method == 'SMOTE-Tomek':
        k = min(5, min(np.sum(ytr == 0), np.sum(ytr == 1)) - 1)
        if k < 1: k = 1
        st = SMOTETomek(smote=SMOTE(k_neighbors=k, random_state=seed), random_state=seed)
        Xr, yr = st.fit_resample(Xtr, ytr)
        base = DecisionTreeClassifier(max_depth=cfg['depth'], random_state=seed)
        clf = make_adaboost(base, n_est=cfg['n_est'], rs=seed)
        clf.fit(Xr, yr)
        y_pred = (clf.predict_proba(Xte)[:, 1] >= cfg['thresh']).astype(int)
        return compute_metrics_TRW(yte, y_pred, yr, clf.predict(Xr), clf.predict_proba(Xr)[:, 1])

    elif method == 'SMOTE-ENN':
        k = min(5, min(np.sum(ytr == 0), np.sum(ytr == 1)) - 1)
        if k < 1: k = 1
        se = SMOTEENN(smote=SMOTE(k_neighbors=k, random_state=seed), random_state=seed)
        Xr, yr = se.fit_resample(Xtr, ytr)
        base = DecisionTreeClassifier(max_depth=cfg['depth'], random_state=seed)
        clf = make_adaboost(base, n_est=cfg['n_est'], rs=seed)
        clf.fit(Xr, yr)
        y_pred = (clf.predict_proba(Xte)[:, 1] >= cfg['thresh']).astype(int)
        return compute_metrics_TR(yte, y_pred, yr, clf.predict(Xr), clf.predict_proba(Xr)[:, 1])

    elif method == 'AdaBoost':
        base = DecisionTreeClassifier(max_depth=cfg['depth'], random_state=seed)
        clf = make_adaboost(base, n_est=cfg['n_est'], rs=seed)
        clf.fit(Xtr, ytr)
        y_pred = (clf.predict_proba(Xte)[:, 1] >= cfg['thresh']).astype(int)
        return compute_metrics_TR(yte, y_pred, ytr, clf.predict(Xtr), clf.predict_proba(Xtr)[:, 1])

    elif method == 'RUSBoost':
        base = DecisionTreeClassifier(max_depth=cfg['depth'], random_state=seed)
        try: clf = ImbRUSBoost(estimator=base, n_estimators=cfg['n_est'], learning_rate=1.0, random_state=seed)
        except TypeError: clf = ImbRUSBoost(base_estimator=base, n_estimators=cfg['n_est'], learning_rate=1.0, random_state=seed)
        clf.fit(Xtr, ytr)
        y_pred = (clf.predict_proba(Xte)[:, 1] >= cfg['thresh']).astype(int)
        return compute_metrics_T(yte, y_pred)

    elif method == 'HUE':
        bags = []
        for i in range(cfg['n_estimators']):
            rf = RandomForestClassifier(n_estimators=10, max_depth=cfg['max_depth'], random_state=seed + i)
            k = min(5, min(np.sum(ytr == 0), np.sum(ytr == 1)) - 1)
            if k < 1: k = 1
            sm = SMOTE(k_neighbors=k, random_state=seed + i)
            try: Xr, yr = sm.fit_resample(Xtr, ytr)
            except: Xr, yr = Xtr, ytr
            rf.fit(Xr, yr); bags.append(rf)
        probas = np.mean([b.predict_proba(Xte)[:, 1] for b in bags], axis=0)
        y_pred = (probas >= cfg['thresh']).astype(int)
        return compute_metrics_T(yte, y_pred)

def main():
    X, y = load_ilpd(DATA_PATH)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)  # 100 folds
    folds = list(cv.split(X, y))
    n_folds = len(folds)
    print(f"Running {n_folds} folds (5-split x 20-repeat)\n")

    all_rows = []
    for method in TABLE_ORDER:
        cfg = BEST_CONFIGS[method]
        paper = PAPER_TABLE[method]

        t0 = time.time()
        fold_metrics = []
        for fi, (tr, te) in enumerate(folds):
            seed = SEED + fi
            try:
                m = run_fold(method, cfg, X[tr], y[tr], X[te], y[te], seed)
                fold_metrics.append(m)
            except Exception as e:
                pass

        avg = {k: np.mean([f[k] for f in fold_metrics]) for k in ['ACC', 'AUC', 'F1', 'AP', 'GMEAN']}

        # Calculate 5-metric error (all metrics included)
        errs = [abs(avg['ACC'] - paper['ACC'])]
        for k in ['AUC', 'F1', 'AP', 'GMEAN']:
            errs.append(abs(avg[k] - paper[k]) * 100)
        mean_err = np.mean(errs)

        tag = 'OK' if mean_err < 3 else ('~' if mean_err < 5 else 'X')
        elapsed = time.time() - t0
        print(f"{method:20s} avg={mean_err:.1f}% [{tag}]  ACC={avg['ACC']:.2f} AUC={avg['AUC']:.3f} F1={avg['F1']:.3f} AP={avg['AP']:.3f} GM={avg['GMEAN']:.3f}  ({elapsed:.0f}s)")

        all_rows.append({'Method': method, **avg, 'note': f'{mean_err:.1f}% [{tag}]'})

    # ── Print final table ──
    print(f"\n{'='*110}")
    print(f"{'Method':20s} | {'ACC':>7s} {'AUC':>6s} {'F1':>6s} {'AP':>6s} {'GM':>6s} | {'Paper ACC':>9s} {'AUC':>6s} {'F1':>6s} {'AP':>6s} {'GM':>6s} | {'Err':>8s}")
    print('-' * 110)
    for row in all_rows:
        m = row['Method']
        p = PAPER_TABLE[m]
        print(f"{m:20s} | {row['ACC']:6.2f}% {row['AUC']:.3f} {row['F1']:.3f} {row['AP']:.3f} {row['GMEAN']:.3f} | {p['ACC']:8.2f}% {p['AUC']:.3f} {p['F1']:.3f} {p['AP']:.3f} {p['GMEAN']:.3f} | {row['note']}")
    print('=' * 110)

    out_path = os.path.join(SCRIPT_DIR, "ilpd_replication_results.csv")
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

if __name__ == '__main__':
    main()
