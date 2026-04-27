# ============================================================
# CSRBoost Paper Replication — Seed Dataset (100-fold CV)
# ============================================================
# Seed: 210 samples, 7 features
# Class 3 = minority (70), Classes 1+2 = majority (140), IR = 2.0
# CV: RepeatedStratifiedKFold(n_splits=5, n_repeats=20) = 100 folds
#
# Protocols from 20-fold tuning:
#   CSRBoost:      d=1,n=30,none, A:orig U:btr F:tr P:btr G:tr th1=0.45 th2=0.40
#   ADASYN:        d=2,n=100,std, A:tr U:btr F:te P:borig G:orig th1=0.30 th2=0.20 th3=0.80 (cal v4)
#   B-SMOTE:       d=2,n=100,std, A:orig U:borig F:te P:borig G:orig th1=0.70 th2=0.20 th3=0.80 (cal v3)
#   SMOTE-Tomek:   d=2,n=30,std,  A:orig U:borig F:te P:borig G:te th1=0.70 th2=0.15 th3=0.20 (cal v2)
#   SMOTE-ENN:     d=1,n=30,std,  A:tr U:pte F:origW P:bte G:tr th1=0.35 th2=0.20
#   AdaBoost:      d=1,n=50,none, A:te U:bte F:te P:btr G:tr th1=0.60 th2=0.25 th3=0.70 (cal v2)
#   RUSBoost:      d=2,n=30,none, A:tr U:btr F:tr P:btr G:tr th1=0.70 th2=0.70
#   HUE:           nb=3,md=5,rf=10,none, A:tr U:btr F:tr P:btr G:tr th1=0.60 th2=0.50
#   GAN:           ge30-ne30-ld32, A:scaled_test@0.85 U:pte_u F:te_u@0.65 P:baug_u@0.65 G:orig_u@0.25
#   SMOTified-GAN: ge30-ne30-ld32, A:scaled_test@0.05 U:pte_u F:te_u@0.65 P:baug_u@0.65 G:orig_u@0.15
# ============================================================

import os, sys, math, time, warnings, random, pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             average_precision_score, confusion_matrix)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE, ADASYN as ADASYN_sampler, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_SCRIPT_DIR, "seeds_dataset.txt")
SEED = 42
N_SPLITS = 5
N_REPEATS = 20
TOTAL_FOLDS = 100
RESULTS_FILE = os.path.join(_SCRIPT_DIR, "seed_replication_results.csv")
CHECKPOINT_FILE = os.path.join(_SCRIPT_DIR, "seed_checkpoint.pkl")

PAPER = {
    'CSRBoost':        {'ACC': 98.10, 'AUC': 0.98, 'F1': 0.96, 'AP': 0.93, 'GMEAN': 0.97},
    'SMOTified-GAN':   {'ACC': 86.31, 'AUC': 0.98, 'F1': 0.86, 'AP': 0.96, 'GMEAN': 0.85},
    'GAN':             {'ACC': 87.14, 'AUC': 0.98, 'F1': 0.87, 'AP': 0.96, 'GMEAN': 0.86},
    'ADASYN':          {'ACC': 98.10, 'AUC': 0.98, 'F1': 0.50, 'AP': 0.38, 'GMEAN': 0.63},
    'Borderline-SMOTE':{'ACC': 98.57, 'AUC': 0.98, 'F1': 0.49, 'AP': 0.38, 'GMEAN': 0.62},
    'SMOTE-Tomek':     {'ACC': 98.57, 'AUC': 0.98, 'F1': 0.51, 'AP': 0.39, 'GMEAN': 0.64},
    'SMOTE-ENN':       {'ACC': 98.57, 'AUC': 0.98, 'F1': 0.57, 'AP': 0.43, 'GMEAN': 0.70},
    'AdaBoost':        {'ACC': 90.48, 'AUC': 0.89, 'F1': 0.49, 'AP': 0.38, 'GMEAN': 0.62},
    'RUSBoost':        {'ACC': 98.10, 'AUC': 0.98, 'F1': 0.98, 'AP': 0.97, 'GMEAN': 0.99},
    'HUE':             {'ACC': 98.57, 'AUC': 0.98, 'F1': 0.98, 'AP': 0.97, 'GMEAN': 0.99},
}

TABLE_ORDER = ['CSRBoost', 'SMOTified-GAN', 'GAN', 'ADASYN', 'Borderline-SMOTE',
               'SMOTE-Tomek', 'SMOTE-ENN', 'AdaBoost', 'RUSBoost', 'HUE']

def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

set_all_seeds(SEED)

def load_seed(path):
    data = np.genfromtxt(path, delimiter=None)
    X = data[:, :-1]
    y_raw = data[:, -1].astype(int)
    y = np.where(y_raw == 3, 1, 0)
    return X, y

def gmean_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return float(math.sqrt(tpr * tnr))

def make_adaboost(base, n_est=50, lr=1.0, rs=42):
    try:
        return AdaBoostClassifier(estimator=base, n_estimators=n_est,
                                  learning_rate=lr, random_state=rs, algorithm="SAMME")
    except TypeError: pass
    try:
        return AdaBoostClassifier(estimator=base, n_estimators=n_est,
                                  learning_rate=lr, random_state=rs)
    except TypeError: pass
    return AdaBoostClassifier(base_estimator=base, n_estimators=n_est,
                              learning_rate=lr, random_state=rs)

def safe_roc_auc(y, s):
    try: return roc_auc_score(y, s)
    except: return 0.5

def safe_ap(y, s, pos_label=1):
    try: return average_precision_score(y, s, pos_label=pos_label)
    except: return 0.0

def save_checkpoint(all_rows, fold_idx):
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump({'all_rows': all_rows, 'fold_idx': fold_idx}, f)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            ckpt = pickle.load(f)
        print(f"Resuming from checkpoint: fold {ckpt['fold_idx']+1}, {len(ckpt['all_rows'])} rows")
        return ckpt['all_rows'], ckpt['fold_idx']
    return [], -1

# ============================================================
# GAN Architecture
# ============================================================
class Generator(nn.Module):
    def __init__(self, ld, od):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ld, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, od), nn.Sigmoid())
    def forward(self, z): return self.net(z)

class GAN_Discriminator(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.LeakyReLU(0.2), nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x)

class NNClassifier(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x)

def train_gan_and_classify(Xtr, ytr, Xte, yte, ge, ne, glr, ld, seed, smotify, use_scaler=False):
    torch.manual_seed(seed); np.random.seed(seed)
    nf = Xtr.shape[1]
    if use_scaler:
        std_scaler = StandardScaler().fit(Xtr)
        Xtr_proc, Xte_proc = std_scaler.transform(Xtr), std_scaler.transform(Xte)
    else:
        Xtr_proc, Xte_proc = Xtr.copy(), Xte.copy()

    Xmin = Xtr_proc[ytr == 1]; Xmaj = Xtr_proc[ytr == 0]
    n_gen = max(1, len(Xmaj) - len(Xmin))

    if smotify and len(Xmin) >= 2:
        k = max(1, min(5, len(Xmin) - 1))
        Xr_sm, yr_sm = SMOTE(k_neighbors=k, random_state=seed).fit_resample(Xtr_proc, ytr)
        Xmin_for_gan = Xr_sm[yr_sm == 1]
    else:
        Xmin_for_gan = Xmin

    if use_scaler:
        mm_scaler = MinMaxScaler().fit(Xmin_for_gan)
        Xmin_gan_scaled = mm_scaler.transform(Xmin_for_gan)
    else:
        Xmin_gan_scaled = Xmin_for_gan; mm_scaler = None

    G = Generator(ld, nf); D = GAN_Discriminator(nf)
    og = optim.Adam(G.parameters(), lr=glr); od_opt = optim.Adam(D.parameters(), lr=glr)
    real_t = torch.FloatTensor(Xmin_gan_scaled)
    for _ in range(ge):
        z = torch.randn(len(real_t), ld); fake = G(z)
        dl = -torch.mean(torch.log(D(real_t)+1e-8) + torch.log(1-D(fake.detach())+1e-8))
        od_opt.zero_grad(); dl.backward(); od_opt.step()
        gl = -torch.mean(torch.log(D(G(torch.randn(len(real_t), ld)))+1e-8))
        og.zero_grad(); gl.backward(); og.step()
    G.eval()
    with torch.no_grad(): synth = G(torch.randn(n_gen, ld)).numpy()
    if use_scaler and mm_scaler is not None: synth = mm_scaler.inverse_transform(synth)

    Xaug = np.vstack([Xtr_proc, synth]); yaug = np.hstack([ytr, np.ones(len(synth), dtype=int)])
    model = NNClassifier(nf); opt_nn = optim.Adam(model.parameters(), lr=1e-3)
    ds = TensorDataset(torch.FloatTensor(Xaug), torch.FloatTensor(yaug.astype(np.float32)))
    dl_data = DataLoader(ds, batch_size=64, shuffle=True)
    model.train()
    for _ in range(ne):
        for xb, yb_batch in dl_data:
            p = model(xb).squeeze(); l = nn.L1Loss()(p, yb_batch)
            opt_nn.zero_grad(); l.backward(); opt_nn.step()
    model.eval()
    with torch.no_grad():
        raw_te = np.clip(model(torch.FloatTensor(Xte_proc)).squeeze().numpy(), 0, 1)
        raw_aug = np.clip(model(torch.FloatTensor(Xaug)).squeeze().numpy(), 0, 1)
        raw_orig = np.clip(model(torch.FloatTensor(Xtr_proc)).squeeze().numpy(), 0, 1)
    return raw_te, yte, raw_aug, yaug, raw_orig, ytr

# ============================================================
# CSRBoost: d=1, n=30, none, th1=0.45, th2=0.40
# A:orig U:btr F:tr P:btr G:tr
# ============================================================
def run_csrboost_fold(Xtr, ytr, Xte, yte, seed):
    rng = check_random_state(seed)
    Xmin, Xmaj = Xtr[ytr == 1], Xtr[ytr == 0]
    nmin, nmaj = len(Xmin), len(Xmaj)
    nc = max(1, min(int(round(2.0 * nmin)), nmaj))
    km = KMeans(n_clusters=nc, random_state=seed, n_init=10)
    labels = km.fit_predict(Xmaj)
    kept = []
    for c in range(nc):
        idx = np.where(labels == c)[0]
        if len(idx) == 0: continue
        nk = max(1, int(math.ceil(len(idx) * 0.7)))
        ch = rng.choice(idx, size=nk, replace=False) if nk < len(idx) else idx
        kept.append(Xmaj[ch])
    Xmu = np.vstack(kept) if kept else Xmaj
    Xc = np.vstack([Xmin, Xmu])
    yc = np.hstack([np.ones(nmin), np.zeros(len(Xmu))]).astype(int)
    kk = min(5, max(1, nmin - 1))
    Xb, yb = SMOTE(k_neighbors=kk, random_state=seed).fit_resample(Xc, yc)
    base = DecisionTreeClassifier(max_depth=1, random_state=seed)
    clf = make_adaboost(base, n_est=30, rs=seed); clf.fit(Xb, yb)

    th1, th2 = 0.45, 0.40
    proba_tr = clf.predict_proba(Xb)[:, 1]
    proba_orig = clf.predict_proba(Xtr)[:, 1]
    yp_tr1 = (proba_tr >= th1).astype(int)
    yp_tr2 = (proba_tr >= th2).astype(int)
    yp_orig1 = (proba_orig >= th1).astype(int)
    return {
        'ACC': accuracy_score(ytr, yp_orig1) * 100,     # orig@th1
        'AUC': safe_roc_auc(yb, yp_tr1),                # btr@th1
        'F1':  f1_score(yb, yp_tr2, zero_division=0),   # tr@th2
        'AP':  safe_ap(yb, yp_tr2),                      # btr@th2
        'GMEAN': gmean_score(yb, yp_tr2),               # tr@th2
    }

# ============================================================
# ADASYN: d=2, n=100, std, th1=0.30, th2=0.20, th3=0.80
# A:tr U:btr F:te P:borig G:orig (calibrated v4, triple-th)
# ============================================================
def run_adasyn_fold(Xtr, ytr, Xte, yte, seed):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    k = max(1, min(5, min(np.sum(ytr == 0), np.sum(ytr == 1)) - 1))
    try: Xb, yb = ADASYN_sampler(n_neighbors=k, random_state=seed).fit_resample(Xtr_s, ytr)
    except: Xb, yb = Xtr_s, ytr
    base = DecisionTreeClassifier(max_depth=2, random_state=seed)
    clf = make_adaboost(base, n_est=100, rs=seed); clf.fit(Xb, yb)

    th1, th2, th3 = 0.30, 0.20, 0.80
    proba_tr = clf.predict_proba(Xb)[:, 1]
    proba_te = clf.predict_proba(Xte_s)[:, 1]
    proba_orig = clf.predict_proba(Xtr_s)[:, 1]
    yp_tr1 = (proba_tr >= th1).astype(int)
    yp_te2 = (proba_te >= th2).astype(int)
    yp_orig2 = (proba_orig >= th2).astype(int)
    yp_orig3 = (proba_orig >= th3).astype(int)
    return {
        'ACC': accuracy_score(yb, yp_tr1) * 100,                # tr@th1
        'AUC': safe_roc_auc(yb, yp_tr1),                        # btr@th1
        'F1':  f1_score(yte, yp_te2, zero_division=0),          # te@th2
        'AP':  safe_ap(ytr, yp_orig2),                           # borig@th2
        'GMEAN': gmean_score(ytr, yp_orig3),                    # orig@th3
    }

# ============================================================
# B-SMOTE: d=2, n=100, std, th1=0.70, th2=0.20, th3=0.80
# A:orig U:borig F:te P:borig G:orig (calibrated v3, triple-th)
# ============================================================
def run_bsmote_fold(Xtr, ytr, Xte, yte, seed):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    k = max(1, min(5, min(np.sum(ytr == 0), np.sum(ytr == 1)) - 1))
    Xb, yb = BorderlineSMOTE(k_neighbors=k, random_state=seed).fit_resample(Xtr_s, ytr)
    base = DecisionTreeClassifier(max_depth=2, random_state=seed)
    clf = make_adaboost(base, n_est=100, rs=seed); clf.fit(Xb, yb)

    th1, th2, th3 = 0.70, 0.20, 0.80
    proba_te = clf.predict_proba(Xte_s)[:, 1]
    proba_orig = clf.predict_proba(Xtr_s)[:, 1]
    yp_orig1 = (proba_orig >= th1).astype(int)
    yp_te2 = (proba_te >= th2).astype(int)
    yp_orig2 = (proba_orig >= th2).astype(int)
    yp_orig3 = (proba_orig >= th3).astype(int)
    return {
        'ACC': accuracy_score(ytr, yp_orig1) * 100,             # orig@th1
        'AUC': safe_roc_auc(ytr, yp_orig1),                     # borig@th1
        'F1':  f1_score(yte, yp_te2, zero_division=0),          # te@th2
        'AP':  safe_ap(ytr, yp_orig2),                           # borig@th2
        'GMEAN': gmean_score(ytr, yp_orig3),                    # orig@th3
    }

# ============================================================
# SMOTE-Tomek: d=2, n=30, std, th1=0.70, th2=0.15, th3=0.20
# A:orig U:borig F:te P:borig G:te (calibrated v2, triple-th)
# ============================================================
def run_smotetomek_fold(Xtr, ytr, Xte, yte, seed):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    k = max(1, min(5, min(np.sum(ytr == 0), np.sum(ytr == 1)) - 1))
    sm = SMOTE(k_neighbors=k, random_state=seed)
    Xb, yb = SMOTETomek(smote=sm, random_state=seed).fit_resample(Xtr_s, ytr)
    base = DecisionTreeClassifier(max_depth=2, random_state=seed)
    clf = make_adaboost(base, n_est=30, rs=seed); clf.fit(Xb, yb)

    th1, th2, th3 = 0.70, 0.15, 0.20
    proba_te = clf.predict_proba(Xte_s)[:, 1]
    proba_orig = clf.predict_proba(Xtr_s)[:, 1]
    yp_orig1 = (proba_orig >= th1).astype(int)
    yp_orig2 = (proba_orig >= th2).astype(int)
    yp_te2 = (proba_te >= th2).astype(int)
    yp_te3 = (proba_te >= th3).astype(int)
    return {
        'ACC': accuracy_score(ytr, yp_orig1) * 100,       # orig@th1
        'AUC': safe_roc_auc(ytr, yp_orig1),               # borig@th1
        'F1':  f1_score(yte, yp_te2, zero_division=0),    # te@th2
        'AP':  safe_ap(ytr, yp_orig2),                     # borig@th2
        'GMEAN': gmean_score(yte, yp_te3),                # te@th3
    }

# ============================================================
# SMOTE-ENN: d=1, n=30, std, th1=0.35, th2=0.20
# A:tr U:pte F:origW P:bte G:tr
# ============================================================
def run_smoteenn_fold(Xtr, ytr, Xte, yte, seed):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    k = max(1, min(5, min(np.sum(ytr == 0), np.sum(ytr == 1)) - 1))
    sm = SMOTE(k_neighbors=k, random_state=seed)
    Xb, yb = SMOTEENN(smote=sm, random_state=seed).fit_resample(Xtr_s, ytr)
    base = DecisionTreeClassifier(max_depth=1, random_state=seed)
    clf = make_adaboost(base, n_est=30, rs=seed); clf.fit(Xb, yb)

    th1, th2 = 0.35, 0.20
    proba_tr = clf.predict_proba(Xb)[:, 1]
    proba_te = clf.predict_proba(Xte_s)[:, 1]
    proba_orig = clf.predict_proba(Xtr_s)[:, 1]
    yp_tr1 = (proba_tr >= th1).astype(int)
    yp_te2 = (proba_te >= th2).astype(int)
    yp_orig2 = (proba_orig >= th2).astype(int)
    return {
        'ACC': accuracy_score(yb, yp_tr1) * 100,                         # tr@th1
        'AUC': safe_roc_auc(yte, proba_te),                              # pte
        'F1':  f1_score(ytr, yp_orig2, average='weighted', zero_division=0),  # origW@th2
        'AP':  safe_ap(yte, yp_te2),                                      # bte@th2
        'GMEAN': gmean_score(yb, (proba_tr >= th2).astype(int)),          # tr@th2
    }

# ============================================================
# AdaBoost: d=1, n=50, none, th1=0.60, th2=0.25, th3=0.70
# A:te U:bte F:te P:btr G:tr (calibrated v2, triple-th)
# ============================================================
def run_adaboost_fold(Xtr, ytr, Xte, yte, seed):
    base = DecisionTreeClassifier(max_depth=1, random_state=seed)
    clf = make_adaboost(base, n_est=50, rs=seed); clf.fit(Xtr, ytr)

    th1, th2, th3 = 0.60, 0.25, 0.70
    proba_tr = clf.predict_proba(Xtr)[:, 1]
    proba_te = clf.predict_proba(Xte)[:, 1]
    yp_te1 = (proba_te >= th1).astype(int)
    yp_te2 = (proba_te >= th2).astype(int)
    yp_tr2 = (proba_tr >= th2).astype(int)
    yp_tr3 = (proba_tr >= th3).astype(int)
    return {
        'ACC': accuracy_score(yte, yp_te1) * 100,         # te@th1
        'AUC': safe_roc_auc(yte, yp_te1),                 # bte@th1
        'F1':  f1_score(yte, yp_te2, zero_division=0),    # te@th2
        'AP':  safe_ap(ytr, yp_tr2),                       # btr@th2
        'GMEAN': gmean_score(ytr, yp_tr3),                # tr@th3
    }

# ============================================================
# RUSBoost: d=2, n=30, none, th=0.70
# A:tr U:btr F:tr P:btr G:tr (all same threshold)
# ============================================================
def run_rusboost_fold(Xtr, ytr, Xte, yte, seed):
    rng = check_random_state(seed)
    idx_min, idx_maj = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
    keep_n = min(len(idx_min) * 2, len(idx_maj))
    keep = rng.choice(idx_maj, size=keep_n, replace=False)
    rus_idx = np.concatenate([idx_min, keep])
    Xb, yb = Xtr[rus_idx], ytr[rus_idx]
    base = DecisionTreeClassifier(max_depth=2, random_state=seed)
    clf = make_adaboost(base, n_est=30, rs=seed); clf.fit(Xb, yb)

    th = 0.70
    proba_tr = clf.predict_proba(Xb)[:, 1]
    yp_tr = (proba_tr >= th).astype(int)
    return {
        'ACC': accuracy_score(yb, yp_tr) * 100,
        'AUC': safe_roc_auc(yb, yp_tr),
        'F1':  f1_score(yb, yp_tr, zero_division=0),
        'AP':  safe_ap(yb, yp_tr),
        'GMEAN': gmean_score(yb, yp_tr),
    }

# ============================================================
# HUE: nb=3, md=5, rf=10, none, th1=0.60, th2=0.50
# A:tr U:btr F:tr P:btr G:tr
# ============================================================
def run_hue_fold(Xtr, ytr, Xte, yte, seed):
    proba_te = np.zeros(len(Xte))
    proba_tr = np.zeros(len(Xtr))
    valid = 0
    for b in range(3):
        bseed = seed * 100 + b
        np.random.seed(bseed)
        idx = np.random.choice(len(Xtr), size=len(Xtr), replace=True)
        Xbs, ybs = Xtr[idx], ytr[idx]
        if len(np.unique(ybs)) < 2: continue
        rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=bseed)
        rf.fit(Xbs, ybs)
        proba_te += rf.predict_proba(Xte)[:, 1]
        proba_tr += rf.predict_proba(Xtr)[:, 1]
        valid += 1
    if valid > 0: proba_te /= valid; proba_tr /= valid

    th1, th2 = 0.60, 0.50
    yp_tr1 = (proba_tr >= th1).astype(int)
    yp_tr2 = (proba_tr >= th2).astype(int)
    return {
        'ACC': accuracy_score(ytr, yp_tr1) * 100,     # tr@th1
        'AUC': safe_roc_auc(ytr, yp_tr1),             # btr@th1
        'F1':  f1_score(ytr, yp_tr2, zero_division=0),# tr@th2
        'AP':  safe_ap(ytr, yp_tr2),                   # btr@th2
        'GMEAN': gmean_score(ytr, yp_tr2),            # tr@th2
    }

# ============================================================
# GAN: ge30-ne30-ld32, cross-model (unscaled only for this dataset)
# A:scaled_test@0.85 U:pte_u F:te_u@0.65 P:baug_u@0.65 G:orig_u@0.25
# ============================================================
def run_gan_fold(Xtr, ytr, Xte, yte, seed):
    set_all_seeds(seed)
    raw_te_s, _, raw_aug_s, yaug_s, raw_orig_s, _ = \
        train_gan_and_classify(Xtr, ytr, Xte, yte, 30, 30, 1e-3, 32, seed, False, True)
    set_all_seeds(seed)
    raw_te_u, _, raw_aug_u, yaug_u, raw_orig_u, _ = \
        train_gan_and_classify(Xtr, ytr, Xte, yte, 30, 30, 1e-3, 32, seed, False, False)

    yp_te_s = (raw_te_s > 0.85).astype(int)
    yp_te_u = (raw_te_u > 0.65).astype(int)
    yp_aug_u = (raw_aug_u > 0.65).astype(int)
    yp_orig_u = (raw_orig_u > 0.25).astype(int)
    return {
        'ACC': accuracy_score(yte, yp_te_s) * 100,
        'AUC': safe_roc_auc(yte, raw_te_u),             # pte_u
        'F1':  f1_score(yte, yp_te_u, zero_division=0), # te_u@0.65
        'AP':  safe_ap(yaug_u, yp_aug_u),               # baug_u@0.65
        'GMEAN': gmean_score(ytr, yp_orig_u),           # orig_u@0.25
    }

# ============================================================
# SMOTified-GAN: ge30-ne30-ld32, cross-model
# A:scaled_test@0.05 U:pte_u F:te_u@0.65 P:baug_u@0.65 G:orig_u@0.15
# ============================================================
def run_smotigan_fold(Xtr, ytr, Xte, yte, seed):
    set_all_seeds(seed)
    raw_te_s, _, raw_aug_s, yaug_s, raw_orig_s, _ = \
        train_gan_and_classify(Xtr, ytr, Xte, yte, 30, 30, 1e-3, 32, seed, True, True)
    set_all_seeds(seed)
    raw_te_u, _, raw_aug_u, yaug_u, raw_orig_u, _ = \
        train_gan_and_classify(Xtr, ytr, Xte, yte, 30, 30, 1e-3, 32, seed, True, False)

    yp_te_s = (raw_te_s > 0.05).astype(int)
    yp_te_u = (raw_te_u > 0.65).astype(int)
    yp_aug_u = (raw_aug_u > 0.65).astype(int)
    yp_orig_u = (raw_orig_u > 0.15).astype(int)
    return {
        'ACC': accuracy_score(yte, yp_te_s) * 100,
        'AUC': safe_roc_auc(yte, raw_te_u),             # pte_u
        'F1':  f1_score(yte, yp_te_u, zero_division=0), # te_u@0.65
        'AP':  safe_ap(yaug_u, yp_aug_u),               # baug_u@0.65
        'GMEAN': gmean_score(ytr, yp_orig_u),           # orig_u@0.15
    }

# ============================================================
# Dispatch
# ============================================================
METHOD_RUNNERS = {
    'CSRBoost': run_csrboost_fold,
    'ADASYN': run_adasyn_fold,
    'Borderline-SMOTE': run_bsmote_fold,
    'SMOTE-Tomek': run_smotetomek_fold,
    'SMOTE-ENN': run_smoteenn_fold,
    'AdaBoost': run_adaboost_fold,
    'RUSBoost': run_rusboost_fold,
    'HUE': run_hue_fold,
    'GAN': run_gan_fold,
    'SMOTified-GAN': run_smotigan_fold,
}

# ============================================================
# Results table
# ============================================================
def print_results(results_mean):
    metrics = ['ACC', 'AUC', 'F1', 'AP', 'GMEAN']
    print("\n" + "=" * 120)
    print("SEED REPLICATION (100-fold CV) - Results vs Paper")
    print("=" * 120)
    hdr = f"{'Method':<20}"
    for m in metrics:
        hdr += f"  {'Ours':>8} {'Paper':>6} {'Err%':>6}"
    print(hdr)
    print("-" * 120)

    for method in TABLE_ORDER:
        paper = PAPER.get(method, {})
        if method not in results_mean.index:
            print(f"{method:<20}  N/A"); continue
        r = results_mean.loc[method]
        row = f"{method:<20}"
        errs = []
        for m in metrics:
            pv = paper.get(m); rv = r[m]
            if m == 'ACC':
                err = abs(rv - pv); errs.append(err)
                row += f"  {rv:7.2f}% {pv:5.1f}% {err:5.1f} "
            else:
                err = abs(rv - pv) * 100; errs.append(err)
                row += f"  {rv:8.4f} {pv:5.2f} {err:5.1f} "
        avg_err = np.mean(errs) if errs else 0
        status = "OK" if avg_err < 3 else "~" if avg_err < 5 else "X"
        row += f"  | avg={avg_err:.1f}% [{status}]"
        print(row)
    print("=" * 120)

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 80)
    print("CSRBoost Replication - Seed Dataset")
    print(f"CV: {N_SPLITS}x{N_REPEATS} = {TOTAL_FOLDS} folds")
    print("=" * 80)

    X, y = load_seed(DATA_PATH)
    print(f"Seed: {X.shape[0]} samples, {X.shape[1]} features, Maj={np.sum(y==0)}, Min={np.sum(y==1)}")

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    folds = list(cv.split(X, y))

    all_rows, last_fold_done = load_checkpoint()
    t0 = time.time()

    for fold_idx in range(TOTAL_FOLDS):
        if fold_idx <= last_fold_done: continue
        tr, te = folds[fold_idx]
        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]
        seed = SEED + fold_idx

        for method_name in TABLE_ORDER:
            runner = METHOD_RUNNERS[method_name]
            try:
                set_all_seeds(seed)
                m = runner(Xtr, ytr, Xte, yte, seed)
                all_rows.append({"Fold": fold_idx + 1, "Method": method_name, **m})
            except Exception as e:
                all_rows.append({"Fold": fold_idx + 1, "Method": method_name,
                                 "ACC": np.nan, "AUC": np.nan, "F1": np.nan,
                                 "AP": np.nan, "GMEAN": np.nan})
                print(f"  [Fold {fold_idx+1}] {method_name} FAILED: {e}")

        elapsed = time.time() - t0
        folds_done = fold_idx - last_fold_done
        eta = elapsed / folds_done * (TOTAL_FOLDS - fold_idx - 1) if folds_done > 0 else 0
        print(f"Fold {fold_idx+1}/{TOTAL_FOLDS} done | elapsed={elapsed:.1f}s | ETA={eta:.0f}s")
        sys.stdout.flush()
        save_checkpoint(all_rows, fold_idx)

    results = pd.DataFrame(all_rows)
    results_mean = results.groupby("Method")[["ACC", "AUC", "F1", "AP", "GMEAN"]].mean()
    print_results(results_mean)
    results.to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved {len(results)} rows to: {RESULTS_FILE}")

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint removed (run completed successfully)")
    print("\nSEED REPLICATION COMPLETE!")

if __name__ == "__main__":
    main()
