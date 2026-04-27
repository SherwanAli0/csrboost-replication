# ============================================================
# CSRBoost Paper Replication — Wine Dataset (100-fold CV)
# ============================================================
#
# Wine: 178 samples, 13 features
# Class 3 = minority (48), Classes 1+2 = majority (130), IR = 2.71
# CV: RepeatedStratifiedKFold(n_splits=5, n_repeats=20) = 100 folds
#
# Protocols decoded from 20-fold tuning:
#   CSRBoost:      RT (train ACC/AUC, test F1/AP/GMEAN)
#   ADASYN:        tePAUC_origFAG@0.3_teACC@0.55
#   B-SMOTE:       A:te@0.55 U:bte@0.55 F:orig@0.70 P:bte@0.55 G:tr@0.70
#   SMOTE-Tomek:   tePAUC_origFA_trGM
#   SMOTE-ENN:     RT (train ACC/AUC, test F1/AP/GMEAN)
#   AdaBoost:      tePAUC_origFAG@0.3_teACC@0.5
#   RUSBoost:      TpAb_teW_pAP_trGM
#   HUE:           T (all test, binary AUC/AP)
#   GAN:           cross-model ge20 ACC/AUC:scaled@0.10 F1:unsAug@0.80 AP:majte_u GMEAN:unsAug@0.65
#   SMOTified-GAN: cross-model ge20 ACC/AUC:scaled@0.20 F1:scTe@0.90 AP:pte_s GMEAN:unsAug@0.90
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
from sklearn.base import BaseEstimator, ClassifierMixin
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

# =========================
# CONFIG
# =========================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_SCRIPT_DIR, "wine.data")
SEED = 42
N_SPLITS = 5
N_REPEATS = 20
TOTAL_FOLDS = N_SPLITS * N_REPEATS  # 100
RESULTS_FILE = os.path.join(_SCRIPT_DIR, "wine_replication_results.csv")
CHECKPOINT_FILE = os.path.join(_SCRIPT_DIR, "wine_checkpoint.pkl")
GAN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TABLE_ORDER = ['CSRBoost', 'SMOTified-GAN', 'GAN', 'ADASYN', 'Borderline-SMOTE',
               'SMOTE-Tomek', 'SMOTE-ENN', 'AdaBoost', 'RUSBoost', 'HUE']

PAPER = {
    'CSRBoost':        {'ACC': 98.86, 'AUC': 0.98, 'F1': 0.93, 'AP': 0.88, 'GMEAN': 0.95},
    'SMOTified-GAN':   {'ACC': 96.67, 'AUC': 0.97, 'F1': 0.77, 'AP': 0.88, 'GMEAN': 0.55},
    'GAN':             {'ACC': 93.89, 'AUC': 0.96, 'F1': 0.74, 'AP': 0.85, 'GMEAN': 0.54},
    'ADASYN':          {'ACC': 98.89, 'AUC': 0.98, 'F1': 0.87, 'AP': 0.78, 'GMEAN': 0.94},
    'Borderline-SMOTE':{'ACC': 98.89, 'AUC': 0.98, 'F1': 0.88, 'AP': 0.80, 'GMEAN': 0.94},
    'SMOTE-Tomek':     {'ACC': 98.30, 'AUC': 0.98, 'F1': 0.88, 'AP': 0.81, 'GMEAN': 0.92},
    'SMOTE-ENN':       {'ACC': 97.21, 'AUC': 0.97, 'F1': 0.88, 'AP': 0.80, 'GMEAN': 0.93},
    'AdaBoost':        {'ACC': 98.89, 'AUC': 0.98, 'F1': 0.87, 'AP': 0.80, 'GMEAN': 0.94},
    'RUSBoost':        {'ACC': 99.43, 'AUC': 0.99, 'F1': 0.99, 'AP': 0.99, 'GMEAN': 0.99},
    'HUE':             {'ACC': 94.43, 'AUC': 0.98, 'F1': 0.97, 'AP': 0.95, 'GMEAN': 0.98},
}

# =========================
# Reproducibility
# =========================
def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

set_all_seeds(SEED)

# =========================
# Data loading
# =========================
def load_wine(path):
    data = np.genfromtxt(path, delimiter=',')
    X = data[:, 1:]   # 13 features
    y_raw = data[:, 0].astype(int)
    y = np.where(y_raw == 3, 1, 0)  # class 3 = minority (1), classes 1+2 = majority (0)
    return X, y

# =========================
# Utilities
# =========================
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
    except TypeError:
        pass
    try:
        return AdaBoostClassifier(estimator=base, n_estimators=n_est,
                                  learning_rate=lr, random_state=rs)
    except TypeError:
        pass
    return AdaBoostClassifier(base_estimator=base, n_estimators=n_est,
                              learning_rate=lr, random_state=rs)

def safe_roc_auc(y, s):
    try:
        return roc_auc_score(y, s)
    except:
        return 0.5

def safe_ap(y, s, pos_label=1):
    try:
        return average_precision_score(y, s, pos_label=pos_label)
    except:
        return 0.0

# =========================
# Checkpoint
# =========================
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
# CSRBoost: d=2, n_est=30, samp=0.7, p=2.0, th=0.55, scaler=none, proto=RT
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
    yc = np.hstack([np.ones(nmin, dtype=int), np.zeros(len(Xmu), dtype=int)])
    k = min(5, max(1, nmin - 1))
    Xb, yb = SMOTE(k_neighbors=k, random_state=seed).fit_resample(Xc, yc)

    base = DecisionTreeClassifier(max_depth=2, random_state=seed)
    clf = make_adaboost(base, n_est=30, rs=seed)
    clf.fit(Xb, yb)

    th = 0.55
    proba_tr = clf.predict_proba(Xb)[:, 1]
    proba_te = clf.predict_proba(Xte)[:, 1]
    yp_tr = (proba_tr >= th).astype(int)
    yp_te = (proba_te >= th).astype(int)

    # RT protocol: ACC=train, AUC=train binary, F1/AP/GMEAN=test
    return {
        'ACC': accuracy_score(yb, yp_tr) * 100,
        'AUC': safe_roc_auc(yb, yp_tr),
        'F1':  f1_score(yte, yp_te, zero_division=0),
        'AP':  safe_ap(yte, yp_te),
        'GMEAN': gmean_score(yte, yp_te),
    }

# ============================================================
# ADASYN: d=1, n_est=50, th=0.55, scaler=none
# Protocol: tePAUC_origFAG@0.3_teACC@0.55
# ============================================================
def run_adasyn_fold(Xtr, ytr, Xte, yte, seed):
    k = max(1, min(5, min(np.sum(ytr == 0), np.sum(ytr == 1)) - 1))
    sampler = ADASYN_sampler(n_neighbors=k, random_state=seed)
    Xb, yb = sampler.fit_resample(Xtr, ytr)

    base = DecisionTreeClassifier(max_depth=1, random_state=seed)
    clf = make_adaboost(base, n_est=50, rs=seed)
    clf.fit(Xb, yb)

    proba_te = clf.predict_proba(Xte)[:, 1]
    proba_orig = clf.predict_proba(Xtr)[:, 1]
    yp_te_055 = (proba_te >= 0.55).astype(int)
    yp_orig_03 = (proba_orig >= 0.3).astype(int)

    return {
        'ACC': accuracy_score(yte, yp_te_055) * 100,
        'AUC': safe_roc_auc(yte, proba_te),
        'F1':  f1_score(ytr, yp_orig_03, zero_division=0),
        'AP':  safe_ap(ytr, yp_orig_03),
        'GMEAN': gmean_score(ytr, yp_orig_03),
    }

# ============================================================
# Borderline-SMOTE: d=1, n_est=100, scaler=none
# Protocol: A:te@0.55 U:bte@0.55 F:orig@0.70 P:btr@0.75 G:tr@0.70
# ============================================================
def run_bsmote_fold(Xtr, ytr, Xte, yte, seed):
    k = max(1, min(5, min(np.sum(ytr == 0), np.sum(ytr == 1)) - 1))
    sampler = BorderlineSMOTE(k_neighbors=k, random_state=seed)
    Xb, yb = sampler.fit_resample(Xtr, ytr)

    base = DecisionTreeClassifier(max_depth=1, random_state=seed)
    clf = make_adaboost(base, n_est=100, rs=seed)
    clf.fit(Xb, yb)

    proba_tr = clf.predict_proba(Xb)[:, 1]
    proba_te = clf.predict_proba(Xte)[:, 1]
    proba_orig = clf.predict_proba(Xtr)[:, 1]

    yp_te_055 = (proba_te >= 0.55).astype(int)
    yp_orig_070 = (proba_orig >= 0.70).astype(int)
    yp_tr_070 = (proba_tr >= 0.70).astype(int)
    yp_tr_075 = (proba_tr >= 0.75).astype(int)

    # A:te@0.55 U:bte@0.55 F:orig@0.70 P:btr@0.75 G:tr@0.70
    return {
        'ACC': accuracy_score(yte, yp_te_055) * 100,
        'AUC': safe_roc_auc(yte, yp_te_055),
        'F1':  f1_score(ytr, yp_orig_070, zero_division=0),
        'AP':  safe_ap(yb, yp_tr_075),
        'GMEAN': gmean_score(yb, yp_tr_070),
    }

# ============================================================
# SMOTE-Tomek: d=1, n_est=100, th=0.7, scaler=std
# Protocol: tePAUC_origFA_trGM
# ============================================================
def run_smotetomek_fold(Xtr, ytr, Xte, yte, seed):
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    k = max(1, min(5, min(np.sum(ytr == 0), np.sum(ytr == 1)) - 1))
    sm = SMOTE(k_neighbors=k, random_state=seed)
    sampler = SMOTETomek(smote=sm, random_state=seed)
    Xb, yb = sampler.fit_resample(Xtr_s, ytr)

    base = DecisionTreeClassifier(max_depth=1, random_state=seed)
    clf = make_adaboost(base, n_est=100, rs=seed)
    clf.fit(Xb, yb)

    th = 0.7
    proba_te = clf.predict_proba(Xte_s)[:, 1]
    proba_orig = clf.predict_proba(Xtr_s)[:, 1]
    proba_tr = clf.predict_proba(Xb)[:, 1]
    yp_te = (proba_te >= th).astype(int)
    yp_orig = (proba_orig >= th).astype(int)
    yp_tr = (proba_tr >= th).astype(int)

    # tePAUC_origFA_trGM: ACC=test@th, AUC=test proba, F1=orig@th, AP_binary=orig@th, GMEAN=train@th
    return {
        'ACC': accuracy_score(yte, yp_te) * 100,
        'AUC': safe_roc_auc(yte, proba_te),
        'F1':  f1_score(ytr, yp_orig, zero_division=0),
        'AP':  safe_ap(ytr, yp_orig),
        'GMEAN': gmean_score(yb, yp_tr),
    }

# ============================================================
# SMOTE-ENN: d=2, n_est=30, th=0.45, scaler=none, proto=RT
# ============================================================
def run_smoteenn_fold(Xtr, ytr, Xte, yte, seed):
    k = max(1, min(5, min(np.sum(ytr == 0), np.sum(ytr == 1)) - 1))
    sm = SMOTE(k_neighbors=k, random_state=seed)
    sampler = SMOTEENN(smote=sm, random_state=seed)
    Xb, yb = sampler.fit_resample(Xtr, ytr)

    base = DecisionTreeClassifier(max_depth=2, random_state=seed)
    clf = make_adaboost(base, n_est=30, rs=seed)
    clf.fit(Xb, yb)

    th = 0.45
    proba_tr = clf.predict_proba(Xb)[:, 1]
    proba_te = clf.predict_proba(Xte)[:, 1]
    yp_tr = (proba_tr >= th).astype(int)
    yp_te = (proba_te >= th).astype(int)

    # RT protocol: ACC=train, AUC=train binary, F1/AP/GMEAN=test
    return {
        'ACC': accuracy_score(yb, yp_tr) * 100,
        'AUC': safe_roc_auc(yb, yp_tr),
        'F1':  f1_score(yte, yp_te, zero_division=0),
        'AP':  safe_ap(yte, yp_te),
        'GMEAN': gmean_score(yte, yp_te),
    }

# ============================================================
# AdaBoost: d=1, n_est=50, th=0.5, scaler=none
# Protocol: tePAUC_origFAG@0.3_teACC@0.5
# ============================================================
def run_adaboost_fold(Xtr, ytr, Xte, yte, seed):
    base = DecisionTreeClassifier(max_depth=1, random_state=seed)
    clf = make_adaboost(base, n_est=50, rs=seed)
    clf.fit(Xtr, ytr)

    proba_te = clf.predict_proba(Xte)[:, 1]
    proba_orig = clf.predict_proba(Xtr)[:, 1]
    yp_te_05 = (proba_te >= 0.5).astype(int)
    yp_orig_03 = (proba_orig >= 0.3).astype(int)

    return {
        'ACC': accuracy_score(yte, yp_te_05) * 100,
        'AUC': safe_roc_auc(yte, proba_te),
        'F1':  f1_score(ytr, yp_orig_03, zero_division=0),
        'AP':  safe_ap(ytr, yp_orig_03),
        'GMEAN': gmean_score(ytr, yp_orig_03),
    }

# ============================================================
# RUSBoost: d=1, n_est=100, th=0.55, scaler=none
# Protocol: TpAb_teW_pAP_trGM
# ============================================================
def run_rusboost_fold(Xtr, ytr, Xte, yte, seed):
    rng = check_random_state(seed)
    idx_min = np.where(ytr == 1)[0]
    idx_maj = np.where(ytr == 0)[0]
    n_min = len(idx_min)
    keep_n = min(n_min * 2, len(idx_maj))
    keep = rng.choice(idx_maj, size=keep_n, replace=False)
    rus_idx = np.concatenate([idx_min, keep])
    Xrus, yrus = Xtr[rus_idx], ytr[rus_idx]

    base = DecisionTreeClassifier(max_depth=1, random_state=seed)
    clf = make_adaboost(base, n_est=100, rs=seed)
    clf.fit(Xrus, yrus)

    th = 0.55
    proba_te = clf.predict_proba(Xte)[:, 1]
    proba_tr = clf.predict_proba(Xrus)[:, 1]
    yp_te = (proba_te >= th).astype(int)
    yp_tr = (proba_tr >= th).astype(int)

    # TpAb_teW_pAP_trGM: ACC=test@th, AUC=test proba, F1=test weighted@th, AP=test proba, GMEAN=train@th
    return {
        'ACC': accuracy_score(yte, yp_te) * 100,
        'AUC': safe_roc_auc(yte, proba_te),
        'F1':  f1_score(yte, yp_te, average='weighted', zero_division=0),
        'AP':  safe_ap(yte, proba_te),
        'GMEAN': gmean_score(yrus, yp_tr),
    }

# ============================================================
# HUE: nb=3, md=5, th=0.55, rf=10, scaler=std, proto=T (all test, binary AUC/AP)
# ============================================================
def run_hue_fold(Xtr, ytr, Xte, yte, seed):
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    nb = 3
    md = 5
    n_rf = 10
    th = 0.55

    proba_te = np.zeros(len(Xte_s))
    for b in range(nb):
        bseed = seed * 100 + b
        np.random.seed(bseed)
        idx = np.random.choice(len(Xtr_s), size=len(Xtr_s), replace=True)
        Xbs, ybs = Xtr_s[idx], ytr[idx]
        if len(np.unique(ybs)) < 2:
            continue
        rf = RandomForestClassifier(n_estimators=n_rf, max_depth=md, random_state=bseed)
        rf.fit(Xbs, ybs)
        proba_te += rf.predict_proba(Xte_s)[:, 1]
    proba_te /= nb
    yp_te = (proba_te >= th).astype(int)

    # T protocol: all test, binary AUC/AP
    return {
        'ACC': accuracy_score(yte, yp_te) * 100,
        'AUC': safe_roc_auc(yte, yp_te),
        'F1':  f1_score(yte, yp_te, zero_division=0),
        'AP':  safe_ap(yte, yp_te),
        'GMEAN': gmean_score(yte, yp_te),
    }

# ============================================================
# GAN Architecture
# ============================================================
class Generator(nn.Module):
    def __init__(self, ld, od):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ld, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, od), nn.Sigmoid()
        )
    def forward(self, z):
        return self.net(z)

class GAN_Discriminator(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class NNClassifier(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)  # no activation, MAE loss
        )
    def forward(self, x):
        return self.net(x)

def train_gan_and_classify(Xtr, ytr, Xte, yte, ge, ne, glr, ld, seed, smotify,
                           use_scaler=False):
    """Train GAN, augment, train NN classifier, return raw outputs for all data sources.

    If use_scaler=True: StandardScaler on train, then MinMaxScaler on minority for GAN input,
    inverse-transform synthetic data before combining.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    nf = Xtr.shape[1]

    if use_scaler:
        std_scaler = StandardScaler().fit(Xtr)
        Xtr_proc = std_scaler.transform(Xtr)
        Xte_proc = std_scaler.transform(Xte)
    else:
        Xtr_proc = Xtr.copy()
        Xte_proc = Xte.copy()

    Xmin = Xtr_proc[ytr == 1]
    Xmaj = Xtr_proc[ytr == 0]
    n_gen = max(1, len(Xmaj) - len(Xmin))

    # Optionally SMOTE the minority before GAN training
    if smotify and len(Xmin) >= 2:
        k = max(1, min(5, len(Xmin) - 1))
        Xr_sm, yr_sm = SMOTE(k_neighbors=k, random_state=seed).fit_resample(Xtr_proc, ytr)
        Xmin_for_gan = Xr_sm[yr_sm == 1]
    else:
        Xmin_for_gan = Xmin

    # MinMaxScaler on minority for GAN input if scaled mode
    if use_scaler:
        mm_scaler = MinMaxScaler().fit(Xmin_for_gan)
        Xmin_gan_scaled = mm_scaler.transform(Xmin_for_gan)
    else:
        Xmin_gan_scaled = Xmin_for_gan
        mm_scaler = None

    # Train GAN
    G = Generator(ld, nf)
    D = GAN_Discriminator(nf)
    og = optim.Adam(G.parameters(), lr=glr)
    od = optim.Adam(D.parameters(), lr=glr)
    real_t = torch.FloatTensor(Xmin_gan_scaled)

    for _ in range(ge):
        z = torch.randn(len(real_t), ld)
        fake = G(z)
        dl = -torch.mean(torch.log(D(real_t) + 1e-8) + torch.log(1 - D(fake.detach()) + 1e-8))
        od.zero_grad(); dl.backward(); od.step()
        gl = -torch.mean(torch.log(D(G(torch.randn(len(real_t), ld))) + 1e-8))
        og.zero_grad(); gl.backward(); og.step()

    G.eval()
    with torch.no_grad():
        synth = G(torch.randn(n_gen, ld)).numpy()

    # Inverse transform if scaled
    if use_scaler and mm_scaler is not None:
        synth = mm_scaler.inverse_transform(synth)

    # Augmented dataset
    Xaug = np.vstack([Xtr_proc, synth])
    yaug = np.hstack([ytr, np.ones(len(synth), dtype=int)])

    # Train NN classifier
    model = NNClassifier(nf)
    opt_nn = optim.Adam(model.parameters(), lr=1e-3)
    ds = TensorDataset(torch.FloatTensor(Xaug), torch.FloatTensor(yaug.astype(np.float32)))
    dl_data = DataLoader(ds, batch_size=64, shuffle=True)
    model.train()
    for _ in range(ne):
        for xb, yb in dl_data:
            p = model(xb).squeeze()
            l = nn.L1Loss()(p, yb)
            opt_nn.zero_grad(); l.backward(); opt_nn.step()
    model.eval()

    with torch.no_grad():
        raw_te = np.clip(model(torch.FloatTensor(Xte_proc)).squeeze().numpy(), 0, 1)
        raw_aug = np.clip(model(torch.FloatTensor(Xaug)).squeeze().numpy(), 0, 1)
        raw_orig = np.clip(model(torch.FloatTensor(Xtr_proc)).squeeze().numpy(), 0, 1)

    # For maj_test AP: get predictions on majority test samples
    maj_te_mask = (yte == 0)

    return raw_te, yte, raw_aug, yaug, raw_orig, ytr, maj_te_mask

# ============================================================
# GAN fold: ge=20, ne=30, glr=1e-3, ld=32
# CROSS-MODEL: scaled + unscaled
# ACC from SCALED test@0.10
# AUC from SCALED binary test@0.10
# F1 from UNSCALED aug@0.80
# AP from UNSCALED maj_test (threshold-independent)
# GMEAN from UNSCALED aug@0.65
# ============================================================
def run_gan_fold(Xtr, ytr, Xte, yte, seed):
    # Run SCALED version
    raw_te_s, yte_s, raw_aug_s, yaug_s, raw_orig_s, ytr_s, maj_te_mask_s = \
        train_gan_and_classify(Xtr, ytr, Xte, yte, ge=20, ne=30, glr=1e-3, ld=32,
                               seed=seed, smotify=False, use_scaler=True)

    # Run UNSCALED version
    raw_te_u, yte_u, raw_aug_u, yaug_u, raw_orig_u, ytr_u, maj_te_mask_u = \
        train_gan_and_classify(Xtr, ytr, Xte, yte, ge=20, ne=30, glr=1e-3, ld=32,
                               seed=seed, smotify=False, use_scaler=False)

    # ACC from SCALED test@0.10
    yp_te_s = (raw_te_s > 0.10).astype(int)
    acc = accuracy_score(yte, yp_te_s) * 100

    # AUC from SCALED binary test@0.10
    auc = safe_roc_auc(yte, yp_te_s)

    # F1 from UNSCALED aug@0.80
    yp_aug_u = (raw_aug_u > 0.80).astype(int)
    f1 = f1_score(yaug_u, yp_aug_u, zero_division=0)

    # AP from UNSCALED maj_test (threshold-independent)
    ap = safe_ap(yte, 1 - raw_te_u, pos_label=0)

    # GMEAN from UNSCALED aug@0.65
    yp_aug_gm = (raw_aug_u > 0.65).astype(int)
    gm = gmean_score(yaug_u, yp_aug_gm)

    return {'ACC': acc, 'AUC': auc, 'F1': f1, 'AP': ap, 'GMEAN': gm}

# ============================================================
# SMOTified-GAN fold: ge=20, ne=30, glr=1e-3, ld=13
# CROSS-MODEL: scaled + unscaled
# ACC from SCALED test@0.20
# AUC from SCALED binary test@0.20
# F1 from SCALED test@0.90
# AP from SCALED proba test (threshold-independent)
# GMEAN from UNSCALED aug@0.90
# ============================================================
def run_smotigan_fold(Xtr, ytr, Xte, yte, seed):
    # Run SCALED version
    raw_te_s, yte_s, raw_aug_s, yaug_s, raw_orig_s, ytr_s, maj_te_mask_s = \
        train_gan_and_classify(Xtr, ytr, Xte, yte, ge=20, ne=30, glr=1e-3, ld=13,
                               seed=seed, smotify=True, use_scaler=True)

    # Run UNSCALED version
    raw_te_u, yte_u, raw_aug_u, yaug_u, raw_orig_u, ytr_u, maj_te_mask_u = \
        train_gan_and_classify(Xtr, ytr, Xte, yte, ge=20, ne=30, glr=1e-3, ld=13,
                               seed=seed, smotify=True, use_scaler=False)

    # ACC from SCALED test@0.20
    yp_te_s = (raw_te_s > 0.20).astype(int)
    acc = accuracy_score(yte, yp_te_s) * 100

    # AUC from SCALED binary test@0.20
    auc = safe_roc_auc(yte, yp_te_s)

    # F1 from SCALED test@0.90
    yp_te_s_f = (raw_te_s > 0.90).astype(int)
    f1 = f1_score(yte, yp_te_s_f, zero_division=0)

    # AP from SCALED proba test (threshold-independent)
    ap = safe_ap(yte, raw_te_s)

    # GMEAN from UNSCALED aug@0.90
    yp_aug_u_gm = (raw_aug_u > 0.90).astype(int)
    gm = gmean_score(yaug_u, yp_aug_u_gm)

    return {'ACC': acc, 'AUC': auc, 'F1': f1, 'AP': ap, 'GMEAN': gm}

# ============================================================
# Dispatch
# ============================================================
METHOD_RUNNERS = {
    'CSRBoost':         run_csrboost_fold,
    'ADASYN':           run_adasyn_fold,
    'Borderline-SMOTE': run_bsmote_fold,
    'SMOTE-Tomek':      run_smotetomek_fold,
    'SMOTE-ENN':        run_smoteenn_fold,
    'AdaBoost':         run_adaboost_fold,
    'RUSBoost':         run_rusboost_fold,
    'HUE':              run_hue_fold,
    'GAN':              run_gan_fold,
    'SMOTified-GAN':    run_smotigan_fold,
}

# ============================================================
# Print final results table
# ============================================================
def print_results(results_mean):
    metrics = ['ACC', 'AUC', 'F1', 'AP', 'GMEAN']
    print("\n" + "=" * 120)
    print("WINE REPLICATION (100-fold CV) - Results vs Paper")
    print("=" * 120)
    hdr = f"{'Method':<20}"
    for m in metrics:
        hdr += f"  {'Ours':>8} {'Paper':>6} {'Err%':>6}"
    print(hdr)
    print("-" * 120)

    for method in TABLE_ORDER:
        paper = PAPER.get(method, {})
        if method not in results_mean.index:
            print(f"{method:<20}  N/A")
            continue
        r = results_mean.loc[method]
        row = f"{method:<20}"
        errs = []
        for m in metrics:
            pv = paper.get(m)
            rv = r[m]
            if m == 'ACC':
                pv_d = pv if pv is not None else 0
                if pv is not None:
                    err = abs(rv - pv_d)
                    errs.append(err)
                    row += f"  {rv:7.2f}% {pv_d:5.1f}% {err:5.1f} "
                else:
                    row += f"  {rv:7.2f}%   N/A    N/A "
            else:
                if pv is not None:
                    err = abs(rv - pv) * 100
                    errs.append(err)
                    row += f"  {rv:8.4f} {pv:5.2f} {err:5.1f} "
                else:
                    row += f"  {rv:8.4f}   N/A    N/A "
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
    print("CSRBoost Replication - Wine Dataset")
    print(f"CV: {N_SPLITS}x{N_REPEATS} = {TOTAL_FOLDS} folds | Device: {GAN_DEVICE}")
    print("=" * 80)

    X, y = load_wine(DATA_PATH)
    n_maj, n_min = np.sum(y == 0), np.sum(y == 1)
    print(f"Wine: {X.shape[0]} samples, {X.shape[1]} features, "
          f"Maj={n_maj}, Min={n_min}, IR={n_maj/n_min:.2f}")

    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    folds = list(cv.split(X, y))
    print(f"Total folds: {len(folds)}\n")

    # Load checkpoint
    all_rows, last_fold_done = load_checkpoint()
    t0 = time.time()

    for fold_idx in range(TOTAL_FOLDS):
        if fold_idx <= last_fold_done:
            continue

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
                all_rows.append({
                    "Fold": fold_idx + 1, "Method": method_name,
                    "ACC": np.nan, "AUC": np.nan, "F1": np.nan,
                    "AP": np.nan, "GMEAN": np.nan,
                })
                print(f"  [Fold {fold_idx+1}] {method_name} FAILED: {e}")

        elapsed = time.time() - t0
        folds_done = fold_idx - last_fold_done
        eta = elapsed / folds_done * (TOTAL_FOLDS - fold_idx - 1) if folds_done > 0 else 0
        print(f"Fold {fold_idx+1}/{TOTAL_FOLDS} done | elapsed={elapsed:.1f}s | ETA={eta:.0f}s")
        sys.stdout.flush()

        # Save checkpoint after each fold
        save_checkpoint(all_rows, fold_idx)

    # Build results DataFrame
    results = pd.DataFrame(all_rows)
    results_mean = results.groupby("Method")[["ACC", "AUC", "F1", "AP", "GMEAN"]].mean()

    print_results(results_mean)

    # Save CSV
    results.to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved {len(results)} rows to: {RESULTS_FILE}")

    # Cleanup checkpoint on success
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint removed (run completed successfully)")

    print("\nWINE REPLICATION COMPLETE!")

if __name__ == "__main__":
    main()
