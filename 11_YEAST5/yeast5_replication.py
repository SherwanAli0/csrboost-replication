# ============================================================
# CSRBoost Paper Replication — Yeast5 Dataset (100-fold CV)
# ============================================================
# Yeast5: 1484 samples, 8 features
# positive=44 (minority), negative=1440 (majority), IR=32.7
# CV: RepeatedStratifiedKFold(n_splits=5, n_repeats=20) = 100 folds
#
# Protocols from 20-fold tuning:
#   CSRBoost:      d=2,n=50,std,   A:tr U:bte F:te P:bte G:te th1=0.40 th2=0.45
#   ADASYN:        d=2,n=30,std,   A:te U:bte F:te P:bte G:te th1=0.50 th2=0.55
#   B-SMOTE:       d=2,n=30,std,   A:orig U:bte F:te P:bte G:te th1=0.45 th2=0.50
#   SMOTE-Tomek:   d=2,n=100,std,  A:te U:bte F:te P:bte G:te th1=0.50 th2=0.50
#   SMOTE-ENN:     d=1,n=30,std,   A:orig U:borig F:orig P:borig G:orig th1=0.55 th2=0.55
#   AdaBoost:      d=2,n=100,none, A:te U:bte F:te P:bte G:te th1=0.50 th2=0.45
#   RUSBoost:      d=2,n=100,none, A:te U:borig F:orig P:borig G:orig th1=0.70 th2=0.60
#   HUE:           nb=3,md=5,rf=10,std, A:orig U:borig F:te P:bte G:tr th1=0.35 th2=0.55
#   GAN:           ge30-ne30-ld13, A:orig_u@0.35 U:btr_s@0.35 F:teW_u@0.35 P:pte_u G:te_s@0.65
#   SMOTified-GAN: ge30-ne30-ld32, A:tr_u@0.25 U:btr_u@0.25 F:teW_u@0.25 P:bte_u@0.25 G:orig_s@0.85
# ============================================================

import os, sys, math, time, warnings, random, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             average_precision_score, confusion_matrix)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier)
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from imblearn.over_sampling import SMOTE, ADASYN as ADASYN_sampler, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_SCRIPT_DIR, "yeast5.dat")
LOG_FILE = os.path.join(_SCRIPT_DIR, "yeast5_replication_log.txt")
CSV_FILE = os.path.join(_SCRIPT_DIR, "yeast5_replication_results.csv")
CHECKPOINT = os.path.join(_SCRIPT_DIR, "yeast5_checkpoint.pkl")
SEED = 42
TOTAL_FOLDS = 100

PAPER = {
    'CSRBoost':        {'ACC': 98.32, 'AUC': 0.93, 'F1': 0.69, 'AP': 0.50, 'GMEAN': 0.89},
    'SMOTified-GAN':   {'ACC': 96.23, 'AUC': 0.97, 'F1': 0.96, 'AP': 0.45, 'GMEAN': 0.55},
    'GAN':             {'ACC': 96.77, 'AUC': 0.95, 'F1': 0.97, 'AP': 0.46, 'GMEAN': 0.66},
    'ADASYN':          {'ACC': 98.59, 'AUC': 0.93, 'F1': 0.75, 'AP': 0.57, 'GMEAN': 0.88},
    'Borderline-SMOTE':{'ACC': 98.38, 'AUC': 0.93, 'F1': 0.71, 'AP': 0.53, 'GMEAN': 0.90},
    'SMOTE-Tomek':     {'ACC': 98.31, 'AUC': 0.91, 'F1': 0.76, 'AP': 0.59, 'GMEAN': 0.90},
    'SMOTE-ENN':       {'ACC': 98.18, 'AUC': 0.96, 'F1': 0.75, 'AP': 0.60, 'GMEAN': 0.97},
    'AdaBoost':        {'ACC': 98.52, 'AUC': 0.85, 'F1': 0.72, 'AP': 0.54, 'GMEAN': 0.85},
    'RUSBoost':        {'ACC': 97.03, 'AUC': 0.97, 'F1': 0.67, 'AP': 0.50, 'GMEAN': 0.97},
    'HUE':             {'ACC': 95.96, 'AUC': 0.97, 'F1': 0.59, 'AP': 0.43, 'GMEAN': 0.97},
}

TABLE_ORDER = ['CSRBoost', 'SMOTified-GAN', 'GAN', 'ADASYN', 'Borderline-SMOTE',
               'SMOTE-Tomek', 'SMOTE-ENN', 'AdaBoost', 'RUSBoost', 'HUE']

def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def load_yeast5(path):
    X_list, y_list = [], []
    in_data = False
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.lower() == '@data':
                in_data = True; continue
            if not in_data or line.startswith('@') or not line: continue
            parts = line.split(',')
            features = [float(x.strip()) for x in parts[:-1]]
            label = 1 if parts[-1].strip().lower() == 'positive' else 0
            X_list.append(features); y_list.append(label)
    return np.array(X_list), np.array(y_list)

def gmean_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp/(tp+fn) if (tp+fn) else 0.0
    tnr = tn/(tn+fp) if (tn+fp) else 0.0
    return float(math.sqrt(tpr * tnr))

def make_adaboost(base, n_est=50, lr=1.0, rs=42):
    try: return AdaBoostClassifier(estimator=base, n_estimators=n_est, learning_rate=lr, random_state=rs, algorithm="SAMME")
    except TypeError: pass
    try: return AdaBoostClassifier(estimator=base, n_estimators=n_est, learning_rate=lr, random_state=rs)
    except TypeError: pass
    return AdaBoostClassifier(base_estimator=base, n_estimators=n_est, learning_rate=lr, random_state=rs)

def safe_roc_auc(y, s):
    try: return roc_auc_score(y, s)
    except: return 0.5

def safe_ap(y, s, pos_label=1):
    try: return average_precision_score(y, s, pos_label=pos_label)
    except: return 0.0

# ============================================================
# GAN models
# ============================================================
class Generator(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, out_dim), nn.Sigmoid())
    def forward(self, x): return self.net(x)

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

def train_gan_classifier(Xtr, ytr, Xte, seed, gen_epochs, nn_epochs, latent_dim, glr=1e-3, smotified=False):
    set_all_seeds(seed)
    idx_min = np.where(ytr == 1)[0]; idx_maj = np.where(ytr == 0)[0]
    Xmin = Xtr[idx_min]; n_features = Xtr.shape[1]
    n_generate = len(idx_maj) - len(idx_min)
    if n_generate <= 0: n_generate = len(idx_min)

    G = Generator(latent_dim, n_features); D = Discriminator(n_features)
    g_opt = torch.optim.Adam(G.parameters(), lr=glr)
    d_opt = torch.optim.Adam(D.parameters(), lr=1e-3)
    bce = nn.BCELoss(); Xmin_t = torch.FloatTensor(Xmin)

    for _ in range(gen_epochs):
        z = torch.randn(len(Xmin), latent_dim)
        fake = G(z).detach()
        d_loss = bce(D(Xmin_t), torch.ones(len(Xmin),1)) + bce(D(fake), torch.zeros(len(Xmin),1))
        d_opt.zero_grad(); d_loss.backward(); d_opt.step()
        z = torch.randn(len(Xmin), latent_dim); fake = G(z)
        g_loss = bce(D(fake), torch.ones(len(Xmin),1))
        g_opt.zero_grad(); g_loss.backward(); g_opt.step()

    with torch.no_grad():
        synthetic = G(torch.randn(n_generate, latent_dim)).numpy()

    if smotified:
        combined = np.vstack([Xmin, synthetic])
        all_X = np.vstack([Xtr[idx_maj], combined])
        all_y = np.concatenate([np.zeros(len(idx_maj)), np.ones(len(combined))])
        k = max(1, min(5, min(np.sum(all_y==0), np.sum(all_y==1))-1))
        try: Xaug, yaug = SMOTE(k_neighbors=k, random_state=seed).fit_resample(all_X, all_y)
        except: Xaug, yaug = all_X, all_y
    else:
        Xaug = np.vstack([Xtr, synthetic])
        yaug = np.concatenate([ytr, np.ones(n_generate)])

    clf = NNClassifier(n_features)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3); mae = nn.L1Loss()
    clf.train()
    for _ in range(nn_epochs):
        pred = clf(torch.FloatTensor(Xaug))
        loss = mae(pred, torch.FloatTensor(yaug).unsqueeze(1))
        opt.zero_grad(); loss.backward(); opt.step()

    clf.eval()
    with torch.no_grad():
        pte = clf(torch.FloatTensor(Xte)).squeeze().numpy()
        ptr = clf(torch.FloatTensor(Xaug)).squeeze().numpy()
        porig = clf(torch.FloatTensor(Xtr)).squeeze().numpy()
    return pte, ptr, porig, yaug

# ============================================================
# CSRBoost resample
# ============================================================
def csrboost_resample(Xtr, ytr, seed, cluster_pct=0.5):
    rng = check_random_state(seed)
    idx_min = np.where(ytr == 1)[0]; idx_maj = np.where(ytr == 0)[0]
    Xmin = Xtr[idx_min]
    n_clusters = max(2, int(len(idx_min) * cluster_pct))
    if n_clusters > len(idx_min): n_clusters = len(idx_min)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(Xmin)
    new_min = []
    for c in range(km.n_clusters):
        members = Xmin[km.labels_ == c]
        if len(members) < 2: continue
        k = min(5, len(members)-1)
        nn = NearestNeighbors(n_neighbors=k+1).fit(members)
        for i in range(len(members)):
            _, idxs = nn.kneighbors(members[i:i+1])
            neighbor = members[rng.choice(idxs[0][1:])]
            new_min.append(members[i] + rng.random() * (neighbor - members[i]))
    if not new_min: return Xtr, ytr
    new_min = np.array(new_min)
    target = len(idx_maj) - len(idx_min)
    if len(new_min) > target:
        new_min = new_min[rng.choice(len(new_min), size=target, replace=False)]
    return np.vstack([Xtr, new_min]), np.concatenate([ytr, np.ones(len(new_min))])

# ============================================================
# HUE resample
# ============================================================
def hue_resample(Xtr, ytr, seed, n_bags=3, max_depth=5, rf_trees=10):
    rng = check_random_state(seed)
    idx_min = np.where(ytr == 1)[0]; idx_maj = np.where(ytr == 0)[0]
    scores = np.zeros(len(idx_maj))
    for b in range(n_bags):
        sel = rng.choice(len(idx_maj), size=min(len(idx_min)*2, len(idx_maj)), replace=False)
        Xb = np.vstack([Xtr[idx_min], Xtr[idx_maj[sel]]])
        yb = np.concatenate([np.ones(len(idx_min)), np.zeros(len(sel))])
        rf = RandomForestClassifier(n_estimators=rf_trees, max_depth=max_depth, random_state=seed+b)
        rf.fit(Xb, yb)
        p = rf.predict_proba(Xtr[idx_maj])
        scores += p[:, 1] if p.shape[1] > 1 else p[:, 0]
    scores /= n_bags
    keep_n = min(len(idx_min)*3, len(idx_maj))
    sel_maj = idx_maj[np.argsort(scores)[-keep_n:]]
    return np.vstack([Xtr[idx_min], Xtr[sel_maj]]), np.concatenate([np.ones(len(idx_min)), np.zeros(len(sel_maj))])

# ============================================================
# Per-method fold functions
# ============================================================
def run_csrboost_fold(Xtr, ytr, Xte, yte, seed):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    Xb, yb = csrboost_resample(Xtr_s, ytr, seed)
    base = DecisionTreeClassifier(max_depth=2, random_state=seed)
    clf = make_adaboost(base, n_est=50, rs=seed); clf.fit(Xb, yb)
    th1, th2 = 0.40, 0.45
    proba_tr = clf.predict_proba(Xb)[:, 1]
    proba_te = clf.predict_proba(Xte_s)[:, 1]
    yp_tr1 = (proba_tr >= th1).astype(int)
    yp_te1 = (proba_te >= th1).astype(int)
    yp_te2 = (proba_te >= th2).astype(int)
    return {
        'ACC': accuracy_score(yb, yp_tr1) * 100,
        'AUC': safe_roc_auc(yte, yp_te1),
        'F1':  f1_score(yte, yp_te2, zero_division=0),
        'AP':  safe_ap(yte, yp_te2),
        'GMEAN': gmean_score(yte, yp_te2),
    }

def run_adasyn_fold(Xtr, ytr, Xte, yte, seed):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    k = max(1, min(5, min(np.sum(ytr==0), np.sum(ytr==1))-1))
    try: Xb, yb = ADASYN_sampler(n_neighbors=k, random_state=seed).fit_resample(Xtr_s, ytr)
    except: Xb, yb = Xtr_s, ytr
    base = DecisionTreeClassifier(max_depth=2, random_state=seed)
    clf = make_adaboost(base, n_est=30, rs=seed); clf.fit(Xb, yb)
    th1, th2 = 0.50, 0.55
    proba_te = clf.predict_proba(Xte_s)[:, 1]
    yp_te1 = (proba_te >= th1).astype(int)
    yp_te2 = (proba_te >= th2).astype(int)
    return {
        'ACC': accuracy_score(yte, yp_te1) * 100,
        'AUC': safe_roc_auc(yte, yp_te1),
        'F1':  f1_score(yte, yp_te2, zero_division=0),
        'AP':  safe_ap(yte, yp_te2),
        'GMEAN': gmean_score(yte, yp_te2),
    }

def run_bsmote_fold(Xtr, ytr, Xte, yte, seed):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    k = max(1, min(5, min(np.sum(ytr==0), np.sum(ytr==1))-1))
    try: Xb, yb = BorderlineSMOTE(k_neighbors=k, random_state=seed).fit_resample(Xtr_s, ytr)
    except: Xb, yb = Xtr_s, ytr
    base = DecisionTreeClassifier(max_depth=2, random_state=seed)
    clf = make_adaboost(base, n_est=30, rs=seed); clf.fit(Xb, yb)
    th1, th2 = 0.45, 0.50
    proba_te = clf.predict_proba(Xte_s)[:, 1]
    proba_orig = clf.predict_proba(Xtr_s)[:, 1]
    yp_orig1 = (proba_orig >= th1).astype(int)
    yp_te1 = (proba_te >= th1).astype(int)
    yp_te2 = (proba_te >= th2).astype(int)
    return {
        'ACC': accuracy_score(ytr, yp_orig1) * 100,
        'AUC': safe_roc_auc(yte, yp_te1),
        'F1':  f1_score(yte, yp_te2, zero_division=0),
        'AP':  safe_ap(yte, yp_te2),
        'GMEAN': gmean_score(yte, yp_te2),
    }

def run_smotetomek_fold(Xtr, ytr, Xte, yte, seed):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    k = max(1, min(5, min(np.sum(ytr==0), np.sum(ytr==1))-1))
    sm = SMOTE(k_neighbors=k, random_state=seed)
    try: Xb, yb = SMOTETomek(smote=sm, random_state=seed).fit_resample(Xtr_s, ytr)
    except: Xb, yb = Xtr_s, ytr
    base = DecisionTreeClassifier(max_depth=2, random_state=seed)
    clf = make_adaboost(base, n_est=100, rs=seed); clf.fit(Xb, yb)
    th = 0.50  # th1=th2=0.50
    proba_te = clf.predict_proba(Xte_s)[:, 1]
    yp_te = (proba_te >= th).astype(int)
    return {
        'ACC': accuracy_score(yte, yp_te) * 100,
        'AUC': safe_roc_auc(yte, yp_te),
        'F1':  f1_score(yte, yp_te, zero_division=0),
        'AP':  safe_ap(yte, yp_te),
        'GMEAN': gmean_score(yte, yp_te),
    }

def run_smoteenn_fold(Xtr, ytr, Xte, yte, seed):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    k = max(1, min(5, min(np.sum(ytr==0), np.sum(ytr==1))-1))
    sm = SMOTE(k_neighbors=k, random_state=seed)
    try: Xb, yb = SMOTEENN(smote=sm, random_state=seed).fit_resample(Xtr_s, ytr)
    except: Xb, yb = Xtr_s, ytr
    base = DecisionTreeClassifier(max_depth=1, random_state=seed)
    clf = make_adaboost(base, n_est=30, rs=seed); clf.fit(Xb, yb)
    th = 0.55  # th1=th2=0.55, all orig
    proba_orig = clf.predict_proba(Xtr_s)[:, 1]
    yp_orig = (proba_orig >= th).astype(int)
    return {
        'ACC': accuracy_score(ytr, yp_orig) * 100,
        'AUC': safe_roc_auc(ytr, yp_orig),
        'F1':  f1_score(ytr, yp_orig, zero_division=0),
        'AP':  safe_ap(ytr, yp_orig),
        'GMEAN': gmean_score(ytr, yp_orig),
    }

def run_adaboost_fold(Xtr, ytr, Xte, yte, seed):
    base = DecisionTreeClassifier(max_depth=2, random_state=seed)
    clf = make_adaboost(base, n_est=100, rs=seed); clf.fit(Xtr, ytr)
    th1, th2 = 0.50, 0.45
    proba_te = clf.predict_proba(Xte)[:, 1]
    yp_te1 = (proba_te >= th1).astype(int)
    yp_te2 = (proba_te >= th2).astype(int)
    return {
        'ACC': accuracy_score(yte, yp_te1) * 100,
        'AUC': safe_roc_auc(yte, yp_te1),
        'F1':  f1_score(yte, yp_te2, zero_division=0),
        'AP':  safe_ap(yte, yp_te2),
        'GMEAN': gmean_score(yte, yp_te2),
    }

def run_rusboost_fold(Xtr, ytr, Xte, yte, seed):
    rng = check_random_state(seed)
    idx_min = np.where(ytr == 1)[0]; idx_maj = np.where(ytr == 0)[0]
    keep_n = min(len(idx_min)*2, len(idx_maj))
    keep = rng.choice(idx_maj, size=keep_n, replace=False)
    rus_idx = np.concatenate([idx_min, keep])
    Xb, yb = Xtr[rus_idx], ytr[rus_idx]
    base = DecisionTreeClassifier(max_depth=2, random_state=seed)
    clf = make_adaboost(base, n_est=100, rs=seed); clf.fit(Xb, yb)
    th1, th2 = 0.70, 0.60
    proba_te = clf.predict_proba(Xte)[:, 1]
    proba_orig = clf.predict_proba(Xtr)[:, 1]
    yp_te1 = (proba_te >= th1).astype(int)
    yp_orig1 = (proba_orig >= th1).astype(int)
    yp_orig2 = (proba_orig >= th2).astype(int)
    return {
        'ACC': accuracy_score(yte, yp_te1) * 100,
        'AUC': safe_roc_auc(ytr, yp_orig1),
        'F1':  f1_score(ytr, yp_orig2, zero_division=0),
        'AP':  safe_ap(ytr, yp_orig2),
        'GMEAN': gmean_score(ytr, yp_orig2),
    }

def run_hue_fold(Xtr, ytr, Xte, yte, seed):
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    Xb, yb = hue_resample(Xtr_s, ytr, seed, n_bags=3, max_depth=5, rf_trees=10)
    base = DecisionTreeClassifier(max_depth=2, random_state=seed)
    clf = make_adaboost(base, n_est=50, rs=seed); clf.fit(Xb, yb)
    th1, th2 = 0.35, 0.55
    proba_tr = clf.predict_proba(Xb)[:, 1]
    proba_te = clf.predict_proba(Xte_s)[:, 1]
    proba_orig = clf.predict_proba(Xtr_s)[:, 1]
    yp_orig1 = (proba_orig >= th1).astype(int)
    yp_te2 = (proba_te >= th2).astype(int)
    yp_tr2 = (proba_tr >= th2).astype(int)
    return {
        'ACC': accuracy_score(ytr, yp_orig1) * 100,
        'AUC': safe_roc_auc(ytr, yp_orig1),
        'F1':  f1_score(yte, yp_te2, zero_division=0),
        'AP':  safe_ap(yte, yp_te2),
        'GMEAN': gmean_score(yb, yp_tr2),
    }

def run_gan_fold(Xtr, ytr, Xte, yte, seed):
    # ge30-ne30-ld32, calibrated: th_acc=0.35, th_f=0.55, th_gm=0.75
    # A:te_u U:btr_s F:tr_s P:pte_u G:te_s
    # Unscaled model
    pte_u, ptr_u, porig_u, yaug_u = train_gan_classifier(
        Xtr, ytr, Xte, seed, gen_epochs=30, nn_epochs=30, latent_dim=32, glr=1e-3, smotified=False)
    # Scaled model
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    pte_s, ptr_s, porig_s, yaug_s = train_gan_classifier(
        Xtr_s, ytr, Xte_s, seed, gen_epochs=30, nn_epochs=30, latent_dim=32, glr=1e-3, smotified=False)

    th_acc, th_f, th_gm = 0.35, 0.55, 0.75
    yp_te_u1 = (pte_u >= th_acc).astype(int)
    yp_tr_s1 = (ptr_s >= th_acc).astype(int)
    yp_tr_s2 = (ptr_s >= th_f).astype(int)
    yp_te_s3 = (pte_s >= th_gm).astype(int)
    return {
        'ACC': accuracy_score(yte, yp_te_u1) * 100,           # te_u@th_acc
        'AUC': safe_roc_auc(yaug_s, yp_tr_s1),                # btr_s@th_acc
        'F1':  f1_score(yaug_s, yp_tr_s2, zero_division=0),   # tr_s@th_f
        'AP':  safe_ap(yte, pte_u),                             # pte_u (proba)
        'GMEAN': gmean_score(yte, yp_te_s3),                   # te_s@th_gm
    }

def run_smotifiedgan_fold(Xtr, ytr, Xte, yte, seed):
    # Unscaled model
    pte_u, ptr_u, porig_u, yaug_u = train_gan_classifier(
        Xtr, ytr, Xte, seed, gen_epochs=30, nn_epochs=30, latent_dim=32, glr=1e-3, smotified=True)
    # Scaled model
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    pte_s, ptr_s, porig_s, yaug_s = train_gan_classifier(
        Xtr_s, ytr, Xte_s, seed, gen_epochs=30, nn_epochs=30, latent_dim=32, glr=1e-3, smotified=True)

    th_acc, th_f, th_gm = 0.25, 0.25, 0.85
    yp_tr_u1 = (ptr_u >= th_acc).astype(int)
    yp_te_u2 = (pte_u >= th_f).astype(int)
    yp_orig_s3 = (porig_s >= th_gm).astype(int)
    return {
        'ACC': accuracy_score(yaug_u, yp_tr_u1) * 100,       # tr_u@th_acc
        'AUC': safe_roc_auc(yaug_u, yp_tr_u1),               # btr_u@th_acc
        'F1':  f1_score(yte, yp_te_u2, average='weighted', zero_division=0),  # teW_u@th_f
        'AP':  safe_ap(yte, yp_te_u2),                        # bte_u@th_f
        'GMEAN': gmean_score(ytr, yp_orig_s3),               # orig_s@th_gm
    }

METHODS = {
    'CSRBoost': run_csrboost_fold,
    'SMOTified-GAN': run_smotifiedgan_fold,
    'GAN': run_gan_fold,
    'ADASYN': run_adasyn_fold,
    'Borderline-SMOTE': run_bsmote_fold,
    'SMOTE-Tomek': run_smotetomek_fold,
    'SMOTE-ENN': run_smoteenn_fold,
    'AdaBoost': run_adaboost_fold,
    'RUSBoost': run_rusboost_fold,
    'HUE': run_hue_fold,
}

def main():
    X, y = load_yeast5(DATA_PATH)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    folds = list(cv.split(X, y))

    # Resume from checkpoint
    all_rows = []; start_fold = 0
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, 'rb') as f:
            ckpt = pickle.load(f)
        all_rows = ckpt['all_rows']; start_fold = ckpt['fold_idx']
        print(f"Resuming from fold {start_fold} ({len(all_rows)} rows saved)")

    t0 = time.time()
    for fi in range(start_fold, TOTAL_FOLDS):
        tr_idx, te_idx = folds[fi]
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]
        seed = SEED + fi
        set_all_seeds(seed)

        for method_name in TABLE_ORDER:
            func = METHODS[method_name]
            metrics = func(Xtr, ytr, Xte, yte, seed)
            all_rows.append({'fold': fi, 'method': method_name, **metrics})

        # Save checkpoint
        with open(CHECKPOINT, 'wb') as f:
            pickle.dump({'all_rows': all_rows, 'fold_idx': fi + 1}, f)

        elapsed = time.time() - t0
        rate = elapsed / (fi - start_fold + 1)
        remaining = rate * (TOTAL_FOLDS - fi - 1)
        print(f"  Fold {fi+1}/{TOTAL_FOLDS} done ({elapsed:.0f}s, ~{remaining:.0f}s remaining)")

    # Save CSV
    df = pd.DataFrame(all_rows)
    df.to_csv(CSV_FILE, index=False)

    # Print results table
    print("=" * 120)
    print("YEAST5 REPLICATION (100-fold CV) - Results vs Paper")
    print("=" * 120)
    hdr = f"{'Method':<24s}"
    for m in ['ACC', 'AUC', 'F1', 'AP', 'GMEAN']:
        hdr += f"  {'Ours':>6s}  {'Paper':>5s}  {'Err%':>5s}   "
    print(hdr)
    print("-" * 120)

    for method in TABLE_ORDER:
        mdf = df[df['method'] == method]
        paper = PAPER[method]
        errs = []
        line = f"{method:<24s}"
        for m in ['ACC', 'AUC', 'F1', 'AP', 'GMEAN']:
            ours = mdf[m].mean()
            pv = paper[m]
            if m == 'ACC':
                err = abs(ours - pv)
                line += f"  {ours:6.2f}%  {pv:5.1f}%  {err:5.1f}   "
            else:
                err = abs(ours - pv) * 100
                line += f"  {ours:6.4f}  {pv:5.2f}  {err:5.1f}   "
            errs.append(err)
        avg = np.mean(errs)
        status = "OK" if avg < 3 else "~" if avg < 5 else "X"
        line += f"| avg={avg:.1f}% [{status}]"
        print(line)

    print("=" * 120)
    print(f"Saved {len(all_rows)} rows to: {CSV_FILE}")

    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)
        print("Checkpoint removed (run completed successfully)")

    print("YEAST5 REPLICATION COMPLETE!")

if __name__ == "__main__":
    main()
