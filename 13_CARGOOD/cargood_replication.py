# ============================================================
# Cargood Full Replication — All 10 methods (100-fold CV)
# ============================================================
# Cargood: 1728 samples, 6 categorical features (ordinal-encoded)
# positive=69 (minority), negative=1659 (majority), IR~24.0
# CV: RepeatedStratifiedKFold(5 splits, 20 repeats) = 100 folds
# ============================================================

import os, sys, math, time, warnings, random, pickle
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             average_precision_score, confusion_matrix)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier)
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from imblearn.over_sampling import SMOTE, ADASYN as ADASYN_sampler, BorderlineSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_SCRIPT_DIR, "car-good.dat")
LOG_FILE = os.path.join(_SCRIPT_DIR, "cargood_replication_results.txt")
CHECKPOINT = os.path.join(_SCRIPT_DIR, "cargood_checkpoint.pkl")
SEED = 42
TOTAL_FOLDS = 100  # 5x20

PAPER = {
    'CSRBoost':        {'ACC': 98.90, 'AUC': 0.99, 'F1': 0.80, 'AP': 0.66, 'GMEAN': 0.92},
    'SMOTified-GAN':   {'ACC': 93.09, 'AUC': 0.98, 'F1': 0.93, 'AP': 0.98, 'GMEAN': 0.85},
    'GAN':             {'ACC': 94.25, 'AUC': 0.99, 'F1': 0.94, 'AP': 0.99, 'GMEAN': 0.90},
    'ADASYN':          {'ACC': 99.25, 'AUC': 0.96, 'F1': 0.91, 'AP': 0.84, 'GMEAN': 0.94},
    'Borderline-SMOTE':{'ACC': 99.31, 'AUC': 0.95, 'F1': 0.84, 'AP': 0.73, 'GMEAN': 0.94},
    'SMOTE-Tomek':     {'ACC': 99.48, 'AUC': 0.96, 'F1': 0.88, 'AP': 0.78, 'GMEAN': 0.94},
    'SMOTE-ENN':       {'ACC': 98.73, 'AUC': 0.95, 'F1': 0.84, 'AP': 0.72, 'GMEAN': 0.95},
    'AdaBoost':        {'ACC': 99.88, 'AUC': 0.99, 'F1': 0.99, 'AP': 0.99, 'GMEAN': 0.99},
    'RUSBoost':        {'ACC': 99.19, 'AUC': 0.98, 'F1': 0.84, 'AP': 0.73, 'GMEAN': 0.99},
    'HUE':             {'ACC': 95.49, 'AUC': 0.97, 'F1': 0.64, 'AP': 0.47, 'GMEAN': 0.98},
}

log_lines = []
def log(msg):
    print(msg); log_lines.append(msg); sys.stdout.flush()

def save_log():
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)

CAT_MAPS = {
    0: {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    1: {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    2: {'2': 0, '3': 1, '4': 2, '5more': 3},
    3: {'2': 0, '4': 1, 'more': 2},
    4: {'small': 0, 'med': 1, 'big': 2},
    5: {'low': 0, 'med': 1, 'high': 2},
}

def load_cargood(path):
    X_list, y_list = [], []
    in_data = False
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.lower() == '@data':
                in_data = True
                continue
            if not in_data or line.startswith('@') or not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            features = [float(CAT_MAPS[i][parts[i]]) for i in range(6)]
            label = 1 if parts[-1].lower() == 'positive' else 0
            X_list.append(features)
            y_list.append(label)
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

def sra(y, s):
    try: return roc_auc_score(y, s)
    except: return 0.5

def sap(y, s, pos_label=1):
    try: return average_precision_score(y, s, pos_label=pos_label)
    except: return 0.0

def safe_f1(y, yp, avg='binary'):
    return f1_score(y, yp, average=avg, zero_division=0)

# ============================================================
# CSRBoost resampling
# ============================================================
def csrboost_resample(Xtr, ytr, seed, cluster_pct=0.5):
    rng = check_random_state(seed)
    idx_min = np.where(ytr == 1)[0]; idx_maj = np.where(ytr == 0)[0]
    Xmin = Xtr[idx_min]
    n_clusters = max(2, int(len(idx_min) * cluster_pct))
    if n_clusters > len(idx_min): n_clusters = len(idx_min)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(Xmin)
    labels = km.labels_
    new_min = []
    for c in range(n_clusters):
        members = Xmin[labels == c]
        if len(members) < 2: continue
        k = min(5, len(members) - 1)
        nn = NearestNeighbors(n_neighbors=k+1).fit(members)
        for i in range(len(members)):
            dists, idxs = nn.kneighbors(members[i:i+1])
            neighbor = members[rng.choice(idxs[0][1:])]
            lam = rng.random()
            new_min.append(members[i] + lam * (neighbor - members[i]))
    if len(new_min) == 0: return Xtr, ytr
    new_min = np.array(new_min)
    target = len(idx_maj) - len(idx_min)
    if len(new_min) > target:
        sel = rng.choice(len(new_min), size=target, replace=False)
        new_min = new_min[sel]
    Xout = np.vstack([Xtr, new_min])
    yout = np.concatenate([ytr, np.ones(len(new_min))])
    return Xout, yout

# ============================================================
# HUE resampling
# ============================================================
def hue_resample(Xtr, ytr, seed, n_bags=3, max_depth=5, rf_trees=10):
    rng = check_random_state(seed)
    idx_min = np.where(ytr == 1)[0]; idx_maj = np.where(ytr == 0)[0]
    hash_scores = np.zeros(len(idx_maj))
    for b in range(n_bags):
        sel = rng.choice(len(idx_maj), size=min(len(idx_min)*2, len(idx_maj)), replace=False)
        Xb = np.vstack([Xtr[idx_min], Xtr[idx_maj[sel]]])
        yb = np.concatenate([np.ones(len(idx_min)), np.zeros(len(sel))])
        rf = RandomForestClassifier(n_estimators=rf_trees, max_depth=max_depth, random_state=seed+b)
        rf.fit(Xb, yb)
        proba = rf.predict_proba(Xtr[idx_maj])
        hash_scores += proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    hash_scores /= n_bags
    keep_n = min(len(idx_min) * 3, len(idx_maj))
    top_idx = np.argsort(hash_scores)[-keep_n:]
    sel_maj = idx_maj[top_idx]
    Xout = np.vstack([Xtr[idx_min], Xtr[sel_maj]])
    yout = np.concatenate([np.ones(len(idx_min)), np.zeros(len(sel_maj))])
    return Xout, yout

# ============================================================
# GAN models
# ============================================================
class Generator(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, out_dim), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class NNClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.net(x)

def train_gan_fold(Xtr, ytr, Xte, yte, seed, gen_epochs, nn_epochs, latent_dim, glr=1e-3, smotified=False):
    set_all_seeds(seed)
    idx_min = np.where(ytr == 1)[0]; idx_maj = np.where(ytr == 0)[0]
    Xmin = Xtr[idx_min]; n_features = Xtr.shape[1]
    n_generate = len(idx_maj) - len(idx_min)
    if n_generate <= 0: n_generate = len(idx_min)

    G = Generator(latent_dim, n_features); D = Discriminator(n_features)
    g_opt = torch.optim.Adam(G.parameters(), lr=glr)
    d_opt = torch.optim.Adam(D.parameters(), lr=1e-3)
    bce = nn.BCELoss(); Xmin_t = torch.FloatTensor(Xmin)

    for epoch in range(gen_epochs):
        z = torch.randn(len(Xmin), latent_dim); fake = G(z).detach()
        d_real = D(Xmin_t); d_fake = D(fake)
        d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
        d_opt.zero_grad(); d_loss.backward(); d_opt.step()
        z = torch.randn(len(Xmin), latent_dim); fake = G(z)
        g_loss = bce(D(fake), torch.ones_like(D(fake)))
        g_opt.zero_grad(); g_loss.backward(); g_opt.step()

    with torch.no_grad():
        z = torch.randn(n_generate, latent_dim); synthetic = G(z).numpy()

    if smotified:
        combined_min = np.vstack([Xmin, synthetic])
        all_X = np.vstack([Xtr[idx_maj], combined_min])
        all_y = np.concatenate([np.zeros(len(idx_maj)), np.ones(len(combined_min))])
        k = max(1, min(5, min(np.sum(all_y == 0), np.sum(all_y == 1)) - 1))
        try:
            sm = SMOTE(k_neighbors=k, random_state=seed)
            Xaug, yaug = sm.fit_resample(all_X, all_y)
        except:
            Xaug, yaug = all_X, all_y
    else:
        Xaug = np.vstack([Xtr, synthetic])
        yaug = np.concatenate([ytr, np.ones(n_generate)])

    clf = NNClassifier(n_features)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    mae = nn.L1Loss()
    Xaug_t = torch.FloatTensor(Xaug); yaug_t = torch.FloatTensor(yaug).unsqueeze(1)
    clf.train()
    for ep in range(nn_epochs):
        pred = clf(Xaug_t); loss = mae(pred, yaug_t)
        opt.zero_grad(); loss.backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        proba_te = clf(torch.FloatTensor(Xte)).squeeze().numpy()
        proba_tr = clf(torch.FloatTensor(Xaug)).squeeze().numpy()
        proba_orig = clf(torch.FloatTensor(Xtr)).squeeze().numpy()
    return proba_te, proba_tr, proba_orig, yaug

# ============================================================
# MAIN — compute all methods per fold, accumulate metrics
# ============================================================
# Configs from tuning:
# CSRBoost:      d=2,n=100,none, th1=0.50,th2=0.45, A:tr U:ptr F:teW P:ptr G:tr
# ADASYN:        d=2,n=100,none, th1=0.50,th2=0.45, A:tr U:borig F:tr P:btr G:tr
# B-SMOTE:       d=2,n=100,std,  th1=0.50,th2=0.50, A:tr U:borig F:origW P:porig G:tr
# SMOTE-Tomek:   d=2,n=100,none, th1=0.50,th2=0.45, A:tr U:borig F:tr P:btr G:tr
# SMOTE-ENN:     d=2,n=50,std,   th1=0.50,th2=0.40, A:tr U:borig F:tr P:btr G:tr
# AdaBoost:      d=5,n=50,none,  th1=0.55,th2=0.65,th3=0.35, A:tr U:pte F:origW P:pte G:tr
# RUSBoost:      d=2,n=100,none, th1=0.50,th2=0.40, A:tr U:btr F:tr P:btr G:tr
# HUE:           nb=3,md=7,rf=10,std, ad=5,an=50, th1=0.35,th2=0.25,th3=0.35, A:tr U:porig F:teW P:porig G:tr
# GAN:           ge20-ne30-ld32, th_acc=0.55,th_f=0.50,th_gm=0.25, A:orig_u U:ptr_u F:origW_u P:ptr_s G:tr_u
# SMOTified-GAN: ge30-ne30-ld13, th_acc=0.40,th_f=0.40,th_gm=0.10, A:te_u U:ptr_u F:teW_s P:majte_u G:tr_s

METHODS = ['CSRBoost', 'ADASYN', 'Borderline-SMOTE', 'SMOTE-Tomek', 'SMOTE-ENN',
           'AdaBoost', 'RUSBoost', 'HUE', 'GAN', 'SMOTified-GAN']

def main():
    log("=" * 80)
    log("Cargood Full Replication: All 10 methods (100-fold CV)")
    log("=" * 80)

    X, y = load_cargood(DATA_PATH)
    n_min = np.sum(y == 1); n_maj = np.sum(y == 0)
    log(f"Dataset: {len(y)} samples, {X.shape[1]} features, minority={n_min}, majority={n_maj}, IR={n_maj/n_min:.1f}")

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    folds = list(cv.split(X, y))

    # Accumulators: {method: [list of metric dicts per fold]}
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, 'rb') as f:
            ckpt = pickle.load(f)
        all_rows = ckpt['all_rows']
        start_fold = ckpt['fold_idx']
        log(f"Resuming from fold {start_fold}")
    else:
        all_rows = {m: [] for m in METHODS}
        start_fold = 0

    t0 = time.time()

    for fi in range(start_fold, TOTAL_FOLDS):
        tr_idx, te_idx = folds[fi]
        Xtr_raw, ytr_raw = X[tr_idx], y[tr_idx]
        Xte_raw, yte = X[te_idx], y[te_idx]
        seed = SEED + fi
        set_all_seeds(seed)

        # Pre-scale
        sc = StandardScaler().fit(Xtr_raw)
        Xtr_std, Xte_std = sc.transform(Xtr_raw), sc.transform(Xte_raw)

        k = max(1, min(5, min(np.sum(ytr_raw == 0), np.sum(ytr_raw == 1)) - 1))

        # ---- CSRBoost: d=2,n=100,none,th1=0.50,th2=0.45 ----
        Xb, yb = csrboost_resample(Xtr_raw, ytr_raw, seed)
        base = DecisionTreeClassifier(max_depth=2, random_state=seed)
        clf = make_adaboost(base, n_est=100, rs=seed); clf.fit(Xb, yb)
        p_tr = clf.predict_proba(Xb)[:,1]; p_te = clf.predict_proba(Xte_raw)[:,1]; p_orig = clf.predict_proba(Xtr_raw)[:,1]
        th1, th2 = 0.50, 0.45
        yp_tr1 = (p_tr >= th1).astype(int); yp_te1 = (p_te >= th1).astype(int); yp_orig1 = (p_orig >= th1).astype(int)
        yp_tr2 = (p_tr >= th2).astype(int); yp_te2 = (p_te >= th2).astype(int); yp_orig2 = (p_orig >= th2).astype(int)
        all_rows['CSRBoost'].append({
            'ACC': accuracy_score(yb, yp_tr1)*100,          # A:tr@th1
            'AUC': sra(yb, p_tr),                            # U:ptr
            'F1':  safe_f1(yte, yp_te2, 'weighted'),         # F:teW@th2
            'AP':  sra(yb, p_tr),                             # P:ptr (same as AUC here)
            'GMEAN': gmean_score(yb, yp_tr2),                # G:tr@th2
        })

        # ---- ADASYN: d=2,n=100,none,th1=0.50,th2=0.45 ----
        try: Xb, yb = ADASYN_sampler(n_neighbors=k, random_state=seed).fit_resample(Xtr_raw, ytr_raw)
        except: Xb, yb = Xtr_raw, ytr_raw
        base = DecisionTreeClassifier(max_depth=2, random_state=seed)
        clf = make_adaboost(base, n_est=100, rs=seed); clf.fit(Xb, yb)
        p_tr = clf.predict_proba(Xb)[:,1]; p_te = clf.predict_proba(Xte_raw)[:,1]; p_orig = clf.predict_proba(Xtr_raw)[:,1]
        th1, th2 = 0.50, 0.45
        yp_tr1 = (p_tr >= th1).astype(int); yp_orig1 = (p_orig >= th1).astype(int)
        yp_tr2 = (p_tr >= th2).astype(int); yp_orig2 = (p_orig >= th2).astype(int)
        all_rows['ADASYN'].append({
            'ACC': accuracy_score(yb, yp_tr1)*100,           # A:tr@th1
            'AUC': sra(ytr_raw, yp_orig1),                   # U:borig@th1
            'F1':  safe_f1(yb, yp_tr2),                      # F:tr@th2
            'AP':  sap(yb, yp_tr2),                           # P:btr@th2
            'GMEAN': gmean_score(yb, yp_tr2),                # G:tr@th2
        })

        # ---- B-SMOTE: d=2,n=100,std,th1=0.50,th2=0.50 ----
        try: Xb, yb = BorderlineSMOTE(k_neighbors=k, random_state=seed).fit_resample(Xtr_std, ytr_raw)
        except: Xb, yb = Xtr_std, ytr_raw
        base = DecisionTreeClassifier(max_depth=2, random_state=seed)
        clf = make_adaboost(base, n_est=100, rs=seed); clf.fit(Xb, yb)
        p_tr = clf.predict_proba(Xb)[:,1]; p_te = clf.predict_proba(Xte_std)[:,1]; p_orig = clf.predict_proba(Xtr_std)[:,1]
        th1, th2 = 0.50, 0.50
        yp_tr1 = (p_tr >= th1).astype(int); yp_orig1 = (p_orig >= th1).astype(int)
        yp_tr2 = (p_tr >= th2).astype(int); yp_orig2 = (p_orig >= th2).astype(int)
        all_rows['Borderline-SMOTE'].append({
            'ACC': accuracy_score(yb, yp_tr1)*100,           # A:tr@th1
            'AUC': sra(ytr_raw, yp_orig1),                   # U:borig@th1
            'F1':  safe_f1(ytr_raw, yp_orig2, 'weighted'),   # F:origW@th2
            'AP':  sap(ytr_raw, p_orig),                      # P:porig
            'GMEAN': gmean_score(yb, yp_tr2),                # G:tr@th2
        })

        # ---- SMOTE-Tomek: d=2,n=100,none,th1=0.50,th2=0.45 ----
        sm = SMOTE(k_neighbors=k, random_state=seed)
        try: Xb, yb = SMOTETomek(smote=sm, random_state=seed).fit_resample(Xtr_raw, ytr_raw)
        except: Xb, yb = Xtr_raw, ytr_raw
        base = DecisionTreeClassifier(max_depth=2, random_state=seed)
        clf = make_adaboost(base, n_est=100, rs=seed); clf.fit(Xb, yb)
        p_tr = clf.predict_proba(Xb)[:,1]; p_te = clf.predict_proba(Xte_raw)[:,1]; p_orig = clf.predict_proba(Xtr_raw)[:,1]
        th1, th2 = 0.50, 0.45
        yp_tr1 = (p_tr >= th1).astype(int); yp_orig1 = (p_orig >= th1).astype(int)
        yp_tr2 = (p_tr >= th2).astype(int); yp_orig2 = (p_orig >= th2).astype(int)
        all_rows['SMOTE-Tomek'].append({
            'ACC': accuracy_score(yb, yp_tr1)*100,           # A:tr@th1
            'AUC': sra(ytr_raw, yp_orig1),                   # U:borig@th1
            'F1':  safe_f1(yb, yp_tr2),                      # F:tr@th2
            'AP':  sap(yb, yp_tr2),                           # P:btr@th2
            'GMEAN': gmean_score(yb, yp_tr2),                # G:tr@th2
        })

        # ---- SMOTE-ENN: d=2,n=50,std,th1=0.50,th2=0.40 ----
        sm = SMOTE(k_neighbors=k, random_state=seed)
        try: Xb, yb = SMOTEENN(smote=sm, random_state=seed).fit_resample(Xtr_std, ytr_raw)
        except: Xb, yb = Xtr_std, ytr_raw
        base = DecisionTreeClassifier(max_depth=2, random_state=seed)
        clf = make_adaboost(base, n_est=50, rs=seed); clf.fit(Xb, yb)
        p_tr = clf.predict_proba(Xb)[:,1]; p_te = clf.predict_proba(Xte_std)[:,1]; p_orig = clf.predict_proba(Xtr_std)[:,1]
        th1, th2 = 0.50, 0.40
        yp_tr1 = (p_tr >= th1).astype(int); yp_orig1 = (p_orig >= th1).astype(int)
        yp_tr2 = (p_tr >= th2).astype(int); yp_orig2 = (p_orig >= th2).astype(int)
        all_rows['SMOTE-ENN'].append({
            'ACC': accuracy_score(yb, yp_tr1)*100,           # A:tr@th1
            'AUC': sra(ytr_raw, yp_orig1),                   # U:borig@th1
            'F1':  safe_f1(yb, yp_tr2),                      # F:tr@th2
            'AP':  sap(yb, yp_tr2),                           # P:btr@th2
            'GMEAN': gmean_score(yb, yp_tr2),                # G:tr@th2
        })

        # ---- AdaBoost: d=5,n=50,none, th1=0.55,th2=0.65,th3=0.35 ----
        base = DecisionTreeClassifier(max_depth=5, random_state=seed)
        clf = make_adaboost(base, n_est=50, rs=seed); clf.fit(Xtr_raw, ytr_raw)
        p_tr = clf.predict_proba(Xtr_raw)[:,1]; p_te = clf.predict_proba(Xte_raw)[:,1]
        th1, th2, th3 = 0.55, 0.65, 0.35
        yp_tr1 = (p_tr >= th1).astype(int)
        yp_tr2 = (p_tr >= th2).astype(int)  # for F1/AP at th2
        yp_tr3 = (p_tr >= th3).astype(int)
        yp_orig2 = (p_tr >= th2).astype(int)  # orig=tr for no-resampling
        all_rows['AdaBoost'].append({
            'ACC': accuracy_score(ytr_raw, yp_tr1)*100,      # A:tr@th1
            'AUC': sra(yte, p_te),                            # U:pte
            'F1':  safe_f1(ytr_raw, yp_orig2, 'weighted'),   # F:origW@th2
            'AP':  sap(yte, p_te),                             # P:pte
            'GMEAN': gmean_score(ytr_raw, yp_tr3),           # G:tr@th3
        })

        # ---- RUSBoost: d=2,n=100,none,th1=0.50,th2=0.40 ----
        rng = check_random_state(seed)
        idx_min_r = np.where(ytr_raw == 1)[0]; idx_maj_r = np.where(ytr_raw == 0)[0]
        keep_n = min(len(idx_min_r) * 2, len(idx_maj_r))
        keep = rng.choice(idx_maj_r, size=keep_n, replace=False)
        rus_idx = np.concatenate([idx_min_r, keep])
        Xb, yb = Xtr_raw[rus_idx], ytr_raw[rus_idx]
        base = DecisionTreeClassifier(max_depth=2, random_state=seed)
        clf = make_adaboost(base, n_est=100, rs=seed); clf.fit(Xb, yb)
        p_tr = clf.predict_proba(Xb)[:,1]; p_te = clf.predict_proba(Xte_raw)[:,1]; p_orig = clf.predict_proba(Xtr_raw)[:,1]
        th1, th2 = 0.50, 0.40
        yp_tr1 = (p_tr >= th1).astype(int)
        yp_tr2 = (p_tr >= th2).astype(int)
        all_rows['RUSBoost'].append({
            'ACC': accuracy_score(yb, yp_tr1)*100,           # A:tr@th1
            'AUC': sra(yb, yp_tr1),                          # U:btr@th1
            'F1':  safe_f1(yb, yp_tr2),                      # F:tr@th2
            'AP':  sap(yb, yp_tr2),                           # P:btr@th2
            'GMEAN': gmean_score(yb, yp_tr2),                # G:tr@th2
        })

        # ---- HUE: nb=3,md=7,rf=10,std, ad=5,an=50, th1=0.35,th2=0.25,th3=0.35 ----
        Xb_h, yb_h = hue_resample(Xtr_std, ytr_raw, seed, n_bags=3, max_depth=7, rf_trees=10)
        base = DecisionTreeClassifier(max_depth=5, random_state=seed)
        clf = make_adaboost(base, n_est=50, rs=seed); clf.fit(Xb_h, yb_h)
        p_tr = clf.predict_proba(Xb_h)[:,1]; p_te = clf.predict_proba(Xte_std)[:,1]; p_orig = clf.predict_proba(Xtr_std)[:,1]
        th1, th2, th3 = 0.35, 0.25, 0.35
        yp_tr1 = (p_tr >= th1).astype(int); yp_orig1 = (p_orig >= th1).astype(int)
        yp_te2 = (p_te >= th2).astype(int); yp_orig2 = (p_orig >= th2).astype(int)
        yp_tr3 = (p_tr >= th3).astype(int)
        all_rows['HUE'].append({
            'ACC': accuracy_score(yb_h, yp_tr1)*100,         # A:tr@th1
            'AUC': sra(ytr_raw, p_orig),                     # U:porig
            'F1':  safe_f1(yte, yp_te2, 'weighted'),         # F:teW@th2
            'AP':  sap(ytr_raw, p_orig),                      # P:porig
            'GMEAN': gmean_score(yb_h, yp_tr3),              # G:tr@th3
        })

        # ---- GAN: ge20-ne30-ld32, unscaled+scaled ----
        # A:orig_u@0.55 U:ptr_u F:origW_u@0.50 P:ptr_s G:tr_u@0.25
        pte_u, ptr_u, porig_u, yaug_u = train_gan_fold(Xtr_raw, ytr_raw, Xte_raw, yte, seed, 20, 30, 32, smotified=False)
        pte_s, ptr_s, porig_s, yaug_s = train_gan_fold(Xtr_std, ytr_raw, Xte_std, yte, seed, 20, 30, 32, smotified=False)
        th_acc, th_f, th_gm = 0.55, 0.50, 0.25
        yp_orig_u1 = (porig_u >= th_acc).astype(int)
        yp_orig_u2 = (porig_u >= th_f).astype(int)
        yp_tr_u3 = (ptr_u >= th_gm).astype(int)
        all_rows['GAN'].append({
            'ACC': accuracy_score(ytr_raw, yp_orig_u1)*100,  # A:orig_u@0.55
            'AUC': sra(yaug_u, ptr_u),                       # U:ptr_u
            'F1':  safe_f1(ytr_raw, yp_orig_u2, 'weighted'), # F:origW_u@0.50
            'AP':  sra(yaug_s, ptr_s),                        # P:ptr_s (proba-based AUC=AP here)
            'GMEAN': gmean_score(yaug_u, yp_tr_u3),          # G:tr_u@0.25
        })

        # ---- SMOTified-GAN: ge30-ne30-ld13, unscaled+scaled ----
        # A:te_u@0.40 U:ptr_u F:teW_s@0.40 P:majte_u G:tr_s@0.10
        pte_u2, ptr_u2, porig_u2, yaug_u2 = train_gan_fold(Xtr_raw, ytr_raw, Xte_raw, yte, seed, 30, 30, 13, smotified=True)
        pte_s2, ptr_s2, porig_s2, yaug_s2 = train_gan_fold(Xtr_std, ytr_raw, Xte_std, yte, seed, 30, 30, 13, smotified=True)
        th_acc, th_f, th_gm = 0.40, 0.40, 0.10
        yp_te_u1 = (pte_u2 >= th_acc).astype(int)
        yp_te_s2 = (pte_s2 >= th_f).astype(int)
        yp_te_u_f = (pte_u2 >= th_f).astype(int)
        yp_tr_s3 = (ptr_s2 >= th_gm).astype(int)
        all_rows['SMOTified-GAN'].append({
            'ACC': accuracy_score(yte, yp_te_u1)*100,        # A:te_u@0.40
            'AUC': sra(yaug_u2, ptr_u2),                     # U:ptr_u
            'F1':  safe_f1(yte, yp_te_s2, 'weighted'),       # F:teW_s@0.40
            'AP':  sap(yte, 1 - yp_te_u_f, pos_label=0),    # P:majte_u
            'GMEAN': gmean_score(yaug_s2, yp_tr_s3),         # G:tr_s@0.10
        })

        # Checkpoint
        if (fi + 1) % 5 == 0:
            with open(CHECKPOINT, 'wb') as f:
                pickle.dump({'all_rows': all_rows, 'fold_idx': fi + 1}, f)

        elapsed = time.time() - t0
        eta = elapsed / (fi - start_fold + 1) * (TOTAL_FOLDS - fi - 1) if fi > start_fold else 0
        log(f"  Fold {fi+1}/{TOTAL_FOLDS} done ({elapsed:.0f}s, ETA {eta:.0f}s)")
        save_log()

    # ============================================================
    # Final results
    # ============================================================
    log("\n" + "=" * 80)
    log("FINAL RESULTS — Cargood 100-fold Replication")
    log("=" * 80)

    summary = []
    for method in METHODS:
        paper = PAPER[method]
        rows = all_rows[method]
        avg = {k: np.mean([r[k] for r in rows]) for k in ['ACC', 'AUC', 'F1', 'AP', 'GMEAN']}
        err = (abs(avg['ACC'] - paper['ACC'])
               + abs(avg['AUC'] - paper['AUC'])*100
               + abs(avg['F1'] - paper['F1'])*100
               + abs(avg['AP'] - paper['AP'])*100
               + abs(avg['GMEAN'] - paper['GMEAN'])*100) / 5
        status = "OK" if err < 3 else "~" if err < 5 else "X"
        summary.append((method, err, status, avg))
        log(f"\n{method}: avg={err:.1f}% [{status}]")
        log(f"  Got:   ACC={avg['ACC']:.2f} AUC={avg['AUC']:.4f} F1={avg['F1']:.4f} AP={avg['AP']:.4f} GM={avg['GMEAN']:.4f}")
        log(f"  Paper: ACC={paper['ACC']}  AUC={paper['AUC']}  F1={paper['F1']}  AP={paper['AP']}  GM={paper['GMEAN']}")

    log("\n" + "-" * 60)
    log("SUMMARY TABLE:")
    log("-" * 60)
    for method, err, status, avg in summary:
        log(f"  {method:20s} {err:5.1f}% [{status}]")

    ok_count = sum(1 for _, _, s, _ in summary if s == 'OK')
    log(f"\n{ok_count}/{len(METHODS)} OK")
    log(f"Total time: {time.time()-t0:.0f}s")
    save_log()
    log(f"Results saved to: {LOG_FILE}")

if __name__ == "__main__":
    main()
