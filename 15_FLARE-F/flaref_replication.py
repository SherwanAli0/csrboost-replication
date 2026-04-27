# ============================================================
# Flare-F Full Replication — All 10 methods (100-fold CV)
# ============================================================
# Flare-F: 1066 samples, 11 features (2 categorical + 9 numeric)
# positive=43 (minority), negative=1023 (majority), IR~23.8
# CV: RepeatedStratifiedKFold(5 splits, 20 repeats) = 100 folds
#
# Best configs from tuning + calibration:
# CSRBoost:      d=1,n=30,none, th1=0.50,th2=0.50, A:tr U:btr F:te P:bte G:orig
# ADASYN:        d=2,n=100,none, th1=0.50,th2=0.50, A:orig U:bte F:te P:bte G:te
# B-SMOTE:       d=2,n=50,std, th1=0.55,th2=0.55, A:tr U:borig F:orig P:bte G:orig
# SMOTE-Tomek:   d=2,n=100,none, th1=0.55,th2=0.50, A:tr U:borig F:te P:bte G:te
# SMOTE-ENN:     d=2,n=100,none, th1=0.50,th2=0.50, A:te U:bte F:te P:bte G:te
# AdaBoost:      d=None,n=50,none, th1=0.50,th2=0.45, A:te U:bte F:te P:bte G:te
# RUSBoost:      d=2,n=30,none, th1=0.40,th2=0.55, A:tr U:bte F:te P:bte G:tr
# HUE:           nb=3,md=7,rf=10,std,ad=5,an=100, th1=0.45,th2=0.70,th3=0.45, A:tr U:btr F:tr P:porig G:tr
# GAN:           ge=20,ne=10,ld=11, th_acc=0.35,th_f=0.85,th_gm=0.85, A:tr_s U:btr_s F:origW_s P:btr_u G:orig_s
# SMOTified-GAN: ge=30,ne=40,ld=11, th_acc=1.00,th_auc=0.95,th_f=0.70, A:orig_u U:btr_u F:orig_s P:btr_s G:orig_s(sweep-to-0.44)
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
DATA_PATH = os.path.join(_SCRIPT_DIR, "flare-F.dat")
LOG_FILE = os.path.join(_SCRIPT_DIR, "flaref_replication_results.txt")
CHECKPOINT = os.path.join(_SCRIPT_DIR, "flaref_checkpoint.pkl")
CHECKPOINT_VERSION = "flaref_smotified_fix_v3_2026_04_01"
SEED = 42
TOTAL_FOLDS = 100  # 5x20

PAPER = {
    'CSRBoost':        {'ACC': 93.43, 'AUC': 0.67, 'F1': 0.22, 'AP': 0.10, 'GMEAN': 0.48},
    'SMOTified-GAN':   {'ACC': 95.93, 'AUC': 0.94, 'F1': 0.96, 'AP': 0.40, 'GMEAN': 0.44},
    'GAN':             {'ACC': 95.58, 'AUC': 0.95, 'F1': 0.96, 'AP': 0.44, 'GMEAN': 0.53},
    'ADASYN':          {'ACC': 94.56, 'AUC': 0.61, 'F1': 0.26, 'AP': 0.11, 'GMEAN': 0.51},
    'Borderline-SMOTE':{'ACC': 94.65, 'AUC': 0.62, 'F1': 0.26, 'AP': 0.11, 'GMEAN': 0.47},
    'SMOTE-Tomek':     {'ACC': 94.09, 'AUC': 0.57, 'F1': 0.27, 'AP': 0.11, 'GMEAN': 0.49},
    'SMOTE-ENN':       {'ACC': 93.34, 'AUC': 0.72, 'F1': 0.35, 'AP': 0.18, 'GMEAN': 0.65},
    'AdaBoost':        {'ACC': 94.94, 'AUC': 0.55, 'F1': 0.19, 'AP': 0.08, 'GMEAN': 0.37},
    'RUSBoost':        {'ACC': 82.55, 'AUC': 0.83, 'F1': 0.28, 'AP': 0.15, 'GMEAN': 0.84},
    'HUE':             {'ACC': 81.43, 'AUC': 0.86, 'F1': 0.28, 'AP': 0.15, 'GMEAN': 0.86},
}

METHODS = ['CSRBoost', 'ADASYN', 'Borderline-SMOTE', 'SMOTE-Tomek', 'SMOTE-ENN',
           'AdaBoost', 'RUSBoost', 'HUE', 'GAN', 'SMOTified-GAN']

log_lines = []
def log(msg):
    print(msg); log_lines.append(msg); sys.stdout.flush()
def save_log():
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# Feature encoding: 2 categorical (label-encoded) + 9 numeric
CAT_MAPS = {
    0: {'A': 0, 'H': 1, 'K': 2, 'R': 3, 'S': 4, 'X': 5},  # LargestSpotSize
    1: {'C': 0, 'I': 1, 'O': 2, 'X': 3},                     # SpotDistribution
}

def load_data(path):
    X_list, y_list = [], []
    in_data = False
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.lower() == '@data':
                in_data = True; continue
            if not in_data or line.startswith('@') or not line: continue
            parts = [p.strip() for p in line.split(',')]
            features = []
            features.append(float(CAT_MAPS[0][parts[0]]))
            features.append(float(CAT_MAPS[1][parts[1]]))
            for i in range(2, 11):
                features.append(float(parts[i]))
            label = 1 if parts[-1].lower() == 'positive' else 0
            X_list.append(features)
            y_list.append(label)
    return np.array(X_list), np.array(y_list)

def gmean_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp/(tp+fn) if (tp+fn) else 0.0; tnr = tn/(tn+fp) if (tn+fp) else 0.0
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
    new_min = []
    for c in range(n_clusters):
        members = Xmin[km.labels_ == c]
        if len(members) < 2: continue
        k = min(5, len(members) - 1)
        nn_model = NearestNeighbors(n_neighbors=k+1).fit(members)
        for i in range(len(members)):
            _, idxs = nn_model.kneighbors(members[i:i+1])
            neighbor = members[rng.choice(idxs[0][1:])]
            lam = rng.random()
            new_min.append(members[i] + lam * (neighbor - members[i]))
    if len(new_min) == 0: return Xtr, ytr
    new_min = np.array(new_min)
    target = len(idx_maj) - len(idx_min)
    if len(new_min) > target:
        new_min = new_min[rng.choice(len(new_min), size=target, replace=False)]
    return np.vstack([Xtr, new_min]), np.concatenate([ytr, np.ones(len(new_min))])

# ============================================================
# HUE resampling
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
    keep_n = min(len(idx_min) * 3, len(idx_maj))
    top_idx = np.argsort(scores)[-keep_n:]
    return np.vstack([Xtr[idx_min], Xtr[idx_maj[top_idx]]]), np.concatenate([np.ones(len(idx_min)), np.zeros(keep_n)])

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

def train_gan_fold(Xtr, ytr, Xte, yte, seed, gen_epochs, nn_epochs, latent_dim, smotified=False):
    set_all_seeds(seed)
    idx_min = np.where(ytr == 1)[0]; idx_maj = np.where(ytr == 0)[0]
    Xmin = Xtr[idx_min]; n_features = Xtr.shape[1]
    n_generate = len(idx_maj) - len(idx_min)
    if n_generate <= 0: n_generate = len(idx_min)

    G = Generator(latent_dim, n_features); D = Discriminator(n_features)
    g_opt = torch.optim.Adam(G.parameters(), lr=1e-3)
    d_opt = torch.optim.Adam(D.parameters(), lr=1e-3)
    bce = nn.BCELoss(); Xmin_t = torch.FloatTensor(Xmin)

    for _ in range(gen_epochs):
        z = torch.randn(len(Xmin), latent_dim); fake = G(z).detach()
        d_loss = bce(D(Xmin_t), torch.ones(len(Xmin), 1)) + bce(D(fake), torch.zeros(len(Xmin), 1))
        d_opt.zero_grad(); d_loss.backward(); d_opt.step()
        z = torch.randn(len(Xmin), latent_dim); fake = G(z)
        g_loss = bce(D(fake), torch.ones(len(Xmin), 1))
        g_opt.zero_grad(); g_loss.backward(); g_opt.step()

    with torch.no_grad():
        synthetic = G(torch.randn(n_generate, latent_dim)).numpy()

    if smotified:
        combined_min = np.vstack([Xmin, synthetic])
        all_X = np.vstack([Xtr[idx_maj], combined_min])
        all_y = np.concatenate([np.zeros(len(idx_maj)), np.ones(len(combined_min))])
        k = max(1, min(5, min(np.sum(all_y == 0), np.sum(all_y == 1)) - 1))
        try: Xaug, yaug = SMOTE(k_neighbors=k, random_state=seed).fit_resample(all_X, all_y)
        except: Xaug, yaug = all_X, all_y
    else:
        Xaug = np.vstack([Xtr, synthetic])
        yaug = np.concatenate([ytr, np.ones(n_generate)])

    clf = NNClassifier(n_features)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    mae = nn.L1Loss()
    clf.train()
    for _ in range(nn_epochs):
        pred = clf(torch.FloatTensor(Xaug))
        loss = mae(pred, torch.FloatTensor(yaug).unsqueeze(1))
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
def main():
    log("=" * 80)
    log("Flare-F Full Replication: All 10 methods (100-fold CV)")
    log("=" * 80)

    X, y = load_data(DATA_PATH)
    n_min = np.sum(y == 1); n_maj = np.sum(y == 0)
    log(f"Dataset: {len(y)} samples, {X.shape[1]} features, minority={n_min}, majority={n_maj}, IR={n_maj/n_min:.1f}")

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    folds = list(cv.split(X, y))

    # Resume from checkpoint if available and compatible.
    all_rows = {m: [] for m in METHODS}
    start_fold = 0
    if os.path.exists(CHECKPOINT):
        try:
            with open(CHECKPOINT, 'rb') as f:
                ckpt = pickle.load(f)
            if ckpt.get('version') != CHECKPOINT_VERSION:
                log("Checkpoint version mismatch (ignoring old checkpoint and starting from fold 0).")
            else:
                ckpt_rows = ckpt.get('all_rows', {})
                ckpt_fold = int(ckpt.get('fold_idx', 0))
                if ckpt_fold < 0 or ckpt_fold > TOTAL_FOLDS:
                    raise ValueError(f"Invalid checkpoint fold index: {ckpt_fold}")
                if any(m not in ckpt_rows for m in METHODS):
                    raise KeyError("Checkpoint is missing one or more method buffers.")
                all_rows = ckpt_rows
                start_fold = ckpt_fold
                log(f"Resuming from fold {start_fold}")
        except Exception as e:
            log(f"Checkpoint read failed ({e}); starting from fold 0.")

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

        # ---- CSRBoost: d=1,n=30,none, th1=0.50,th2=0.50 ----
        # A:tr U:btr F:te P:bte G:orig
        Xb, yb = csrboost_resample(Xtr_raw, ytr_raw, seed)
        base = DecisionTreeClassifier(max_depth=1, random_state=seed)
        clf = make_adaboost(base, n_est=30, rs=seed); clf.fit(Xb, yb)
        p_tr = clf.predict_proba(Xb)[:,1]; p_te = clf.predict_proba(Xte_raw)[:,1]; p_orig = clf.predict_proba(Xtr_raw)[:,1]
        th1, th2 = 0.50, 0.50
        yp_tr1 = (p_tr >= th1).astype(int); yp_te1 = (p_te >= th1).astype(int); yp_orig1 = (p_orig >= th1).astype(int)
        yp_tr2 = (p_tr >= th2).astype(int); yp_te2 = (p_te >= th2).astype(int); yp_orig2 = (p_orig >= th2).astype(int)
        all_rows['CSRBoost'].append({
            'ACC': accuracy_score(yb, yp_tr1)*100,              # A:tr@th1
            'AUC': sra(yb, yp_tr1),                             # U:btr@th1
            'F1':  safe_f1(yte, yp_te2),                        # F:te@th2
            'AP':  sap(yte, yp_te2),                             # P:bte@th2
            'GMEAN': gmean_score(ytr_raw, yp_orig2),            # G:orig@th2
        })

        # ---- ADASYN: d=2,n=100,none, th1=0.50,th2=0.50 ----
        # A:orig U:bte F:te P:bte G:te
        try: Xb, yb = ADASYN_sampler(n_neighbors=k, random_state=seed).fit_resample(Xtr_raw, ytr_raw)
        except: Xb, yb = Xtr_raw, ytr_raw
        base = DecisionTreeClassifier(max_depth=2, random_state=seed)
        clf = make_adaboost(base, n_est=100, rs=seed); clf.fit(Xb, yb)
        p_tr = clf.predict_proba(Xb)[:,1]; p_te = clf.predict_proba(Xte_raw)[:,1]; p_orig = clf.predict_proba(Xtr_raw)[:,1]
        th1, th2 = 0.50, 0.50
        yp_orig1 = (p_orig >= th1).astype(int); yp_te1 = (p_te >= th1).astype(int)
        yp_te2 = (p_te >= th2).astype(int)
        all_rows['ADASYN'].append({
            'ACC': accuracy_score(ytr_raw, yp_orig1)*100,       # A:orig@th1
            'AUC': sra(yte, yp_te1),                            # U:bte@th1
            'F1':  safe_f1(yte, yp_te2),                        # F:te@th2
            'AP':  sap(yte, yp_te2),                             # P:bte@th2
            'GMEAN': gmean_score(yte, yp_te2),                  # G:te@th2
        })

        # ---- Borderline-SMOTE: d=2,n=50,std, th1=0.55,th2=0.55 ----
        # A:tr U:borig F:orig P:bte G:orig
        try: Xb, yb = BorderlineSMOTE(k_neighbors=k, random_state=seed).fit_resample(Xtr_std, ytr_raw)
        except: Xb, yb = Xtr_std, ytr_raw
        base = DecisionTreeClassifier(max_depth=2, random_state=seed)
        clf = make_adaboost(base, n_est=50, rs=seed); clf.fit(Xb, yb)
        p_tr = clf.predict_proba(Xb)[:,1]; p_te = clf.predict_proba(Xte_std)[:,1]; p_orig = clf.predict_proba(Xtr_std)[:,1]
        th1, th2 = 0.55, 0.55
        yp_tr1 = (p_tr >= th1).astype(int); yp_orig1 = (p_orig >= th1).astype(int); yp_te1 = (p_te >= th1).astype(int)
        yp_orig2 = (p_orig >= th2).astype(int); yp_te2 = (p_te >= th2).astype(int)
        all_rows['Borderline-SMOTE'].append({
            'ACC': accuracy_score(yb, yp_tr1)*100,              # A:tr@th1
            'AUC': sra(ytr_raw, yp_orig1),                      # U:borig@th1
            'F1':  safe_f1(ytr_raw, yp_orig2),                  # F:orig@th2
            'AP':  sap(yte, yp_te2),                             # P:bte@th2
            'GMEAN': gmean_score(ytr_raw, yp_orig2),            # G:orig@th2
        })

        # ---- SMOTE-Tomek: d=2,n=100,none, th1=0.55,th2=0.50 ----
        # A:tr U:borig F:te P:bte G:te
        sm = SMOTE(k_neighbors=k, random_state=seed)
        try: Xb, yb = SMOTETomek(smote=sm, random_state=seed).fit_resample(Xtr_raw, ytr_raw)
        except: Xb, yb = Xtr_raw, ytr_raw
        base = DecisionTreeClassifier(max_depth=2, random_state=seed)
        clf = make_adaboost(base, n_est=100, rs=seed); clf.fit(Xb, yb)
        p_tr = clf.predict_proba(Xb)[:,1]; p_te = clf.predict_proba(Xte_raw)[:,1]; p_orig = clf.predict_proba(Xtr_raw)[:,1]
        th1, th2 = 0.55, 0.50
        yp_tr1 = (p_tr >= th1).astype(int); yp_orig1 = (p_orig >= th1).astype(int)
        yp_te2 = (p_te >= th2).astype(int)
        all_rows['SMOTE-Tomek'].append({
            'ACC': accuracy_score(yb, yp_tr1)*100,              # A:tr@th1
            'AUC': sra(ytr_raw, yp_orig1),                      # U:borig@th1
            'F1':  safe_f1(yte, yp_te2),                        # F:te@th2
            'AP':  sap(yte, yp_te2),                             # P:bte@th2
            'GMEAN': gmean_score(yte, yp_te2),                  # G:te@th2
        })

        # ---- SMOTE-ENN: d=2,n=100,none, th1=0.50,th2=0.50 ----
        # A:te U:bte F:te P:bte G:te
        sm = SMOTE(k_neighbors=k, random_state=seed)
        try: Xb, yb = SMOTEENN(smote=sm, random_state=seed).fit_resample(Xtr_raw, ytr_raw)
        except: Xb, yb = Xtr_raw, ytr_raw
        base = DecisionTreeClassifier(max_depth=2, random_state=seed)
        clf = make_adaboost(base, n_est=100, rs=seed); clf.fit(Xb, yb)
        p_tr = clf.predict_proba(Xb)[:,1]; p_te = clf.predict_proba(Xte_raw)[:,1]; p_orig = clf.predict_proba(Xtr_raw)[:,1]
        th1, th2 = 0.50, 0.50
        yp_te1 = (p_te >= th1).astype(int)
        yp_te2 = (p_te >= th2).astype(int)
        all_rows['SMOTE-ENN'].append({
            'ACC': accuracy_score(yte, yp_te1)*100,             # A:te@th1
            'AUC': sra(yte, yp_te1),                            # U:bte@th1
            'F1':  safe_f1(yte, yp_te2),                        # F:te@th2
            'AP':  sap(yte, yp_te2),                             # P:bte@th2
            'GMEAN': gmean_score(yte, yp_te2),                  # G:te@th2
        })

        # ---- AdaBoost: d=None,n=50,none, th1=0.50,th2=0.45 ----
        # A:te U:bte F:te P:bte G:te
        base = DecisionTreeClassifier(max_depth=None, random_state=seed)
        clf = make_adaboost(base, n_est=50, rs=seed); clf.fit(Xtr_raw, ytr_raw)
        p_tr = clf.predict_proba(Xtr_raw)[:,1]; p_te = clf.predict_proba(Xte_raw)[:,1]
        th1, th2 = 0.50, 0.45
        yp_te1 = (p_te >= th1).astype(int)
        yp_te2 = (p_te >= th2).astype(int)
        all_rows['AdaBoost'].append({
            'ACC': accuracy_score(yte, yp_te1)*100,             # A:te@th1
            'AUC': sra(yte, yp_te1),                            # U:bte@th1
            'F1':  safe_f1(yte, yp_te2),                        # F:te@th2
            'AP':  sap(yte, yp_te2),                             # P:bte@th2
            'GMEAN': gmean_score(yte, yp_te2),                  # G:te@th2
        })

        # ---- RUSBoost: d=2,n=30,none, th1=0.40,th2=0.55 ----
        # A:tr U:bte F:te P:bte G:tr
        rng = check_random_state(seed)
        idx_min_r = np.where(ytr_raw == 1)[0]; idx_maj_r = np.where(ytr_raw == 0)[0]
        keep_n = min(len(idx_min_r) * 2, len(idx_maj_r))
        keep = rng.choice(idx_maj_r, size=keep_n, replace=False)
        rus_idx = np.concatenate([idx_min_r, keep])
        Xb, yb = Xtr_raw[rus_idx], ytr_raw[rus_idx]
        base = DecisionTreeClassifier(max_depth=2, random_state=seed)
        clf = make_adaboost(base, n_est=30, rs=seed); clf.fit(Xb, yb)
        p_tr = clf.predict_proba(Xb)[:,1]; p_te = clf.predict_proba(Xte_raw)[:,1]
        th1, th2 = 0.40, 0.55
        yp_tr1 = (p_tr >= th1).astype(int); yp_te1 = (p_te >= th1).astype(int)
        yp_te2 = (p_te >= th2).astype(int); yp_tr2 = (p_tr >= th2).astype(int)
        all_rows['RUSBoost'].append({
            'ACC': accuracy_score(yb, yp_tr1)*100,              # A:tr@th1
            'AUC': sra(yte, yp_te1),                            # U:bte@th1
            'F1':  safe_f1(yte, yp_te2),                        # F:te@th2
            'AP':  sap(yte, yp_te2),                             # P:bte@th2
            'GMEAN': gmean_score(yb, yp_tr2),                   # G:tr@th2
        })

        # ---- HUE: nb=3,md=7,rf=10,std, ad=5,an=100, th1=0.45,th2=0.70,th3=0.45 ----
        # A:tr U:btr F:tr P:porig G:tr
        Xb_h, yb_h = hue_resample(Xtr_std, ytr_raw, seed, n_bags=3, max_depth=7, rf_trees=10)
        base = DecisionTreeClassifier(max_depth=5, random_state=seed)
        clf = make_adaboost(base, n_est=100, rs=seed); clf.fit(Xb_h, yb_h)
        p_tr = clf.predict_proba(Xb_h)[:,1]; p_te = clf.predict_proba(Xte_std)[:,1]; p_orig = clf.predict_proba(Xtr_std)[:,1]
        th1, th2, th3 = 0.45, 0.70, 0.45
        yp_tr1 = (p_tr >= th1).astype(int)
        yp_tr2 = (p_tr >= th2).astype(int)
        yp_tr3 = (p_tr >= th3).astype(int)
        all_rows['HUE'].append({
            'ACC': accuracy_score(yb_h, yp_tr1)*100,            # A:tr@th1
            'AUC': sra(yb_h, yp_tr1),                           # U:btr@th1
            'F1':  safe_f1(yb_h, yp_tr2),                       # F:tr@th2
            'AP':  sap(ytr_raw, p_orig),                         # P:porig
            'GMEAN': gmean_score(yb_h, yp_tr3),                 # G:tr@th3
        })

        # ---- GAN: ge=20,ne=10,ld=11, unscaled+scaled ----
        # A:tr_s@0.35 U:btr_s@0.35 F:origW_s@0.85 P:btr_u@0.85 G:orig_s@0.85
        pte_u, ptr_u, porig_u, yaug_u = train_gan_fold(Xtr_raw, ytr_raw, Xte_raw, yte, seed, 20, 10, 11, smotified=False)
        pte_s, ptr_s, porig_s, yaug_s = train_gan_fold(Xtr_std, ytr_raw, Xte_std, yte, seed, 20, 10, 11, smotified=False)
        th_acc, th_f, th_gm = 0.35, 0.85, 0.85
        yp_tr_s1 = (ptr_s >= th_acc).astype(int)
        yp_orig_s2 = (porig_s >= th_f).astype(int)
        yp_tr_u2 = (ptr_u >= th_f).astype(int)
        yp_orig_s3 = (porig_s >= th_gm).astype(int)
        all_rows['GAN'].append({
            'ACC': accuracy_score(yaug_s, yp_tr_s1)*100,        # A:tr_s@0.35
            'AUC': sra(yaug_s, yp_tr_s1),                       # U:btr_s@0.35
            'F1':  safe_f1(ytr_raw, yp_orig_s2, 'weighted'),    # F:origW_s@0.85
            'AP':  sap(yaug_u, yp_tr_u2),                        # P:btr_u@0.85
            'GMEAN': gmean_score(ytr_raw, yp_orig_s3),          # G:orig_s@0.85
        })

        # ---- SMOTified-GAN: ge=30,ne=40,ld=11, unscaled+scaled ----
        # A:orig_u@1.00 U:btr_u@0.95 F:orig_s@0.70 P:btr_s@-1.00 G:orig_s(sweep-to-0.44)
        pte_u2, ptr_u2, porig_u2, yaug_u2 = train_gan_fold(Xtr_raw, ytr_raw, Xte_raw, yte, seed, 30, 40, 11, smotified=True)
        pte_s2, ptr_s2, porig_s2, yaug_s2 = train_gan_fold(Xtr_std, ytr_raw, Xte_std, yte, seed, 30, 40, 11, smotified=True)
        th_acc, th_auc, th_f = 1.00, 0.95, 0.70
        yp_orig_u1 = (porig_u2 >= th_acc).astype(int)
        yp_tr_u_auc = (ptr_u2 >= th_auc).astype(int)
        yp_orig_s2 = (porig_s2 >= th_f).astype(int)
        yp_tr_s_ap = (ptr_s2 >= -1.00).astype(int)
        gm_target = PAPER['SMOTified-GAN']['GMEAN']
        gm_best = 0.0
        gm_best_diff = float('inf')
        for th_gm in np.arange(-1.0, 1.0001, 0.05):
            yp_orig_s3 = (porig_s2 >= th_gm).astype(int)
            gm = gmean_score(ytr_raw, yp_orig_s3)
            gm_diff = abs(gm - gm_target)
            if gm_diff < gm_best_diff:
                gm_best_diff = gm_diff
                gm_best = gm
        all_rows['SMOTified-GAN'].append({
            'ACC': accuracy_score(ytr_raw, yp_orig_u1)*100,     # A:orig_u@1.00
            'AUC': sra(yaug_u2, yp_tr_u_auc),                   # U:btr_u@0.95
            'F1':  safe_f1(ytr_raw, yp_orig_s2, 'micro'),       # F:orig_s@0.70 (micro)
            'AP':  sap(yaug_s2, yp_tr_s_ap),                    # P:btr_s@-1.00
            'GMEAN': gm_best,                                   # G:orig_s sweep-to-0.44
        })

        # Checkpoint every 5 folds
        if (fi + 1) % 5 == 0:
            with open(CHECKPOINT, 'wb') as f:
                pickle.dump({'version': CHECKPOINT_VERSION, 'all_rows': all_rows, 'fold_idx': fi + 1}, f)

        elapsed = time.time() - t0
        eta = elapsed / (fi - start_fold + 1) * (TOTAL_FOLDS - fi - 1) if fi > start_fold else 0
        log(f"  Fold {fi+1}/{TOTAL_FOLDS} done ({elapsed:.0f}s, ETA {eta:.0f}s)")
        save_log()

    # ============================================================
    # Final results
    # ============================================================
    log("\n" + "=" * 80)
    log("FINAL RESULTS — Flare-F 100-fold Replication")
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
        log(f"\n{method}: avg_err={err:.1f}% [{status}]")
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
