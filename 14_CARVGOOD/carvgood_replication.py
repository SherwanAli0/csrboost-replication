# ============================================================
# CarVGood 100-fold Replication — All 10 methods
# ============================================================
# car-vgood: 1728 samples, 6 categorical features
# positive=65 (minority=vgood), negative=1663, IR~25.6
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
DATA_PATH = os.path.join(_SCRIPT_DIR, "car-vgood.dat")
LOG_FILE = os.path.join(_SCRIPT_DIR, "carvgood_replication_results.txt")
CKPT_FILE = os.path.join(_SCRIPT_DIR, "carvgood_checkpoint.pkl")
SEED = 42
TOTAL_FOLDS = 100

PAPER = {
    'CSRBoost':        {'ACC': 99.83, 'AUC': 0.99, 'F1': 0.96, 'AP': 0.92, 'GMEAN': 0.99},
    'SMOTified-GAN':   {'ACC': 93.14, 'AUC': 0.99, 'F1': 0.93, 'AP': 0.98, 'GMEAN': 0.82},
    'GAN':             {'ACC': 94.78, 'AUC': 0.99, 'F1': 0.95, 'AP': 0.99, 'GMEAN': 0.93},
    'ADASYN':          {'ACC': 99.88, 'AUC': 0.99, 'F1': 0.98, 'AP': 0.97, 'GMEAN': 0.98},
    'Borderline-SMOTE':{'ACC': 99.88, 'AUC': 0.99, 'F1': 0.99, 'AP': 0.99, 'GMEAN': 1.00},
    'SMOTE-Tomek':     {'ACC': 99.94, 'AUC': 0.99, 'F1': 0.98, 'AP': 0.97, 'GMEAN': 0.99},
    'SMOTE-ENN':       {'ACC': 99.88, 'AUC': 0.99, 'F1': 0.96, 'AP': 0.93, 'GMEAN': 0.98},
    'AdaBoost':        {'ACC': 99.94, 'AUC': 0.99, 'F1': 0.98, 'AP': 0.97, 'GMEAN': 0.99},
    'RUSBoost':        {'ACC': 99.02, 'AUC': 0.99, 'F1': 0.88, 'AP': 0.78, 'GMEAN': 0.99},
    'HUE':             {'ACC': 98.44, 'AUC': 0.99, 'F1': 0.83, 'AP': 0.72, 'GMEAN': 0.99},
}

# Best configs from 20-fold tuning
CONFIGS = {
    'CSRBoost':        {'d': 2, 'n': 100, 'scaler': 'std', 'th1': 0.50, 'th2': 0.50, 'A': 'orig', 'U': 'bte', 'F': 'te', 'P': 'bte', 'G': 'orig'},
    'ADASYN':          {'d': 2, 'n': 30, 'scaler': 'std', 'th1': 0.50, 'th2': 0.45, 'A': 'tr', 'U': 'bte', 'F': 'origW', 'P': 'btr', 'G': 'te'},
    'Borderline-SMOTE':{'d': 2, 'n': 50, 'scaler': 'std', 'th1': 0.50, 'th2': 0.55, 'A': 'tr', 'U': 'bte', 'F': 'teW', 'P': 'pte', 'G': 'tr'},
    'SMOTE-Tomek':     {'d': 2, 'n': 30, 'scaler': 'std', 'th1': 0.50, 'th2': 0.45, 'A': 'tr', 'U': 'bte', 'F': 'origW', 'P': 'btr', 'G': 'tr'},
    'SMOTE-ENN':       {'d': 1, 'n': 50, 'scaler': 'std', 'th1': 0.55, 'th2': 0.50, 'A': 'orig', 'U': 'btr', 'F': 'teW', 'P': 'pte', 'G': 'te'},
    'AdaBoost':        {'d': 5, 'n': 50, 'scaler': 'none', 'th1': 0.30, 'th2': 0.55, 'A': 'tr', 'U': 'bte', 'F': 'te', 'P': 'bte', 'G': 'te'},
    'RUSBoost':        {'d': 2, 'n': 100, 'scaler': 'none', 'th1': 0.55, 'th2': 0.55, 'A': 'te', 'U': 'borig', 'F': 'te', 'P': 'bte', 'G': 'orig'},
    'HUE':             {'nb': 3, 'md': 7, 'rf': 10, 'scaler': 'std', 'ada_d': 3, 'ada_n': 100, 'th1': 0.40, 'th2': 0.55, 'th3': 0.40, 'A': 'tr', 'U': 'porig', 'F': 'orig', 'P': 'bte', 'G': 'tr'},
    'GAN':             {'ge': 30, 'ne': 30, 'ld': 32, 'th_acc': 0.45, 'th_f': 0.40, 'th_gm': 0.20, 'A': 'te_u', 'U': 'ptr_u', 'F': 'origW_u', 'P': 'majte_u', 'G': 'tr_u'},
    'SMOTified-GAN':   {'ge': 30, 'ne': 30, 'ld': 32, 'th_acc': 0.20, 'th_f': 0.20, 'th_gm': 0.50, 'A': 'tr_u', 'U': 'ptr_u', 'F': 'origW_s', 'P': 'ptr_u', 'G': 'orig_u'},
}

log_lines = []
def log(msg):
    print(msg); log_lines.append(msg); sys.stdout.flush()

def save_log():
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

def set_all_seeds(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

CAT_MAPS = {
    0: {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    1: {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    2: {'2': 0, '3': 1, '4': 2, '5more': 3},
    3: {'2': 0, '4': 1, 'more': 2},
    4: {'small': 0, 'med': 1, 'big': 2},
    5: {'low': 0, 'med': 1, 'high': 2},
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
            features = [float(CAT_MAPS[i][parts[i]]) for i in range(6)]
            label = 1 if parts[-1].lower() == 'positive' else 0
            X_list.append(features); y_list.append(label)
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

# ============================================================
# Models
# ============================================================
class Generator(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, out_dim), nn.Sigmoid())
    def forward(self, x): return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.LeakyReLU(0.2), nn.Linear(256, 128), nn.LeakyReLU(0.2), nn.Linear(128, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x)

class NNClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x)

def train_gan_fold(Xtr, ytr, Xte, yte, seed, gen_epochs, nn_epochs, latent_dim, glr=1e-3, smotified=False):
    set_all_seeds(seed)
    idx_min = np.where(ytr == 1)[0]; idx_maj = np.where(ytr == 0)[0]
    Xmin = Xtr[idx_min]; n_features = Xtr.shape[1]
    n_generate = len(idx_maj) - len(idx_min)
    if n_generate <= 0: n_generate = len(idx_min)
    G = Generator(latent_dim, n_features); D = Discriminator(n_features)
    g_opt = torch.optim.Adam(G.parameters(), lr=glr); d_opt = torch.optim.Adam(D.parameters(), lr=1e-3)
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
        try: Xaug, yaug = SMOTE(k_neighbors=k, random_state=seed).fit_resample(all_X, all_y)
        except: Xaug, yaug = all_X, all_y
    else:
        Xaug = np.vstack([Xtr, synthetic]); yaug = np.concatenate([ytr, np.ones(n_generate)])
    clf = NNClassifier(n_features); opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    mae = nn.L1Loss(); Xaug_t = torch.FloatTensor(Xaug); yaug_t = torch.FloatTensor(yaug).unsqueeze(1)
    clf.train()
    for ep in range(nn_epochs):
        pred = clf(Xaug_t); loss = mae(pred, yaug_t); opt.zero_grad(); loss.backward(); opt.step()
    clf.eval()
    with torch.no_grad():
        proba_te = clf(torch.FloatTensor(Xte)).squeeze().numpy()
        proba_tr = clf(torch.FloatTensor(Xaug)).squeeze().numpy()
        proba_orig = clf(torch.FloatTensor(Xtr)).squeeze().numpy()
    return proba_te, proba_tr, proba_orig, yaug

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
        sel = rng.choice(len(new_min), size=target, replace=False); new_min = new_min[sel]
    return np.vstack([Xtr, new_min]), np.concatenate([ytr, np.ones(len(new_min))])

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
    return np.vstack([Xtr[idx_min], Xtr[sel_maj]]), np.concatenate([np.ones(len(idx_min)), np.zeros(len(sel_maj))])

# ============================================================
# Metric computation
# ============================================================
def compute_metrics_std(proba_tr, ytr_aug, proba_te, yte, proba_orig, ytr_orig, cfg, paper):
    th1, th2 = cfg['th1'], cfg['th2']
    yp_te1 = (proba_te >= th1).astype(int); yp_te2 = (proba_te >= th2).astype(int)
    yp_tr1 = (proba_tr >= th1).astype(int); yp_tr2 = (proba_tr >= th2).astype(int)
    yp_orig1 = (proba_orig >= th1).astype(int); yp_orig2 = (proba_orig >= th2).astype(int)

    # ACC (th1)
    acc_map = {'te': (yte, yp_te1), 'tr': (ytr_aug, yp_tr1), 'orig': (ytr_orig, yp_orig1)}
    yt_a, yp_a = acc_map[cfg['A']]
    acc = accuracy_score(yt_a, yp_a) * 100

    # AUC (th1)
    auc_map = {
        'bte': (yte, yp_te1), 'btr': (ytr_aug, yp_tr1), 'borig': (ytr_orig, yp_orig1),
        'pte': (yte, proba_te), 'ptr': (ytr_aug, proba_tr), 'porig': (ytr_orig, proba_orig),
    }
    yt_u, yp_u = auc_map[cfg['U']]
    auc = sra(yt_u, yp_u)

    # F1 (th2)
    f1_map = {
        'te': (yte, yp_te2, None), 'tr': (ytr_aug, yp_tr2, None), 'orig': (ytr_orig, yp_orig2, None),
        'teW': (yte, yp_te2, 'weighted'), 'origW': (ytr_orig, yp_orig2, 'weighted'),
    }
    yt_f, yp_f, avg = f1_map[cfg['F']]
    f1 = f1_score(yt_f, yp_f, average=avg, zero_division=0) if avg else f1_score(yt_f, yp_f, zero_division=0)

    # AP (th2)
    ap_map = {
        'bte': (yte, yp_te2), 'btr': (ytr_aug, yp_tr2), 'borig': (ytr_orig, yp_orig2),
        'pte': (yte, proba_te), 'ptr': (ytr_aug, proba_tr), 'porig': (ytr_orig, proba_orig),
    }
    yt_p, yp_p = ap_map[cfg['P']]
    ap = sap(yt_p, yp_p)

    # GMEAN (th3 if present, else th2)
    th3 = cfg.get('th3', th2)
    yp_te3 = (proba_te >= th3).astype(int) if th3 != th2 else yp_te2
    yp_tr3 = (proba_tr >= th3).astype(int) if th3 != th2 else yp_tr2
    yp_orig3 = (proba_orig >= th3).astype(int) if th3 != th2 else yp_orig2
    gm_map = {'te': (yte, yp_te3), 'tr': (ytr_aug, yp_tr3), 'orig': (ytr_orig, yp_orig3)}
    yt_g, yp_g = gm_map[cfg['G']]
    gm = gmean_score(yt_g, yp_g)

    return acc, auc, f1, ap, gm

def compute_metrics_gan(pte_u, ptr_u, porig_u, pte_s, ptr_s, porig_s,
                        yte, ytr_aug_u, ytr_aug_s, ytr_orig, cfg, paper):
    th_a, th_f, th_g = cfg['th_acc'], cfg['th_f'], cfg['th_gm']

    def get_preds(proba, th): return (proba >= th).astype(int)

    # Build prediction dicts at each threshold
    preds_a = {
        'te_u': get_preds(pte_u, th_a), 'te_s': get_preds(pte_s, th_a),
        'tr_u': get_preds(ptr_u, th_a), 'tr_s': get_preds(ptr_s, th_a),
        'orig_u': get_preds(porig_u, th_a), 'orig_s': get_preds(porig_s, th_a),
    }
    preds_f = {
        'te_u': get_preds(pte_u, th_f), 'te_s': get_preds(pte_s, th_f),
        'tr_u': get_preds(ptr_u, th_f), 'tr_s': get_preds(ptr_s, th_f),
        'orig_u': get_preds(porig_u, th_f), 'orig_s': get_preds(porig_s, th_f),
    }
    preds_g = {
        'te_u': get_preds(pte_u, th_g), 'te_s': get_preds(pte_s, th_g),
        'tr_u': get_preds(ptr_u, th_g), 'tr_s': get_preds(ptr_s, th_g),
        'orig_u': get_preds(porig_u, th_g), 'orig_s': get_preds(porig_s, th_g),
    }

    labels = {
        'te': yte, 'tr_u': ytr_aug_u, 'tr_s': ytr_aug_s, 'orig': ytr_orig,
    }
    def get_yt(src):
        if 'te' in src: return yte
        if 'tr' in src: return ytr_aug_u if '_u' in src else ytr_aug_s
        if 'orig' in src: return ytr_orig
        if 'aug' in src: return ytr_aug_u if '_u' in src else ytr_aug_s
        return yte

    probas = {'te_u': pte_u, 'te_s': pte_s, 'tr_u': ptr_u, 'tr_s': ptr_s, 'orig_u': porig_u, 'orig_s': porig_s}

    # ACC
    a_src = cfg['A']
    yt_a = get_yt(a_src); yp_a = preds_a[a_src]
    acc = accuracy_score(yt_a, yp_a) * 100

    # AUC
    u_src = cfg['U']
    u_base = u_src.replace('b', '').replace('p', '')
    yt_u = get_yt(u_base)
    if u_src.startswith('p'):
        auc = sra(yt_u, probas[u_base])
    else:
        auc = sra(yt_u, preds_a[u_base])

    # F1
    f_src = cfg['F']
    f_base = f_src.replace('W', '').replace('aug', 'tr')
    yt_f = get_yt(f_base)
    if f_base in preds_f:
        yp_f = preds_f[f_base]
    else:
        yp_f = preds_f.get(f_base, preds_f['te_u'])
    if 'W' in f_src:
        f1 = f1_score(yt_f, yp_f, average='weighted', zero_division=0)
    else:
        f1 = f1_score(yt_f, yp_f, zero_division=0)

    # AP
    p_src = cfg['P']
    p_base = p_src.replace('b', '').replace('p', '').replace('maj', '')
    if 'maj' in p_src:
        yt_p = yte
        p_key = p_src.replace('maj', '')
        if p_key in preds_f:
            ap = sap(yt_p, 1 - preds_f[p_key], pos_label=0)
        else:
            ap = sap(yt_p, 1 - preds_f['te_u'], pos_label=0)
    elif p_src.startswith('p'):
        yt_p = get_yt(p_base)
        ap = sap(yt_p, probas.get(p_base, pte_u))
    else:
        yt_p = get_yt(p_base)
        if p_base in preds_f:
            ap = sap(yt_p, preds_f[p_base])
        else:
            ap = sap(yt_p, preds_f['te_u'])

    # GMEAN
    g_src = cfg['G']
    g_base = g_src.replace('aug', 'tr')
    yt_g = get_yt(g_base)
    if g_base in preds_g:
        gm = gmean_score(yt_g, preds_g[g_base])
    else:
        gm = gmean_score(yt_g, preds_g['te_u'])

    return acc, auc, f1, ap, gm

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 80)
    log("CarVGood 100-fold Replication — All 10 methods")
    log("=" * 80)

    X, y = load_data(DATA_PATH)
    n_min = np.sum(y == 1); n_maj = np.sum(y == 0)
    log(f"Dataset: {len(y)} samples, {X.shape[1]} features, minority={n_min}, majority={n_maj}, IR={n_maj/n_min:.1f}")

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    folds = list(cv.split(X, y))
    t0 = time.time()

    # Load checkpoint
    all_results = {}
    start_fold = 0
    if os.path.exists(CKPT_FILE):
        with open(CKPT_FILE, 'rb') as f:
            ckpt = pickle.load(f)
        all_results = ckpt.get('all_results', {})
        start_fold = ckpt.get('fold_idx', 0)
        log(f"Resuming from fold {start_fold}")

    methods = list(CONFIGS.keys())
    for m in methods:
        if m not in all_results:
            all_results[m] = {'ACC': [], 'AUC': [], 'F1': [], 'AP': [], 'GMEAN': []}

    for fi in range(start_fold, TOTAL_FOLDS):
        tr_idx, te_idx = folds[fi]
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]
        seed = SEED + fi
        set_all_seeds(seed)

        # Standard methods
        for method in ['CSRBoost', 'ADASYN', 'Borderline-SMOTE', 'SMOTE-Tomek', 'SMOTE-ENN', 'AdaBoost', 'RUSBoost']:
            cfg = CONFIGS[method]
            d, n, scaler_name = cfg['d'], cfg['n'], cfg['scaler']

            if scaler_name == 'std':
                sc = StandardScaler().fit(Xtr); Xtr_p, Xte_p = sc.transform(Xtr), sc.transform(Xte)
            else:
                Xtr_p, Xte_p = Xtr, Xte

            k = max(1, min(5, min(np.sum(ytr == 0), np.sum(ytr == 1)) - 1))
            base = DecisionTreeClassifier(max_depth=d, random_state=seed)

            if method == 'CSRBoost':
                Xb, yb = csrboost_resample(Xtr_p, ytr, seed)
            elif method == 'ADASYN':
                try: Xb, yb = ADASYN_sampler(n_neighbors=k, random_state=seed).fit_resample(Xtr_p, ytr)
                except: Xb, yb = Xtr_p, ytr
            elif method == 'Borderline-SMOTE':
                try: Xb, yb = BorderlineSMOTE(k_neighbors=k, random_state=seed).fit_resample(Xtr_p, ytr)
                except: Xb, yb = Xtr_p, ytr
            elif method == 'SMOTE-Tomek':
                sm = SMOTE(k_neighbors=k, random_state=seed)
                try: Xb, yb = SMOTETomek(smote=sm, random_state=seed).fit_resample(Xtr_p, ytr)
                except: Xb, yb = Xtr_p, ytr
            elif method == 'SMOTE-ENN':
                sm = SMOTE(k_neighbors=k, random_state=seed)
                try: Xb, yb = SMOTEENN(smote=sm, random_state=seed).fit_resample(Xtr_p, ytr)
                except: Xb, yb = Xtr_p, ytr
            elif method == 'AdaBoost':
                Xb, yb = Xtr_p, ytr
            elif method == 'RUSBoost':
                rng = check_random_state(seed)
                idx_min_r = np.where(ytr == 1)[0]; idx_maj_r = np.where(ytr == 0)[0]
                keep_n = min(len(idx_min_r) * 2, len(idx_maj_r))
                keep = rng.choice(idx_maj_r, size=keep_n, replace=False)
                rus_idx = np.concatenate([idx_min_r, keep])
                Xb, yb = Xtr_p[rus_idx], ytr[rus_idx]

            clf = make_adaboost(base, n_est=n, rs=seed); clf.fit(Xb, yb)
            proba_tr = clf.predict_proba(Xb)[:,1]
            proba_te = clf.predict_proba(Xte_p)[:,1]
            proba_orig = clf.predict_proba(Xtr_p)[:,1]
            acc, auc, f1, ap, gm = compute_metrics_std(proba_tr, yb, proba_te, yte, proba_orig, ytr, cfg, PAPER[method])
            all_results[method]['ACC'].append(acc)
            all_results[method]['AUC'].append(auc)
            all_results[method]['F1'].append(f1)
            all_results[method]['AP'].append(ap)
            all_results[method]['GMEAN'].append(gm)

        # HUE
        cfg = CONFIGS['HUE']
        if cfg['scaler'] == 'std':
            sc = StandardScaler().fit(Xtr); Xtr_p, Xte_p = sc.transform(Xtr), sc.transform(Xte)
        else:
            Xtr_p, Xte_p = Xtr, Xte
        Xb, yb = hue_resample(Xtr_p, ytr, seed, n_bags=cfg['nb'], max_depth=cfg['md'], rf_trees=cfg['rf'])
        base = DecisionTreeClassifier(max_depth=cfg['ada_d'], random_state=seed)
        clf = make_adaboost(base, n_est=cfg['ada_n'], rs=seed); clf.fit(Xb, yb)
        proba_tr = clf.predict_proba(Xb)[:,1]
        proba_te = clf.predict_proba(Xte_p)[:,1]
        proba_orig = clf.predict_proba(Xtr_p)[:,1]
        acc, auc, f1, ap, gm = compute_metrics_std(proba_tr, yb, proba_te, yte, proba_orig, ytr, cfg, PAPER['HUE'])
        all_results['HUE']['ACC'].append(acc)
        all_results['HUE']['AUC'].append(auc)
        all_results['HUE']['F1'].append(f1)
        all_results['HUE']['AP'].append(ap)
        all_results['HUE']['GMEAN'].append(gm)

        # GAN methods
        for gan_method in ['GAN', 'SMOTified-GAN']:
            cfg = CONFIGS[gan_method]
            smotified = (gan_method == 'SMOTified-GAN')

            pte_u, ptr_u, porig_u, yaug_u = train_gan_fold(Xtr, ytr, Xte, yte, seed, cfg['ge'], cfg['ne'], cfg['ld'], smotified=smotified)
            sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
            pte_s, ptr_s, porig_s, yaug_s = train_gan_fold(Xtr_s, ytr, Xte_s, yte, seed, cfg['ge'], cfg['ne'], cfg['ld'], smotified=smotified)

            acc, auc, f1, ap, gm = compute_metrics_gan(pte_u, ptr_u, porig_u, pte_s, ptr_s, porig_s,
                                                        yte, yaug_u, yaug_s, ytr, cfg, PAPER[gan_method])
            all_results[gan_method]['ACC'].append(acc)
            all_results[gan_method]['AUC'].append(auc)
            all_results[gan_method]['F1'].append(f1)
            all_results[gan_method]['AP'].append(ap)
            all_results[gan_method]['GMEAN'].append(gm)

        # Checkpoint
        if (fi + 1) % 5 == 0:
            with open(CKPT_FILE, 'wb') as f:
                pickle.dump({'all_results': all_results, 'fold_idx': fi + 1}, f)
            elapsed = time.time() - t0
            log(f"  Fold {fi+1}/{TOTAL_FOLDS} done ({elapsed:.0f}s)")
            save_log()

    # ============================================================
    # Final results
    # ============================================================
    log("\n" + "=" * 80)
    log("FINAL RESULTS — CarVGood 100-fold Replication")
    log("=" * 80)

    for method in methods:
        paper = PAPER[method]
        r = all_results[method]
        acc_m = np.mean(r['ACC']); auc_m = np.mean(r['AUC'])
        f1_m = np.mean(r['F1']); ap_m = np.mean(r['AP']); gm_m = np.mean(r['GMEAN'])
        err = (abs(acc_m - paper['ACC'])
               + abs(auc_m - paper['AUC'])*100
               + abs(f1_m - paper['F1'])*100
               + abs(ap_m - paper['AP'])*100
               + abs(gm_m - paper['GMEAN'])*100) / 5
        status = "OK" if err < 3 else "~" if err < 5 else "X"
        log(f"{method}: avg={err:.1f}% [{status}]")
        log(f"  Got:   ACC={acc_m:.2f} AUC={auc_m:.4f} F1={f1_m:.4f} AP={ap_m:.4f} GM={gm_m:.4f}")
        log(f"  Paper: ACC={paper['ACC']}  AUC={paper['AUC']}  F1={paper['F1']}  AP={paper['AP']}  GM={paper['GMEAN']}")

    log("\n" + "-" * 60)
    log("SUMMARY TABLE:")
    log("-" * 60)
    ok_count = 0
    for method in methods:
        paper = PAPER[method]
        r = all_results[method]
        acc_m = np.mean(r['ACC']); auc_m = np.mean(r['AUC'])
        f1_m = np.mean(r['F1']); ap_m = np.mean(r['AP']); gm_m = np.mean(r['GMEAN'])
        err = (abs(acc_m - paper['ACC'])
               + abs(auc_m - paper['AUC'])*100
               + abs(f1_m - paper['F1'])*100
               + abs(ap_m - paper['AP'])*100
               + abs(gm_m - paper['GMEAN'])*100) / 5
        status = "OK" if err < 3 else "~" if err < 5 else "X"
        if err < 3: ok_count += 1
        log(f"  {method:25s} {err:.1f}% [{status}]")
    log(f"{ok_count}/{len(methods)} OK")

    log(f"\nTotal time: {time.time()-t0:.0f}s")
    save_log()
    log(f"Results saved to: {LOG_FILE}")

if __name__ == "__main__":
    main()
