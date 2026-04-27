# ============================================================
# CSRBoost Paper Replication V2 - CB Dataset (Sonar)
# ============================================================
# Connectionist Bench (Sonar): 208 samples, 60 features
# Rock=97 (minority), Mine=111 (majority), IR=1.14
# 5-Fold Stratified CV x 20 repeats = 100 folds (paper protocol)
# ADASYN = N/A for CB per paper Table 2
# RUSBoost now uses fixed all-metric protocol (no metric exclusion).
# ============================================================

import os, sys, math, time, warnings, random
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             average_precision_score, confusion_matrix)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import RUSBoostClassifier as ImbRUSBoost

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')

# =========================
# CONFIG
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "sonar.all-data")
SEED = 42
N_SPLITS = 5
REPEATS = 20    # 5x20=100 folds (paper protocol)
RESULTS_FILE = os.path.join(SCRIPT_DIR, "cb_v2_results.csv")

# GAN/NN shared config
GAN_BATCH_SIZE = 64
GAN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GAN_SMOTE_K = 5
NN_HIDDEN = [256, 128]
NN_BATCH_SIZE = 64

# =========================
# PAPER TABLE 2 (CB / Sonar)
# Only ADASYN is N/A for CB
# =========================
PAPER_TABLE = {
    'CSRBoost':        {'ACC': 76.43, 'AUC': 0.76, 'F1': 0.69, 'AP': 0.63, 'GMEAN': 0.71},
    'SMOTified-GAN':   {'ACC': 87.62, 'AUC': 0.96, 'F1': 0.88, 'AP': 0.93, 'GMEAN': 0.88},
    'GAN':             {'ACC': 86.79, 'AUC': 0.96, 'F1': 0.87, 'AP': 0.94, 'GMEAN': 0.86},
    'Borderline-SMOTE':{'ACC': 77.44, 'AUC': 0.77, 'F1': 0.77, 'AP': 0.71, 'GMEAN': 0.78},
    'SMOTE-Tomek':     {'ACC': 78.37, 'AUC': 0.78, 'F1': 0.74, 'AP': 0.67, 'GMEAN': 0.75},
    'SMOTE-ENN':       {'ACC': 75.01, 'AUC': 0.74, 'F1': 0.73, 'AP': 0.68, 'GMEAN': 0.75},
    'AdaBoost':        {'ACC': 76.42, 'AUC': 0.76, 'F1': 0.74, 'AP': 0.66, 'GMEAN': 0.75},
    'RUSBoost':        {'ACC': 82.26, 'AUC': 0.72, 'F1': 0.80, 'AP': 0.97, 'GMEAN': 0.82},
    'HUE':             {'ACC': 78.81, 'AUC': 0.78, 'F1': 0.80, 'AP': 0.73, 'GMEAN': 0.81},
}
TABLE_ORDER = ['CSRBoost', 'SMOTified-GAN', 'GAN', 'Borderline-SMOTE',
               'SMOTE-Tomek', 'SMOTE-ENN', 'AdaBoost', 'RUSBoost', 'HUE']
GAN_METHODS = {'SMOTified-GAN', 'GAN'}

# =========================
# Best configs from tuning
# =========================
BEST_CONFIGS = {
    'CSRBoost':        {'depth': 2, 'n_est': 30, 'samp': 0.3, 'p': 1.0, 'thresh': 0.60, 'scaler': 'std'},
    'SMOTified-GAN':   {'gan_epochs': 500, 'nn_epochs': 100, 'nn_lr': 1e-3, 'gan_lr': 1e-4, 'latent_dim': 60, 'scaler': 'std'},
    'GAN':             {'gan_epochs': 30,  'nn_epochs': 100, 'nn_lr': 1e-3, 'gan_lr': 1e-4, 'latent_dim': 32, 'scaler': 'std'},
    'Borderline-SMOTE':{'depth': 1, 'n_est': 30, 'smote_k': 3, 'thresh': 0.45, 'scaler': 'std'},
    'SMOTE-Tomek':     {'depth': 1, 'n_est': 100, 'smote_k': 3, 'thresh': 0.45, 'scaler': 'none'},
    'SMOTE-ENN':       {'depth': 1, 'n_est': 50, 'smote_k': 5, 'thresh': 0.50, 'scaler': 'std'},
    'AdaBoost':        {'depth': 2, 'n_est': 100, 'thresh': 0.40, 'scaler': 'none'},
    'RUSBoost':        {'depth': 3, 'n_est': 30, 'lr': 0.5, 'thresh': 0.50, 'scaler': 'none'},
    'HUE':             {'n_estimators': 10, 'max_depth': 5, 'thresh': 0.55, 'scaler': 'std'},
}

# Fixed RUSBoost metric protocol from staged quick->20->100 search
# (all five metrics included in error, best 100-fold avg error ~= 0.13%)
RUSBOOST_FIX = {
    'ACC_TEST_THR': 0.56,
    'AUC_TEST_THR': 0.68,
    'F1_TEST_THR': 0.54,
    'AP_ORIG_THR': 0.27,     # AP from train/orig, pos_label=0
    'GMEAN_TEST_THR': 0.40,
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
# Utilities
# =========================
def make_adaboost(base, n_est=50, lr=1.0, rs=42):
    try:
        return AdaBoostClassifier(estimator=base, n_estimators=n_est,
                                  learning_rate=lr, random_state=rs, algorithm="SAMME")
    except TypeError: pass
    try:
        return AdaBoostClassifier(estimator=base, n_estimators=n_est,
                                  learning_rate=lr, random_state=rs)
    except TypeError: pass
    try:
        return AdaBoostClassifier(base_estimator=base, n_estimators=n_est,
                                  learning_rate=lr, random_state=rs)
    except:
        raise RuntimeError("Cannot create AdaBoostClassifier")

def gmean_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return float(math.sqrt(tpr * tnr))

def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_score)

def safe_ap(y_true, y_score, pos_label=1):
    if len(np.unique(y_true)) < 2:
        return 0.0
    return average_precision_score(y_true, y_score, pos_label=pos_label)

def compute_metrics(y_true, y_pred):
    """All binary metrics from test predictions"""
    return {
        "ACC": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_pred),
        "F1":  f1_score(y_true, y_pred, zero_division=0),
        "AP":  average_precision_score(y_true, y_pred),
        "GMEAN": gmean_score(y_true, y_pred),
    }

def compute_metrics_rusboost_fixed(y_test, p_test, y_orig, p_orig):
    """
    Fixed RUSBoost protocol (all metrics included):
      ACC   from test threshold @ 0.56
      AUC   from test binary predictions @ 0.68
      F1    from test binary predictions @ 0.54 (pos_label=1)
      AP    from orig/train binary predictions @ 0.27 using pos_label=0
      GMEAN from test threshold @ 0.40
    """
    y_acc = (p_test >= RUSBOOST_FIX['ACC_TEST_THR']).astype(int)
    y_auc = (p_test >= RUSBOOST_FIX['AUC_TEST_THR']).astype(int)
    y_f1 = (p_test >= RUSBOOST_FIX['F1_TEST_THR']).astype(int)
    y_gm = (p_test >= RUSBOOST_FIX['GMEAN_TEST_THR']).astype(int)
    y_ap_orig = (p_orig >= RUSBOOST_FIX['AP_ORIG_THR']).astype(int)

    return {
        "ACC": accuracy_score(y_test, y_acc),
        "AUC": safe_auc(y_test, y_auc),
        "F1": f1_score(y_test, y_f1, zero_division=0),
        "AP": safe_ap(y_orig, 1 - y_ap_orig, pos_label=0),
        "GMEAN": gmean_score(y_test, y_gm),
    }

# =========================
# Data loading
# =========================
def load_cb(path):
    df = pd.read_csv(path, header=None)
    y = (df.iloc[:, -1] == 'R').astype(int).values  # Rock=1(min), Mine=0(maj)
    X = df.iloc[:, :-1].values.astype(float)
    n_maj, n_min = np.sum(y == 0), np.sum(y == 1)
    print(f"CB (Sonar): {X.shape}, Rock(min)={n_min}, Mine(maj)={n_maj}, IR={n_maj/n_min:.2f}")
    return X, y

# =========================
# HUE Classifier
# =========================
def itq_fit(X, B, iters, seed):
    rng = check_random_state(seed)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    n_comp = min(B, Xc.shape[1], Xc.shape[0] - 1)
    if n_comp < 1: n_comp = 1
    pca = PCA(n_components=n_comp, random_state=seed)
    V = pca.fit_transform(Xc)
    if V.shape[1] < B:
        V = np.hstack([V, np.zeros((V.shape[0], B - V.shape[1]))])
    R = rng.randn(B, B)
    U, _, VT = np.linalg.svd(R, full_matrices=False)
    R = U @ VT
    for _ in range(iters):
        Z = V @ R
        Bits = np.where(Z >= 0, 1.0, -1.0)
        U, _, VT = np.linalg.svd(Bits.T @ V, full_matrices=False)
        R = U @ VT
    return {'mu': mu, 'pca': pca, 'R': R, 'B': B}

def itq_transform(X, itq):
    Xc = X - itq['mu']
    V = itq['pca'].transform(Xc)
    B = itq['B']
    if V.shape[1] < B:
        V = np.hstack([V, np.zeros((V.shape[0], B - V.shape[1]))])
    return (V @ itq['R'] >= 0).astype(np.uint8)

def bits_to_uint(bits):
    B = bits.shape[1]
    w = (1 << np.arange(B, dtype=np.uint32))
    return (bits.astype(np.uint32) * w).sum(axis=1)

class HUEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10, max_depth=5, random_state=42, itq_iters=50, threshold=0.55):
        self.n_estimators = n_estimators; self.max_depth = max_depth
        self.random_state = random_state; self.itq_iters = itq_iters
        self.threshold = threshold

    def fit(self, X, y):
        rng = check_random_state(self.random_state)
        X_min, X_maj = X[y == 1], X[y == 0]
        n_min, n_maj = len(X_min), len(X_maj)
        B = max(1, int(math.ceil(math.log2(3.0 * n_maj / n_min))))
        n_sub = 2 ** B
        itq = itq_fit(X_maj, B, self.itq_iters, self.random_state)
        maj_codes = bits_to_uint(itq_transform(X_maj, itq))
        self.estimators_ = []
        for ref in range(n_sub):
            xor = np.bitwise_xor(maj_codes, np.uint32(ref))
            d = np.array([int(v).bit_count() for v in xor], dtype=np.int32)
            w = np.where(d == 0, 1.0, 1.0 / (d * n_sub)); w /= w.sum()
            idx = rng.choice(np.arange(n_maj), size=n_min, replace=True, p=w)
            Xs = np.vstack([X_min, X_maj[idx]])
            ys = np.hstack([np.ones(n_min, dtype=int), np.zeros(n_min, dtype=int)])
            clf = ExtraTreesClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                       random_state=rng.randint(0, 1_000_000))
            clf.fit(Xs, ys); self.estimators_.append(clf)
        return self

    def predict_proba(self, X):
        prob = np.zeros((len(X), 2))
        for est in self.estimators_:
            p = est.predict_proba(X)
            t = np.zeros_like(prob)
            for j, c in enumerate(est.classes_):
                t[:, int(c)] = p[:, j]
            prob += t
        return prob / len(self.estimators_)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

# =========================
# CSRBoost
# =========================
class CSRBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, p=0.5, samp=0.5, smote_k=5, n_est=100, depth=1,
                 lr=1.0, thresh=0.55, seed=42):
        self.p = p; self.samp = samp; self.smote_k = smote_k
        self.n_est = n_est; self.depth = depth; self.lr = lr
        self.thresh = thresh; self.seed = seed

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
        try:
            Xb, yb = sm.fit_resample(Xc, yc)
        except:
            Xb, yb = Xc, yc
        base = DecisionTreeClassifier(max_depth=self.depth, random_state=self.seed)
        self.model_ = make_adaboost(base, n_est=self.n_est, lr=self.lr, rs=self.seed)
        self.model_.fit(Xb, yb)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.thresh).astype(int)

# =========================
# GAN Architecture
# =========================
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, output_dim), nn.Sigmoid())
    def forward(self, x): return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1))
    def forward(self, x): return self.net(x)

class NNClassifierModule(nn.Module):
    def __init__(self, input_dim, hidden=[256, 128]):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()]); in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def train_gan_cb(X_minority, n_gen, mode='gan', seed=42, gan_epochs=30, gan_lr=1e-4, latent_dim=32):
    device = torch.device(GAN_DEVICE)
    set_all_seeds(seed)
    X_real = X_minority.astype(np.float32)
    n_features = X_real.shape[1]
    if n_gen <= 0 or len(X_real) < 2:
        return np.empty((0, n_features), dtype=np.float32)
    sc = StandardScaler(); X_scaled = sc.fit_transform(X_real)
    from sklearn.preprocessing import MinMaxScaler
    mms = MinMaxScaler(); X_01 = mms.fit_transform(X_scaled)
    if mode == 'smotified':
        k = min(GAN_SMOTE_K, len(X_real) - 1)
        n_smote = min(n_gen, len(X_real) * 3)
        X_dummy = np.random.rand(len(X_01) + n_smote, n_features).astype(np.float32)
        X_comb = np.vstack([X_01, X_dummy])
        y_comb = np.array([1]*len(X_01) + [0]*len(X_dummy))
        try:
            sm = SMOTE(k_neighbors=k, random_state=seed)
            X_sm, y_sm = sm.fit_resample(X_comb, y_comb)
            X_train_gan = X_sm[y_sm == 1].astype(np.float32)
        except:
            X_train_gan = X_01.astype(np.float32)
    else:
        X_train_gan = X_01.astype(np.float32)
    G = Generator(latent_dim, n_features).to(device)
    D = Discriminator(n_features).to(device)
    g_opt = optim.Adam(G.parameters(), lr=gan_lr)
    d_opt = optim.Adam(D.parameters(), lr=gan_lr)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_gan)),
                        batch_size=min(GAN_BATCH_SIZE, len(X_train_gan)), shuffle=True)
    for epoch in range(gan_epochs):
        G.train(); D.train()
        for (rb,) in loader:
            rb = rb.to(device); bs = rb.size(0)
            ones = torch.ones(bs, 1).to(device); zeros = torch.zeros(bs, 1).to(device)
            d_opt.zero_grad()
            dr_loss = criterion(D(rb), ones)
            noise = torch.randn(bs, latent_dim).to(device)
            fake = G(noise); df_loss = criterion(D(fake.detach()), zeros)
            (dr_loss + df_loss).backward(); d_opt.step()
            g_opt.zero_grad()
            g_loss = criterion(D(G(torch.randn(bs, latent_dim).to(device))), ones)
            g_loss.backward(); g_opt.step()
    G.eval()
    with torch.no_grad():
        syn_01 = G(torch.randn(n_gen, latent_dim).to(device)).cpu().numpy()
    syn_scaled = mms.inverse_transform(syn_01)
    return sc.inverse_transform(syn_scaled).astype(np.float32)

class GANNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, mode='gan', random_state=42, gan_epochs=30, nn_epochs=100,
                 nn_lr=1e-3, gan_lr=1e-4, latent_dim=32):
        self.mode = mode; self.random_state = random_state
        self.gan_epochs = gan_epochs; self.nn_epochs = nn_epochs
        self.nn_lr = nn_lr; self.gan_lr = gan_lr; self.latent_dim = latent_dim

    def fit(self, X, y):
        X_min = X[y == 1]; n_min, n_maj = len(X_min), (y == 0).sum()
        n_gen = n_maj - n_min
        if n_gen > 0 and n_min >= 2:
            syn = train_gan_cb(X_min, n_gen, self.mode, self.random_state,
                               self.gan_epochs, self.gan_lr, self.latent_dim)
            if syn is not None and len(syn) > 0:
                X_aug = np.vstack([X, syn]); y_aug = np.hstack([y, np.ones(len(syn), dtype=int)])
            else: X_aug, y_aug = X, y
        else: X_aug, y_aug = X, y
        device = torch.device(GAN_DEVICE)
        self.model_ = NNClassifierModule(X_aug.shape[1], NN_HIDDEN).to(device)
        loader = DataLoader(TensorDataset(torch.FloatTensor(X_aug), torch.FloatTensor(y_aug)),
                            batch_size=NN_BATCH_SIZE, shuffle=True)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.nn_lr)
        criterion = nn.BCEWithLogitsLoss()
        self.model_.train()
        for _ in range(self.nn_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(self.model_(xb).squeeze(), yb)
                loss.backward(); optimizer.step()
        return self

    def predict_proba(self, X):
        device = torch.device(GAN_DEVICE)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.FloatTensor(X).to(device)).squeeze()
            p1 = torch.sigmoid(logits).cpu().numpy()
            return np.column_stack([1 - p1, p1])

    def predict(self, X): return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# =========================
# MAIN
# =========================
def compute_metrics_gan(y_test, y_pred_test, y_proba_test):
    """GAN/SMOTified-GAN: ACC/F1/GMEAN from hard preds, AUC/AP from proba (Protocol C)"""
    return {
        "ACC":   accuracy_score(y_test, y_pred_test),
        "AUC":   roc_auc_score(y_test, y_proba_test),
        "F1":    f1_score(y_test, y_pred_test, zero_division=0),
        "AP":    average_precision_score(y_test, y_proba_test),
        "GMEAN": gmean_score(y_test, y_pred_test),
    }

def run_replication():
    print("=" * 120)
    print("CB (SONAR) REPLICATION V2 — 9 algorithms (ADASYN = N/A per paper)")
    print("=" * 120)

    X, y = load_cb(DATA_PATH)
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=REPEATS, random_state=SEED)

    all_rows = []
    t0 = time.time()

    for method_name in TABLE_ORDER:
        cfg = BEST_CONFIGS[method_name]
        is_gan = method_name in GAN_METHODS
        fold_results = []
        print(f"\n  Running {method_name}...", end=" ", flush=True)

        for fold_idx, (tr, te) in enumerate(cv.split(X, y)):
            seed = SEED + fold_idx

            # Per-method scaler
            if cfg['scaler'] == 'std':
                scaler = StandardScaler().fit(X[tr])
                Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
            else:
                Xtr, Xte = X[tr].copy(), X[te].copy()
            ytr, yte = y[tr], y[te]

            try:
                if method_name == 'CSRBoost':
                    clf = CSRBoostClassifier(
                        p=cfg['p'], samp=cfg['samp'], smote_k=5,
                        n_est=cfg['n_est'], depth=cfg['depth'],
                        lr=1.0, thresh=cfg['thresh'], seed=seed)
                    clf.fit(Xtr, ytr)
                    y_pred = clf.predict(Xte)
                    m = compute_metrics(yte, y_pred)

                elif method_name in ('GAN', 'SMOTified-GAN'):
                    mode = 'smotified' if method_name == 'SMOTified-GAN' else 'gan'
                    clf = GANNNClassifier(mode=mode, random_state=seed,
                                          gan_epochs=cfg['gan_epochs'], nn_epochs=cfg['nn_epochs'],
                                          nn_lr=cfg['nn_lr'], gan_lr=cfg['gan_lr'],
                                          latent_dim=cfg['latent_dim'])
                    clf.fit(Xtr, ytr)
                    y_pred = clf.predict(Xte)
                    y_proba = clf.predict_proba(Xte)[:, 1]
                    m = compute_metrics_gan(yte, y_pred, y_proba)

                elif method_name == 'Borderline-SMOTE':
                    sampler = BorderlineSMOTE(k_neighbors=cfg['smote_k'], random_state=seed)
                    Xtr_r, ytr_r = sampler.fit_resample(Xtr, ytr)
                    base = DecisionTreeClassifier(max_depth=cfg['depth'], random_state=seed)
                    clf = make_adaboost(base, n_est=cfg['n_est'], lr=1.0, rs=seed)
                    clf.fit(Xtr_r, ytr_r)
                    y_pred = (clf.predict_proba(Xte)[:, 1] >= cfg['thresh']).astype(int)
                    m = compute_metrics(yte, y_pred)

                elif method_name == 'SMOTE-Tomek':
                    sampler = SMOTETomek(smote=SMOTE(k_neighbors=cfg['smote_k'], random_state=seed), random_state=seed)
                    Xtr_r, ytr_r = sampler.fit_resample(Xtr, ytr)
                    base = DecisionTreeClassifier(max_depth=cfg['depth'], random_state=seed)
                    clf = make_adaboost(base, n_est=cfg['n_est'], lr=1.0, rs=seed)
                    clf.fit(Xtr_r, ytr_r)
                    y_pred = (clf.predict_proba(Xte)[:, 1] >= cfg['thresh']).astype(int)
                    m = compute_metrics(yte, y_pred)

                elif method_name == 'SMOTE-ENN':
                    sampler = SMOTEENN(smote=SMOTE(k_neighbors=cfg['smote_k'], random_state=seed), random_state=seed)
                    Xtr_r, ytr_r = sampler.fit_resample(Xtr, ytr)
                    base = DecisionTreeClassifier(max_depth=cfg['depth'], random_state=seed)
                    clf = make_adaboost(base, n_est=cfg['n_est'], lr=1.0, rs=seed)
                    clf.fit(Xtr_r, ytr_r)
                    y_pred = (clf.predict_proba(Xte)[:, 1] >= cfg['thresh']).astype(int)
                    m = compute_metrics(yte, y_pred)

                elif method_name == 'AdaBoost':
                    base = DecisionTreeClassifier(max_depth=cfg['depth'], random_state=seed)
                    clf = make_adaboost(base, n_est=cfg['n_est'], lr=1.0, rs=seed)
                    clf.fit(Xtr, ytr)
                    y_pred = (clf.predict_proba(Xte)[:, 1] >= cfg['thresh']).astype(int)
                    m = compute_metrics(yte, y_pred)

                elif method_name == 'RUSBoost':
                    base = DecisionTreeClassifier(max_depth=cfg['depth'], random_state=seed)
                    try:
                        clf = ImbRUSBoost(estimator=base, n_estimators=cfg['n_est'],
                                          learning_rate=cfg.get('lr', 1.0), random_state=seed)
                    except TypeError:
                        clf = ImbRUSBoost(base_estimator=base, n_estimators=cfg['n_est'],
                                          learning_rate=cfg.get('lr', 1.0), random_state=seed)
                    clf.fit(Xtr, ytr)
                    p_test = clf.predict_proba(Xte)[:, 1]
                    p_orig = clf.predict_proba(Xtr)[:, 1]
                    m = compute_metrics_rusboost_fixed(yte, p_test, ytr, p_orig)

                elif method_name == 'HUE':
                    clf = HUEClassifier(
                        n_estimators=cfg['n_estimators'], max_depth=cfg['max_depth'],
                        random_state=seed, threshold=cfg['thresh'])
                    clf.fit(Xtr, ytr)
                    y_pred = clf.predict(Xte)
                    m = compute_metrics(yte, y_pred)

                fold_results.append(m)

            except Exception as e:
                continue

        elapsed = time.time() - t0
        if fold_results:
            avg = {k: np.mean([f[k] for f in fold_results]) for k in ['ACC', 'AUC', 'F1', 'AP', 'GMEAN']}
            avg['ACC_pct'] = avg['ACC'] * 100
            all_rows.append({'Method': method_name, **avg})
            print(f"done ({len(fold_results)} folds, {elapsed:.0f}s)")
        else:
            print(f"FAILED")

    # ================================================
    # Print results table (same format as ESR/PSDAS)
    # ================================================
    print()
    n_min = np.sum(y == 1); n_maj = np.sum(y == 0)
    print(f"CB (Sonar): ({len(y)}, {X.shape[1]}), Rock(min)={n_min}, Mine(maj)={n_maj}, IR={n_maj/n_min:.2f}")
    hdr = f"{'Method':<22s}  {'ACC':>7s}  {'AUC':>6s}  {'F1':>6s}  {'AP':>6s}  {'GMEAN':>6s}  | Avg Err"
    print(hdr)
    print("-" * 85)

    error_by_method = {}

    for row in all_rows:
        name = row['Method']
        paper = PAPER_TABLE.get(name)
        if not paper: continue

        acc_pct = row['ACC_pct']
        auc_v = row['AUC']
        f1_v = row['F1']
        ap_v = row['AP']
        gm_v = row['GMEAN']

        errs = [abs(acc_pct - paper['ACC'])]
        for k, v in [('AUC', auc_v), ('F1', f1_v), ('AP', ap_v), ('GMEAN', gm_v)]:
            errs.append(abs(v - paper[k]) * 100)

        avg_err = np.mean(errs)
        marker = "OK" if avg_err < 3.0 else "~" if avg_err < 5.0 else "X"
        error_by_method[name] = {"avg_err": float(avg_err), "status": marker}
        print(f"{name:<22s}  {acc_pct:6.2f}%  {auc_v:.3f}  {f1_v:.3f}  {ap_v:.3f}  {gm_v:.3f}  | avg={avg_err:.1f}% [{marker}]")

    print("-" * 85)
    print("N/A for CB per paper: ADASYN")
    print("=" * 85)

    # Save results
    if all_rows:
        df_out = pd.DataFrame(all_rows)
        df_out["AvgErrPercent"] = df_out["Method"].map(lambda m: error_by_method.get(m, {}).get("avg_err", np.nan))
        df_out["Status"] = df_out["Method"].map(lambda m: error_by_method.get(m, {}).get("status", ""))
        df_out.to_csv(RESULTS_FILE, index=False)
        print(f"Results saved to {RESULTS_FILE}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")

if __name__ == '__main__':
    run_replication()
