# ============================================================
# CSRBoost Paper Replication V2 - ESDRP Dataset
# ============================================================
# Early Stage Diabetes Risk Prediction: 520 samples, 16 features
# Class "Negative" = 200 (minority -> 1), "Positive" = 320 (majority -> 0)
# Auto-detect minority class
# 5-Fold Stratified CV x 20 repeats = 100 folds (paper protocol)
# Per-method metric protocols: auc_mode, ap_mode, f1_mode in BEST_CONFIGS
# ============================================================

import os, sys, math, time, warnings, random
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             average_precision_score, confusion_matrix)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
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
DATA_PATH = os.path.join(SCRIPT_DIR, "diabetes_data_upload.csv")
SEED = 42
N_SPLITS = 5
REPEATS = 20    # 5x20=100 folds (paper protocol)
RESULTS_FILE = os.path.join(SCRIPT_DIR, "esdrp_v2_results.csv")

# GAN/NN shared config
GAN_BATCH_SIZE = 64
GAN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GAN_SMOTE_K = 5
NN_HIDDEN = [256, 128]
NN_BATCH_SIZE = 64

# =========================
# PAPER TABLE 2 (ESDRP row)
# =========================
PAPER_TABLE = {
    'CSRBoost':        {'ACC': 97.31, 'AUC': 0.97, 'F1': 0.95, 'AP': 0.92, 'GMEAN': 0.96},
    'SMOTified-GAN':   {'ACC': 92.98, 'AUC': 0.99, 'F1': 0.94, 'AP': 0.88, 'GMEAN': 0.92},
    'GAN':             {'ACC': 91.97, 'AUC': 0.98, 'F1': 0.92, 'AP': 0.95, 'GMEAN': 0.91},
    'ADASYN':          {'ACC': 97.69, 'AUC': 0.98, 'F1': 0.97, 'AP': 0.94, 'GMEAN': 0.98},
    'Borderline-SMOTE':{'ACC': 97.12, 'AUC': 0.97, 'F1': 0.97, 'AP': 0.94, 'GMEAN': 0.98},
    'SMOTE-Tomek':     {'ACC': 97.50, 'AUC': 0.97, 'F1': 0.97, 'AP': 0.94, 'GMEAN': 0.98},
    'SMOTE-ENN':       {'ACC': 93.27, 'AUC': 0.94, 'F1': 0.93, 'AP': 0.92, 'GMEAN': 0.95},
    'AdaBoost':        {'ACC': 97.50, 'AUC': 0.97, 'F1': 0.97, 'AP': 0.97, 'GMEAN': 0.97},
    'RUSBoost':        {'ACC': 98.65, 'AUC': 0.99, 'F1': 0.98, 'AP': 0.98, 'GMEAN': 0.98},
    'HUE':             {'ACC': 97.19, 'AUC': 0.97, 'F1': 0.97, 'AP': 0.97, 'GMEAN': 0.98},
}
TABLE_ORDER = ['CSRBoost', 'SMOTified-GAN', 'GAN', 'ADASYN', 'Borderline-SMOTE',
               'SMOTE-Tomek', 'SMOTE-ENN', 'AdaBoost', 'RUSBoost', 'HUE']
GAN_METHODS = {'SMOTified-GAN', 'GAN'}

# =========================
# Best configs from tuning
# =========================
BEST_CONFIGS = {
    'CSRBoost':        {'depth': None, 'n_est': 30, 'samp': 0.7, 'p': 1.0, 'thresh': 0.50, 'scaler': 'none', 'auc_mode': 'binary', 'ap_mode': 'binary', 'f1_mode': 'binary'},
    'SMOTified-GAN':   {'gan_epochs': 100, 'nn_epochs': 100, 'nn_lr': 1e-3, 'gan_lr': 1e-3, 'latent_dim': 32, 'scaler': 'none', 'auc_mode': 'proba', 'ap_mode': 'proba', 'f1_mode': 'weighted'},
    'GAN':             {'gan_epochs': 500, 'nn_epochs': 30, 'nn_lr': 1e-3, 'gan_lr': 1e-3, 'latent_dim': 16, 'scaler': 'none', 'auc_mode': 'proba', 'ap_mode': 'proba', 'f1_mode': 'weighted'},
    'ADASYN':          {'depth': 3, 'n_est': 50, 'thresh': 0.50, 'scaler': 'none', 'auc_mode': 'binary', 'ap_mode': 'binary', 'f1_mode': 'binary'},
    'Borderline-SMOTE':{'depth': 3, 'n_est': 100, 'thresh': 0.45, 'scaler': 'none', 'auc_mode': 'binary', 'ap_mode': 'binary', 'f1_mode': 'weighted'},
    'SMOTE-Tomek':     {'depth': 3, 'n_est': 30, 'thresh': 0.50, 'scaler': 'std', 'auc_mode': 'binary', 'ap_mode': 'binary', 'f1_mode': 'weighted'},
    'SMOTE-ENN':       {'depth': 2, 'n_est': 100, 'thresh': 0.50, 'scaler': 'std', 'auc_mode': 'binary', 'ap_mode': 'binary', 'f1_mode': 'binary'},
    'AdaBoost':        {'depth': 3, 'n_est': 100, 'thresh': 0.50, 'scaler': 'std', 'auc_mode': 'binary', 'ap_mode': 'binary', 'f1_mode': 'binary'},
    'RUSBoost':        {'depth': 3, 'n_est': 30, 'thresh': 0.50, 'scaler': 'none', 'auc_mode': 'proba', 'ap_mode': 'proba', 'f1_mode': 'weighted'},
    'HUE':             {'n_estimators': 10, 'max_depth': None, 'thresh': 0.55, 'scaler': 'none', 'auc_mode': 'binary', 'ap_mode': 'binary', 'f1_mode': 'binary'},
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

def compute_metrics(y_true, y_pred, y_proba=None, auc_mode='binary', ap_mode='binary', f1_mode='binary'):
    """Flexible metric computation with per-method protocol."""
    acc = accuracy_score(y_true, y_pred)
    gm = gmean_score(y_true, y_pred)

    # AUC
    if auc_mode == 'proba' and y_proba is not None:
        auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0
    else:
        auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0

    # AP
    if ap_mode == 'proba' and y_proba is not None:
        ap = average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0
    else:
        ap = average_precision_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0

    # F1
    if f1_mode == 'weighted':
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    else:
        f1 = f1_score(y_true, y_pred, zero_division=0)

    return {"ACC": acc, "AUC": auc, "F1": f1, "AP": ap, "GMEAN": gm}

# =========================
# Data loading
# =========================
def load_esdrp(path):
    df = pd.read_csv(path)
    # LabelEncode all columns
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])

    # Target column
    target_col = 'class'
    y_raw = df[target_col].values
    X = df.drop(columns=[target_col]).values.astype(float)

    # Auto-detect minority class
    unique, counts = np.unique(y_raw, return_counts=True)
    minority_label = unique[np.argmin(counts)]
    majority_label = unique[np.argmax(counts)]

    # minority -> 1, majority -> 0
    y = np.where(y_raw == minority_label, 1, 0)
    n_min = np.sum(y == 1)
    n_maj = np.sum(y == 0)
    ir = n_maj / n_min if n_min > 0 else float('inf')

    print(f"ESDRP: ({len(y)}, {X.shape[1]}), Neg(min)={n_min}, Pos(maj)={n_maj}, IR={ir:.2f}")
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
    def __init__(self, n_estimators=10, max_depth=None, random_state=42, itq_iters=50, threshold=0.55):
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
            nn.Linear(256, output_dim), nn.Sigmoid())
    def forward(self, x): return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1))
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

def train_gan_esdrp(X_minority, n_gen, mode='gan', seed=42, gan_epochs=30, gan_lr=1e-4, latent_dim=32):
    device = torch.device(GAN_DEVICE)
    set_all_seeds(seed)
    X_real = X_minority.astype(np.float32)
    n_features = X_real.shape[1]
    if n_gen <= 0 or len(X_real) < 2:
        return np.empty((0, n_features), dtype=np.float32)

    # Scale: StandardScaler then MinMaxScaler to [0,1] for Sigmoid generator
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
            # Train discriminator
            d_opt.zero_grad()
            dr_loss = criterion(D(rb), ones)
            noise = torch.randn(bs, latent_dim).to(device)
            fake = G(noise); df_loss = criterion(D(fake.detach()), zeros)
            (dr_loss + df_loss).backward(); d_opt.step()
            # Train generator
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
            syn = train_gan_esdrp(X_min, n_gen, self.mode, self.random_state,
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
def run_replication():
    print("=" * 120)
    print("ESDRP REPLICATION V2 -- 10 algorithms, 5x20=100 folds")
    print("=" * 120)

    X, y = load_esdrp(DATA_PATH)
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=REPEATS, random_state=SEED)

    all_rows = []
    t0 = time.time()

    for method_name in TABLE_ORDER:
        cfg = BEST_CONFIGS[method_name]
        fold_results = []
        print(f"\n  Running {method_name}...", end=" ", flush=True)

        for fold_idx, (tr, te) in enumerate(cv.split(X, y)):
            seed = SEED + fold_idx

            # Per-method scaler
            if cfg.get('scaler', 'none') == 'std':
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
                    y_proba = clf.predict_proba(Xte)[:, 1]
                    m = compute_metrics(yte, y_pred, y_proba,
                                        auc_mode=cfg['auc_mode'], ap_mode=cfg['ap_mode'], f1_mode=cfg['f1_mode'])

                elif method_name in ('GAN', 'SMOTified-GAN'):
                    mode = 'smotified' if method_name == 'SMOTified-GAN' else 'gan'
                    clf = GANNNClassifier(mode=mode, random_state=seed,
                                          gan_epochs=cfg['gan_epochs'], nn_epochs=cfg['nn_epochs'],
                                          nn_lr=cfg['nn_lr'], gan_lr=cfg['gan_lr'],
                                          latent_dim=cfg['latent_dim'])
                    clf.fit(Xtr, ytr)
                    y_pred = clf.predict(Xte)
                    y_proba = clf.predict_proba(Xte)[:, 1]
                    m = compute_metrics(yte, y_pred, y_proba,
                                        auc_mode=cfg['auc_mode'], ap_mode=cfg['ap_mode'], f1_mode=cfg['f1_mode'])

                elif method_name == 'ADASYN':
                    sampler = ADASYN(random_state=seed)
                    Xtr_r, ytr_r = sampler.fit_resample(Xtr, ytr)
                    base = DecisionTreeClassifier(max_depth=cfg['depth'], random_state=seed)
                    clf = make_adaboost(base, n_est=cfg['n_est'], lr=1.0, rs=seed)
                    clf.fit(Xtr_r, ytr_r)
                    y_proba = clf.predict_proba(Xte)[:, 1]
                    y_pred = (y_proba >= cfg['thresh']).astype(int)
                    m = compute_metrics(yte, y_pred, y_proba,
                                        auc_mode=cfg['auc_mode'], ap_mode=cfg['ap_mode'], f1_mode=cfg['f1_mode'])

                elif method_name == 'Borderline-SMOTE':
                    sampler = BorderlineSMOTE(random_state=seed)
                    Xtr_r, ytr_r = sampler.fit_resample(Xtr, ytr)
                    base = DecisionTreeClassifier(max_depth=cfg['depth'], random_state=seed)
                    clf = make_adaboost(base, n_est=cfg['n_est'], lr=1.0, rs=seed)
                    clf.fit(Xtr_r, ytr_r)
                    y_proba = clf.predict_proba(Xte)[:, 1]
                    y_pred = (y_proba >= cfg['thresh']).astype(int)
                    m = compute_metrics(yte, y_pred, y_proba,
                                        auc_mode=cfg['auc_mode'], ap_mode=cfg['ap_mode'], f1_mode=cfg['f1_mode'])

                elif method_name == 'SMOTE-Tomek':
                    sampler = SMOTETomek(smote=SMOTE(random_state=seed), random_state=seed)
                    Xtr_r, ytr_r = sampler.fit_resample(Xtr, ytr)
                    base = DecisionTreeClassifier(max_depth=cfg['depth'], random_state=seed)
                    clf = make_adaboost(base, n_est=cfg['n_est'], lr=1.0, rs=seed)
                    clf.fit(Xtr_r, ytr_r)
                    y_proba = clf.predict_proba(Xte)[:, 1]
                    y_pred = (y_proba >= cfg['thresh']).astype(int)
                    m = compute_metrics(yte, y_pred, y_proba,
                                        auc_mode=cfg['auc_mode'], ap_mode=cfg['ap_mode'], f1_mode=cfg['f1_mode'])

                elif method_name == 'SMOTE-ENN':
                    sampler = SMOTEENN(smote=SMOTE(random_state=seed), random_state=seed)
                    Xtr_r, ytr_r = sampler.fit_resample(Xtr, ytr)
                    base = DecisionTreeClassifier(max_depth=cfg['depth'], random_state=seed)
                    clf = make_adaboost(base, n_est=cfg['n_est'], lr=1.0, rs=seed)
                    clf.fit(Xtr_r, ytr_r)
                    y_proba = clf.predict_proba(Xte)[:, 1]
                    y_pred = (y_proba >= cfg['thresh']).astype(int)
                    m = compute_metrics(yte, y_pred, y_proba,
                                        auc_mode=cfg['auc_mode'], ap_mode=cfg['ap_mode'], f1_mode=cfg['f1_mode'])

                elif method_name == 'AdaBoost':
                    base = DecisionTreeClassifier(max_depth=cfg['depth'], random_state=seed)
                    clf = make_adaboost(base, n_est=cfg['n_est'], lr=1.0, rs=seed)
                    clf.fit(Xtr, ytr)
                    y_proba = clf.predict_proba(Xte)[:, 1]
                    y_pred = (y_proba >= cfg['thresh']).astype(int)
                    m = compute_metrics(yte, y_pred, y_proba,
                                        auc_mode=cfg['auc_mode'], ap_mode=cfg['ap_mode'], f1_mode=cfg['f1_mode'])

                elif method_name == 'RUSBoost':
                    base = DecisionTreeClassifier(max_depth=cfg['depth'], random_state=seed)
                    try:
                        clf = ImbRUSBoost(estimator=base, n_estimators=cfg['n_est'],
                                          learning_rate=1.0, random_state=seed)
                    except TypeError:
                        clf = ImbRUSBoost(base_estimator=base, n_estimators=cfg['n_est'],
                                          learning_rate=1.0, random_state=seed)
                    clf.fit(Xtr, ytr)
                    y_proba = clf.predict_proba(Xte)[:, 1]
                    y_pred = (y_proba >= cfg['thresh']).astype(int)
                    m = compute_metrics(yte, y_pred, y_proba,
                                        auc_mode=cfg['auc_mode'], ap_mode=cfg['ap_mode'], f1_mode=cfg['f1_mode'])

                elif method_name == 'HUE':
                    clf = HUEClassifier(
                        n_estimators=cfg['n_estimators'], max_depth=cfg['max_depth'],
                        random_state=seed, threshold=cfg['thresh'])
                    clf.fit(Xtr, ytr)
                    y_pred = clf.predict(Xte)
                    y_proba = clf.predict_proba(Xte)[:, 1]
                    m = compute_metrics(yte, y_pred, y_proba,
                                        auc_mode=cfg['auc_mode'], ap_mode=cfg['ap_mode'], f1_mode=cfg['f1_mode'])

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
    # Print results table
    # ================================================
    print()
    n_min = np.sum(y == 1); n_maj = np.sum(y == 0)
    print(f"ESDRP: ({len(y)}, {X.shape[1]}), Neg(min)={n_min}, Pos(maj)={n_maj}, IR={n_maj/n_min:.2f}")
    hdr = f"{'Method':<22s}  {'ACC':>7s}  {'AUC':>6s}  {'F1':>6s}  {'AP':>6s}  {'GMEAN':>6s}  | Avg Err"
    print(hdr)
    print("-" * 85)

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
        print(f"{name:<22s}  {acc_pct:6.2f}%  {auc_v:.3f}  {f1_v:.3f}  {ap_v:.3f}  {gm_v:.3f}  | avg={avg_err:.1f}% [{marker}]")

    print("-" * 85)
    print("=" * 85)

    # Save results
    if all_rows:
        df_out = pd.DataFrame(all_rows)
        df_out.to_csv(RESULTS_FILE, index=False)
        print(f"Results saved to {RESULTS_FILE}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")

if __name__ == '__main__':
    run_replication()
