# ============================================================
# CSRBoost Paper Replication V2 - ESR Dataset
# ============================================================
# KEY FIXES:
# 1. GAN/SMOTified-GAN: Mixed evaluation
# 2. Non-GAN methods: Binary AUC/AP
# 3. HUE: ExtraTrees (n_estimators=20, max_depth=10) - works well on ESR
# 4. Paper numbers from graduation report Table 8a
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
try:
    from sklearn.cluster import MiniBatchKMeans
    _HAS_MINIBATCH = True
except:
    _HAS_MINIBATCH = False

from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

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
DATA_PATH = os.path.join(SCRIPT_DIR, "Epileptic Seizure Recognition.csv")
SEED = 42
N_SPLITS = 5
REPEATS = 20    # 5x20=100 folds (paper replication)
RESULTS_FILE = "esr_v2_results.csv"

# Boosting
N_ESTIMATORS = 50
BASE_TREE_MAX_DEPTH = None
LEARNING_RATE = 1.0
ADABOOST_ALGO = "SAMME"

# CSRBoost
CSR_P = 1.0
CSR_SAMPLE_PCT = 0.5
CSR_SMOTE_K = 5

# HUE - ExtraTrees for ESR (confirmed 95.94% in previous runs)
HUE_N_ESTIMATORS = 20
HUE_MAX_DEPTH = 10
HUE_ITQ_ITERS = 50

# GAN (500 for initial test; mixed eval makes epoch count less critical)
GAN_EPOCHS = 500
GAN_BATCH_SIZE = 128
GAN_LR = 1e-5
GAN_LATENT_DIM = 100
GAN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GAN_SMOTE_K = 5

# NN Classifier
NN_HIDDEN = [256, 128]
NN_EPOCHS = 200
NN_LR = 1e-5
NN_BATCH_SIZE = 128

# =========================
# PAPER TABLE 8a (from graduation report)
# =========================
PAPER_TABLE = {
    'CSRBoost':        {'ACC': 92.05, 'AUC': 0.90, 'F1': 0.80, 'AP': 0.67, 'GMEAN': 0.89},
    'SMOTified-GAN':   {'ACC': 95.84, 'AUC': 0.97, 'F1': 0.96, 'AP': 0.95, 'GMEAN': 0.92},
    'GAN':             {'ACC': 95.89, 'AUC': 0.97, 'F1': 0.96, 'AP': 0.95, 'GMEAN': 0.92},
    'ADASYN':          {'ACC': 89.95, 'AUC': 0.89, 'F1': 0.77, 'AP': 0.63, 'GMEAN': 0.89},
    'Borderline-SMOTE':{'ACC': 91.93, 'AUC': 0.89, 'F1': 0.82, 'AP': 0.69, 'GMEAN': 0.90},
    'SMOTE-Tomek':     {'ACC': 91.81, 'AUC': 0.89, 'F1': 0.81, 'AP': 0.69, 'GMEAN': 0.90},
    'SMOTE-ENN':       {'ACC': 91.72, 'AUC': 0.89, 'F1': 0.81, 'AP': 0.68, 'GMEAN': 0.90},
    'AdaBoost':        {'ACC': 94.18, 'AUC': 0.91, 'F1': 0.85, 'AP': 0.76, 'GMEAN': 0.91},
    'RUSBoost':        {'ACC': None,  'AUC': None, 'F1': None, 'AP': None, 'GMEAN': None},
    'HUE':             {'ACC': 95.55, 'AUC': 0.94, 'F1': 0.87, 'AP': 0.77, 'GMEAN': 0.96},
}
TABLE_ORDER = ['CSRBoost', 'SMOTified-GAN', 'GAN', 'ADASYN', 'Borderline-SMOTE',
               'SMOTE-Tomek', 'SMOTE-ENN', 'AdaBoost', 'RUSBoost', 'HUE']
GAN_METHODS = {'SMOTified-GAN', 'GAN'}

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
def make_adaboost(base, rs):
    try:
        return AdaBoostClassifier(estimator=base, n_estimators=N_ESTIMATORS,
                                  learning_rate=LEARNING_RATE, random_state=rs, algorithm=ADABOOST_ALGO)
    except TypeError:
        return AdaBoostClassifier(base_estimator=base, n_estimators=N_ESTIMATORS,
                                  learning_rate=LEARNING_RATE, random_state=rs, algorithm=ADABOOST_ALGO)

def gmean_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return float(math.sqrt(tpr * tnr))

def compute_metrics_standard(y_true, y_pred):
    return {
        "ACC": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_pred),
        "F1":  f1_score(y_true, y_pred, zero_division=0),
        "AP":  average_precision_score(y_true, y_pred),
        "GMEAN": gmean_score(y_true, y_pred),
    }

def compute_metrics_mixed(y_test, y_pred_test, y_proba_test, y_orig_train, y_proba_orig_train):
    """GAN/SMOTified-GAN evaluation for ESR — Protocol C (all test):
    ACC   = test hard preds
    AUC   = test probabilities
    F1    = test hard preds
    AP    = test probabilities
    GMEAN = test hard preds
    """
    return {
        "ACC":   accuracy_score(y_test, y_pred_test),
        "AUC":   roc_auc_score(y_test, y_proba_test),
        "F1":    f1_score(y_test, y_pred_test, zero_division=0),
        "AP":    average_precision_score(y_test, y_proba_test),
        "GMEAN": gmean_score(y_test, y_pred_test),
    }

def load_esr(path):
    df = pd.read_csv(path)
    cols_to_drop = [c for c in df.columns if 'unnamed' in c.lower()]
    if df.columns[0] != 'X1' and df[df.columns[0]].dtype == 'object':
        cols_to_drop.append(df.columns[0])
    if cols_to_drop: df = df.drop(columns=cols_to_drop)
    target_col = 'y' if 'y' in df.columns else df.columns[-1]
    y_orig = df[target_col].values
    y = np.where(y_orig == 1, 1, 0)  # Class 1 = minority (seizure)
    X = df.drop(columns=[target_col]).values.astype(float)
    n_maj, n_min = np.sum(y == 0), np.sum(y == 1)
    print(f"ESR: {X.shape}, Maj={n_maj}, Min={n_min}, IR={n_maj/n_min:.2f}")
    return X, y

# =========================
# HUE (ExtraTrees for ESR)
# =========================
def itq_fit(X, n_bits, n_iter, random_state):
    rng = check_random_state(random_state)
    mu = X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=n_bits, random_state=random_state)
    V = pca.fit_transform(X - mu)
    R = rng.randn(n_bits, n_bits)
    U, _, VT = np.linalg.svd(R, full_matrices=False); R = U @ VT
    for _ in range(n_iter):
        Z = V @ R; B = np.where(Z >= 0, 1.0, -1.0)
        U, _, VT = np.linalg.svd(B.T @ V, full_matrices=False); R = U @ VT
    return {"mu": mu, "pca": pca, "R": R}

def itq_transform(X, model):
    V = model["pca"].transform(X - model["mu"])
    return (V @ model["R"] >= 0).astype(np.uint8)

def bits_to_uint(bits):
    B = bits.shape[1]
    return (bits.astype(np.uint32) * (1 << np.arange(B, dtype=np.uint32))).sum(axis=1)

class HUEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=20, max_depth=10, random_state=42, itq_iters=50):
        self.n_estimators = n_estimators; self.max_depth = max_depth
        self.random_state = random_state; self.itq_iters = itq_iters

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
        proba = np.zeros((len(X), 2))
        for est in self.estimators_:
            p = est.predict_proba(X)
            tmp = np.zeros_like(proba)
            for j, c in enumerate(est.classes_): tmp[:, int(c)] = p[:, j]
            proba += tmp
        return proba / len(self.estimators_)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# =========================
# CSRBoost
# =========================
class CSRBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, p=1.0, sample_pct=0.5, smote_k=5, base_max_depth=None, random_state=42):
        self.p = p; self.sample_pct = sample_pct; self.smote_k = smote_k
        self.base_max_depth = base_max_depth; self.random_state = random_state

    def fit(self, X, y):
        rng = check_random_state(self.random_state)
        X_min, X_maj = X[y == 1], X[y == 0]
        n_min, n_maj = len(X_min), len(X_maj)
        nc = max(1, min(int(round(self.p * n_min)), n_maj))
        km = MiniBatchKMeans(n_clusters=nc, random_state=self.random_state,
                             n_init=1, max_iter=100, batch_size=1024) if _HAS_MINIBATCH else \
             KMeans(n_clusters=nc, random_state=self.random_state, n_init=1, max_iter=100)
        labels = km.fit_predict(X_maj)
        kept = []
        for c in range(nc):
            idx = np.where(labels == c)[0]
            if idx.size == 0: continue
            nk = max(1, int(math.ceil(idx.size * self.sample_pct)))
            ch = rng.choice(idx, size=nk, replace=False) if nk < idx.size else idx
            kept.append(X_maj[ch])
        Xmu = np.vstack(kept) if kept else X_maj
        Xc = np.vstack([X_min, Xmu])
        yc = np.hstack([np.ones(n_min, dtype=int), np.zeros(len(Xmu), dtype=int)])
        k = min(self.smote_k, max(1, n_min - 1))
        Xb, yb = SMOTE(k_neighbors=k, random_state=self.random_state).fit_resample(Xc, yc)
        base = DecisionTreeClassifier(max_depth=self.base_max_depth, random_state=self.random_state)
        self.model_ = make_adaboost(base, self.random_state)
        self.model_.fit(Xb, yb); return self

    def predict(self, X): return self.model_.predict(X)
    def predict_proba(self, X): return self.model_.predict_proba(X)

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
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, output_dim), nn.Tanh())
    def forward(self, x): return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
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
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def train_gan_esr(X_minority, n_gen, mode='gan', seed=42):
    """GAN training with early stopping for ESR"""
    device = torch.device(GAN_DEVICE)
    rng = check_random_state(seed); set_all_seeds(seed)
    X_real = X_minority.astype(np.float32)
    n_features = X_real.shape[1]
    latent_dim = GAN_LATENT_DIM
    if n_gen <= 0 or len(X_real) < 2:
        return np.empty((0, n_features), dtype=np.float32)

    # Scale internally for better GAN training
    from sklearn.preprocessing import StandardScaler as SS
    sc = SS(); X_scaled = sc.fit_transform(X_real)

    if mode == 'smotified':
        # SMOTE augmentation before GAN
        k = min(GAN_SMOTE_K, len(X_real) - 1)
        n_smote = min(n_gen, len(X_real) * 3)
        X_dummy = np.random.randn(len(X_real) + n_smote, n_features).astype(np.float32)
        X_comb = np.vstack([X_scaled, X_dummy])
        y_comb = np.array([1]*len(X_scaled) + [0]*len(X_dummy))
        try:
            sm = SMOTE(k_neighbors=k, random_state=seed)
            X_sm, y_sm = sm.fit_resample(X_comb, y_comb)
            X_train_gan = X_sm[y_sm == 1].astype(np.float32)
        except:
            X_train_gan = X_scaled.astype(np.float32)
    else:
        X_train_gan = X_scaled.astype(np.float32)

    G = Generator(latent_dim, n_features).to(device)
    D = Discriminator(n_features).to(device)
    g_opt = optim.Adam(G.parameters(), lr=GAN_LR)
    d_opt = optim.Adam(D.parameters(), lr=GAN_LR)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_gan)),
                        batch_size=min(GAN_BATCH_SIZE, len(X_train_gan)), shuffle=True)

    best_balance = float('inf'); patience_counter = 0
    for epoch in range(GAN_EPOCHS):
        G.train(); D.train()
        d_real_c = 0; d_fake_c = 0; total = 0
        for (rb,) in loader:
            rb = rb.to(device); bs = rb.size(0)
            ones = torch.ones(bs, 1).to(device); zeros = torch.zeros(bs, 1).to(device)
            d_opt.zero_grad()
            dr_out = D(rb); dr_loss = criterion(dr_out, ones)
            noise = torch.randn(bs, latent_dim).to(device)
            fake = G(noise); df_out = D(fake.detach()); df_loss = criterion(df_out, zeros)
            (dr_loss + df_loss).backward(); d_opt.step()
            g_opt.zero_grad()
            fake = G(torch.randn(bs, latent_dim).to(device))
            g_loss = criterion(D(fake), ones); g_loss.backward(); g_opt.step()
            d_real_c += (torch.sigmoid(dr_out) > 0.5).sum().item()
            d_fake_c += (torch.sigmoid(df_out) < 0.5).sum().item()
            total += bs
        # Early stopping
        d_ra = d_real_c / total; d_fa = d_fake_c / total
        balance = abs(d_ra - 0.5) + abs(d_fa - 0.5)
        if balance < best_balance: best_balance = balance; patience_counter = 0
        else: patience_counter += 1
        if d_ra < 0.6 and d_fa < 0.6: break
        if patience_counter >= 200: break

    G.eval()
    with torch.no_grad():
        syn = G(torch.randn(n_gen, latent_dim).to(device)).cpu().numpy()
    return sc.inverse_transform(syn).astype(np.float32)

class GANNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, mode='gan', random_state=42):
        self.mode = mode; self.random_state = random_state

    def fit(self, X, y):
        X_min = X[y == 1]; n_min, n_maj = len(X_min), (y == 0).sum()
        n_gen = n_maj - n_min
        if n_gen > 0 and n_min >= 2:
            syn = train_gan_esr(X_min, n_gen, self.mode, self.random_state)
            if syn is not None and len(syn) > 0:
                X_aug = np.vstack([X, syn]); y_aug = np.hstack([y, np.ones(len(syn), dtype=int)])
            else: X_aug, y_aug = X, y
        else: X_aug, y_aug = X, y
        self.X_aug_ = X_aug; self.y_aug_ = y_aug
        device = torch.device(GAN_DEVICE)
        self.model_ = NNClassifierModule(X_aug.shape[1], NN_HIDDEN).to(device)
        loader = DataLoader(TensorDataset(torch.FloatTensor(X_aug), torch.LongTensor(y_aug)),
                            batch_size=NN_BATCH_SIZE, shuffle=True)
        optimizer = optim.Adam(self.model_.parameters(), lr=NN_LR)
        criterion = nn.CrossEntropyLoss()
        self.model_.train()
        for _ in range(NN_EPOCHS):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(); loss = criterion(self.model_(xb), yb)
                loss.backward(); optimizer.step()
        return self

    def predict_proba(self, X):
        device = torch.device(GAN_DEVICE)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.FloatTensor(X).to(device))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X): return self.predict_proba(X).argmax(axis=1)

    def predict_on_training(self):
        return self.predict(self.X_aug_), self.predict_proba(self.X_aug_)[:, 1]

# =========================
# Wrappers
# =========================
class ResampleThenAdaBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, resampler, base_max_depth=None, random_state=42):
        self.resampler = resampler; self.base_max_depth = base_max_depth; self.random_state = random_state
    def fit(self, X, y):
        Xr, yr = self.resampler.fit_resample(X, y)
        base = DecisionTreeClassifier(max_depth=self.base_max_depth, random_state=self.random_state)
        self.model_ = make_adaboost(base, self.random_state); self.model_.fit(Xr, yr); return self
    def predict(self, X): return self.model_.predict(X)
    def predict_proba(self, X): return self.model_.predict_proba(X)

class AdaBoostBaseline(BaseEstimator, ClassifierMixin):
    def __init__(self, base_max_depth=None, random_state=42):
        self.base_max_depth = base_max_depth; self.random_state = random_state
    def fit(self, X, y):
        base = DecisionTreeClassifier(max_depth=self.base_max_depth, random_state=self.random_state)
        self.model_ = make_adaboost(base, self.random_state); self.model_.fit(X, y); return self
    def predict(self, X): return self.model_.predict(X)
    def predict_proba(self, X): return self.model_.predict_proba(X)

# =========================
# RUSBoost
# =========================
class RUSBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=50, learning_rate=1.0, base_max_depth=None, random_state=42):
        self.n_estimators = n_estimators; self.learning_rate = learning_rate
        self.base_max_depth = base_max_depth; self.random_state = random_state

    def fit(self, X, y):
        rng = check_random_state(self.random_state)
        n = len(y); w = np.ones(n) / n
        idx_min, idx_maj = np.where(y == 1)[0], np.where(y == 0)[0]
        n_min = len(idx_min)
        self.estimators_, self.estimator_weights_ = [], []
        for _ in range(self.n_estimators):
            idx_ms = rng.choice(idx_maj, size=n_min, replace=False) if len(idx_maj) > n_min else idx_maj
            idx_t = np.concatenate([idx_min, idx_ms])
            wt = w[idx_t]; wt /= wt.sum()
            est = DecisionTreeClassifier(max_depth=self.base_max_depth, random_state=rng.randint(0, 1_000_000))
            est.fit(X[idx_t], y[idx_t], sample_weight=wt)
            yp = est.predict(X); inc = (yp != y).astype(float)
            err = np.clip(np.sum(w * inc), 1e-12, 1 - 1e-12)
            alpha = self.learning_rate * math.log((1 - err) / err)
            w *= np.exp(alpha * inc); w /= w.sum()
            self.estimators_.append(est); self.estimator_weights_.append(alpha)
        self.estimator_weights_ = np.array(self.estimator_weights_)
        return self

    def predict_proba(self, X):
        proba = np.zeros((len(X), 2))
        W = self.estimator_weights_ / (self.estimator_weights_.sum() + 1e-12)
        for est, a in zip(self.estimators_, W):
            p = est.predict_proba(X)
            tmp = np.zeros_like(proba)
            for j, c in enumerate(est.classes_): tmp[:, int(c)] = p[:, j]
            proba += a * tmp
        return proba

    def predict(self, X): return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# =========================
# Build Models
# =========================
def build_models(seed_offset):
    rs = SEED + seed_offset
    return {
        "CSRBoost": CSRBoostClassifier(p=CSR_P, sample_pct=CSR_SAMPLE_PCT,
                                        smote_k=CSR_SMOTE_K, base_max_depth=BASE_TREE_MAX_DEPTH, random_state=rs),
        "SMOTified-GAN": GANNNClassifier(mode='smotified', random_state=rs),
        "GAN": GANNNClassifier(mode='gan', random_state=rs),
        "ADASYN": ResampleThenAdaBoost(ADASYN(random_state=rs), BASE_TREE_MAX_DEPTH, rs),
        "Borderline-SMOTE": ResampleThenAdaBoost(BorderlineSMOTE(kind="borderline-1", random_state=rs),
                                                  BASE_TREE_MAX_DEPTH, rs),
        "SMOTE-Tomek": ResampleThenAdaBoost(SMOTETomek(random_state=rs), BASE_TREE_MAX_DEPTH, rs),
        "SMOTE-ENN": ResampleThenAdaBoost(SMOTEENN(random_state=rs), BASE_TREE_MAX_DEPTH, rs),
        "AdaBoost": AdaBoostBaseline(BASE_TREE_MAX_DEPTH, rs),
        "RUSBoost": RUSBoostClassifier(N_ESTIMATORS, LEARNING_RATE, BASE_TREE_MAX_DEPTH, rs),
        "HUE": HUEClassifier(n_estimators=HUE_N_ESTIMATORS, max_depth=HUE_MAX_DEPTH,
                             random_state=rs, itq_iters=HUE_ITQ_ITERS),
    }

# =========================
# Print Results
# =========================
def print_results(results_mean, results_std):
    metrics = ['ACC', 'AUC', 'F1', 'AP', 'GMEAN']
    print("\n" + "=" * 110)
    print("ESR REPLICATION V2 - Results vs Paper (Table 8a)")
    print("=" * 110)
    hdr = f"{'Method':<20}"
    for m in metrics: hdr += f"{'Ours':>8} {'Paper':>6} {'Err%':>6}  "
    print(hdr); print("-" * 110)
    for method in TABLE_ORDER:
        paper = PAPER_TABLE.get(method, {})
        if method not in results_mean.index:
            print(f"{method:<20} N/A"); continue
        r = results_mean.loc[method]
        row = f"{method:<20}"; errs = []
        for m in metrics:
            pv = paper.get(m); rv = r[m]
            if m == 'ACC':
                rv_d = rv * 100; pv_d = pv if pv else 0
                if pv:
                    err = abs(rv_d - pv_d) / pv_d * 100; errs.append(err)
                    row += f"{rv_d:7.2f}% {pv_d:5.1f}% {err:5.1f}%  "
                else: row += f"{rv_d:7.2f}%  {'N/A':>5}  {'N/A':>5}  "
            else:
                if pv is not None:
                    err = abs(rv - pv) / pv * 100 if pv != 0 else 0; errs.append(err)
                    row += f"{rv:8.3f} {pv:5.2f} {err:5.1f}%  "
                else: row += f"{rv:8.3f}  {'N/A':>5}  {'N/A':>5}  "
        avg_err = np.mean(errs) if errs else 0
        status = "OK" if avg_err < 3 else "~" if avg_err < 5 else "X"
        row += f" | avg={avg_err:.1f}% [{status}]"
        print(row)
    print("=" * 110)

# =========================
# Main
# =========================
def main():
    print("=" * 80)
    print("CSRBoost Replication V2 - ESR Dataset")
    print(f"CV: {N_SPLITS}x{REPEATS}={N_SPLITS*REPEATS} folds | Device: {GAN_DEVICE}")
    print("=" * 80)

    X, y = load_esr(DATA_PATH)
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=REPEATS, random_state=SEED)
    all_rows = []; t0 = time.time()

    for fold_idx, (tr, te) in enumerate(cv.split(X, y), start=1):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        ytr, yte = y[tr], y[te]
        models = build_models(seed_offset=fold_idx * 997)

        for name, model in models.items():
            try:
                model.fit(Xtr, ytr)
                y_pred = model.predict(Xte)
                if name in GAN_METHODS:
                    y_proba_test = model.predict_proba(Xte)[:, 1]
                    y_proba_orig = model.predict_proba(Xtr)[:, 1]
                    m = compute_metrics_mixed(yte, y_pred, y_proba_test, ytr, y_proba_orig)
                else:
                    m = compute_metrics_standard(yte, y_pred)
                all_rows.append({"Fold": fold_idx, "Method": name, **m})
            except Exception as e:
                all_rows.append({"Fold": fold_idx, "Method": name,
                                 "ACC": np.nan, "AUC": np.nan, "F1": np.nan,
                                 "AP": np.nan, "GMEAN": np.nan})
                print(f"[Fold {fold_idx}] {name} FAILED: {e}")

        elapsed = time.time() - t0
        eta = elapsed / fold_idx * (N_SPLITS * REPEATS - fold_idx)
        print(f"Fold {fold_idx}/{N_SPLITS*REPEATS} | elapsed={elapsed:.1f}s | ETA={eta:.0f}s")
        sys.stdout.flush()

    results = pd.DataFrame(all_rows)
    results_mean = results.groupby("Method")[["ACC","AUC","F1","AP","GMEAN"]].mean()
    results_std = results.groupby("Method")[["ACC","AUC","F1","AP","GMEAN"]].std()
    print_results(results_mean, results_std)
    results.to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved to: {RESULTS_FILE}")
    print("\nESR V2 REPLICATION COMPLETE!")

if __name__ == "__main__":
    main()
