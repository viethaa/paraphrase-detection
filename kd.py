#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESIM-style paraphrase detector trained FROM SCRATCH on integer-token sentences.
- No pretrained weights, no external data.
- Reads train and public test from:
    /data/train.csv (columns: sentence_1, sentence_2, label)
    /data/public_test.csv (columns: sentence_1, sentence_2)
- Produces predictions to:
    /data/public_predictions.csv  (columns: row_id, prob, label)

Model: Shared embedding -> BiLSTM (context) -> Cross-attention (soft alignment)
        -> Compare MLP -> BiLSTM (inference)
        -> Pool (avg+max for both sents) + cheap overlap features
        -> MLP -> sigmoid

You can toggle to a simpler Siamese encoder by setting CFG.model_variant = "siamese".

Run:
    python esim_paraphrase.py  # if you save this file as esim_paraphrase.py

Notes:
- The code makes minimal assumptions about tokenization: each sentence field is a whitespace-separated string of integer ids.
- PAD=0, UNK=1. All observed ids are shifted by +2 so we don't collide with PAD/UNK.
- Dynamic padding per batch; bucketed batching by rough length to stabilize training.
"""

import os
import csv
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Optional sklearn (for stratified split + AUROC); falls back gracefully if missing
try:
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# -----------------------------
# Config
# -----------------------------
@dataclass
class CFG:
    train_csv: str = "/data/train.csv"
    public_test_csv: str = "/data/public_test.csv"
    out_csv: str = "/data/public_predictions.csv"

    seed: int = 42
    epochs: int = 10
    batch_size: int = 128

    # Model
    model_variant: str = "esim"   # "esim" or "siamese"
    vocab_extra: int = 2           # reserve 0:PAD, 1:UNK ; shift observed ids by +2
    embed_dim: int = 256
    hidden: int = 192             # BiLSTM hidden per direction
    comp_hidden: int = 200        # comparison MLP hidden
    mlp_hidden: int = 300         # final MLP hidden
    dropout: float = 0.4

    # Regularization
    word_dropout: float = 0.08     # replace tokens with UNK during training
    label_smoothing: float = 0.05  # 0 to disable

    # Optim
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_ratio: float = 0.05

    # Misc
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    patience: int = 3              # early stopping patience (in epochs)


# -----------------------------
# Utils
# -----------------------------

def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_int_seq(s: str) -> List[int]:
    # Accept empty or NaN as empty sequence
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    s = str(s).strip()
    if not s:
        return []
    toks = s.split()
    out = []
    for t in toks:
        try:
            out.append(int(t))
        except Exception:
            # If there's any non-integer junk, skip it safely
            continue
    return out


def load_data(train_csv: str, public_test_csv: str):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(public_test_csv)
    # Expect columns sentence_1, sentence_2, and label (train only)
    for col in ["sentence_1", "sentence_2"]:
        assert col in train_df.columns, f"Missing column {col} in train.csv"
        assert col in test_df.columns, f"Missing column {col} in public_test.csv"
    assert "label" in train_df.columns, "Missing label column in train.csv"

    # Parse sequences & gather max token id
    def parse_cols(df):
        s1 = df["sentence_1"].apply(parse_int_seq).tolist()
        s2 = df["sentence_2"].apply(parse_int_seq).tolist()
        return s1, s2

    tr_s1, tr_s2 = parse_cols(train_df)
    te_s1, te_s2 = parse_cols(test_df)

    labels = train_df["label"].astype(int).values

    all_tokens = []
    for seq in tr_s1 + tr_s2 + te_s1 + te_s2:
        all_tokens.extend(seq)
    max_id = max(all_tokens) if all_tokens else 0

    # Shift observed ids by +2 so 0:PAD, 1:UNK remain reserved
    def offset(seq):
        return [t + CFG.vocab_extra for t in seq]

    tr_s1 = [offset(x) for x in tr_s1]
    tr_s2 = [offset(x) for x in tr_s2]
    te_s1 = [offset(x) for x in te_s1]
    te_s2 = [offset(x) for x in te_s2]

    vocab_size = max_id + 1 + CFG.vocab_extra  # +1 because ids are inclusive

    return (tr_s1, tr_s2, labels), (te_s1, te_s2), vocab_size


# -----------------------------
# Dataset & Collate (dynamic padding + word dropout)
# -----------------------------
class PairDataset(Dataset):
    def __init__(self, s1: List[List[int]], s2: List[List[int]], y: Optional[np.ndarray] = None):
        self.s1 = s1
        self.s2 = s2
        self.y = y
        assert len(self.s1) == len(self.s2)
        if y is not None:
            assert len(self.s1) == len(y)

    def __len__(self):
        return len(self.s1)

    def __getitem__(self, idx):
        if self.y is None:
            return self.s1[idx], self.s2[idx]
        else:
            return self.s1[idx], self.s2[idx], int(self.y[idx])


def lengths_bucketed_indices(lengths: List[int], n_buckets: int = 10) -> List[int]:
    # Returns indices sorted by approximate length buckets to reduce padding waste
    idxs = list(range(len(lengths)))
    lengths = np.array(lengths)
    bins = np.percentile(lengths, np.linspace(0, 100, n_buckets+1))
    bucket_ids = np.digitize(lengths, bins[1:-1], right=True)
    grouped = [[] for _ in range(n_buckets+1)]
    for i, b in zip(idxs, bucket_ids):
        grouped[b].append(i)
    # Shuffle within buckets, then concatenate
    for g in grouped:
        random.shuffle(g)
    return [i for g in grouped for i in g]


def collate_train(batch, word_dropout=0.0):
    # batch: list of (s1, s2, y)
    PAD = 0
    UNK = 1
    s1_list, s2_list, y_list = [], [], []
    for item in batch:
        s1, s2, y = item
        if word_dropout > 0.0:
            s1 = [ (UNK if (t != PAD and random.random() < word_dropout) else t) for t in s1 ]
            s2 = [ (UNK if (t != PAD and random.random() < word_dropout) else t) for t in s2 ]
        s1_list.append(torch.tensor(s1, dtype=torch.long))
        s2_list.append(torch.tensor(s2, dtype=torch.long))
        y_list.append(y)
    y = torch.tensor(y_list, dtype=torch.float32)

    s1_pad = nn.utils.rnn.pad_sequence(s1_list, batch_first=True, padding_value=PAD)
    s2_pad = nn.utils.rnn.pad_sequence(s2_list, batch_first=True, padding_value=PAD)
    mask1 = (s1_pad != PAD).float()
    mask2 = (s2_pad != PAD).float()
    return s1_pad, s2_pad, mask1, mask2, y


def collate_test(batch):
    PAD = 0
    s1_list, s2_list = [], []
    for item in batch:
        s1, s2 = item
        s1_list.append(torch.tensor(s1, dtype=torch.long))
        s2_list.append(torch.tensor(s2, dtype=torch.long))
    s1_pad = nn.utils.rnn.pad_sequence(s1_list, batch_first=True, padding_value=PAD)
    s2_pad = nn.utils.rnn.pad_sequence(s2_list, batch_first=True, padding_value=PAD)
    mask1 = (s1_pad != PAD).float()
    mask2 = (s2_pad != PAD).float()
    return s1_pad, s2_pad, mask1, mask2


# -----------------------------
# Model components
# -----------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class ESIM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden: int, comp_hidden: int, mlp_hidden: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)

        self.enc = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)

        # Attention projections (optional): here we use scaled dot-product with projections
        self.proj_a = nn.Linear(2*hidden, 2*hidden, bias=False)
        self.proj_b = nn.Linear(2*hidden, 2*hidden, bias=False)

        # Comparison MLP (applied token-wise)
        self.compare = nn.Sequential(
            nn.Linear(8*hidden, comp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(comp_hidden, comp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Inference composition
        self.infer = nn.LSTM(comp_hidden, hidden, batch_first=True, bidirectional=True)

        # Final classifier
        # We will concatenate avg+max pooled for both sequences (4 * 2*hidden) = 8*hidden
        # plus 4 cheap overlap features
        self.classifier = nn.Sequential(
            nn.Linear(8*hidden + 4, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden//2, 1)
        )

        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def masked_softmax(scores: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        # mask: 1 for valid, 0 for pad; broadcast to scores shape
        mask = mask.to(dtype=scores.dtype)
        scores = scores.masked_fill(mask == 0, -1e9)
        return torch.softmax(scores, dim=dim)

    def forward(self, s1, s2, m1, m2):
        # s1/s2: (B, T), m1/m2: (B, T) floats (1 = valid)
        emb1 = self.emb_dropout(self.embedding(s1))
        emb2 = self.emb_dropout(self.embedding(s2))

        # Context encoding
        a, _ = self.enc(emb1)
        b, _ = self.enc(emb2)

        # Cross attention: scores (B, Ta, Tb)
        a_proj = self.proj_a(a)
        b_proj = self.proj_b(b)
        scale = math.sqrt(a_proj.size(-1))
        scores = torch.bmm(a_proj, b_proj.transpose(1, 2)) / scale

        # Build masks for attention
        m1_ = m1.unsqueeze(2)  # (B, Ta, 1)
        m2_ = m2.unsqueeze(1)  # (B, 1, Tb)
        attn_weights_a = self.masked_softmax(scores, m2_, dim=2)   # sum over Tb
        attn_weights_b = self.masked_softmax(scores.transpose(1, 2), m1_, dim=2)  # sum over Ta

        # Aligned representations
        alpha = torch.bmm(attn_weights_a, b)  # (B, Ta, 2H)
        beta  = torch.bmm(attn_weights_b, a)  # (B, Tb, 2H)

        # Compose
        a_comp = torch.cat([a, alpha, a - alpha, a * alpha], dim=-1)
        b_comp = torch.cat([b, beta,  b - beta,  b * beta ], dim=-1)

        v1 = self.compare(a_comp)
        v2 = self.compare(b_comp)

        # Inference composition
        v1, _ = self.infer(v1)
        v2, _ = self.infer(v2)

        # Pooling (mask-aware)
        def masked_pool(x, m):
            m = m.unsqueeze(-1)  # (B, T, 1)
            x_masked = x * m
            # avg
            lengths = m.sum(dim=1).clamp(min=1)
            avg = x_masked.sum(dim=1) / lengths
            # max (set pad positions to very negative)
            x_masked = x.masked_fill(m == 0, -1e9)
            mx = x_masked.max(dim=1).values
            return avg, mx

        avg1, max1 = masked_pool(v1, m1)
        avg2, max2 = masked_pool(v2, m2)

        feats = torch.cat([avg1, max1, avg2, max2], dim=-1)  # (B, 8H)
        return feats

    def classify(self, feats, overlap_feats):
        x = torch.cat([feats, overlap_feats], dim=-1)
        logit = self.classifier(x).squeeze(-1)
        return logit


class SiameseEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)
        self.enc = nn.LSTM(embed_dim, hidden, batch_first=True, bidirectional=True)
        # Optional self-attention to produce a better sentence vector
        self.att_vector = nn.Parameter(torch.randn(2*hidden))
        self.dropout = nn.Dropout(dropout)
        # Final MLP will be defined outside, using feature concat
        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, s, m):
        x = self.emb_dropout(self.embedding(s))
        h, _ = self.enc(x)
        # Self-attention pooling
        # score_t = h_t · w  ; mask-aware softmax
        w = self.att_vector / (torch.norm(self.att_vector) + 1e-6)
        scores = torch.matmul(h, w)
        scores = scores.masked_fill(m == 0, -1e9)
        attn = torch.softmax(scores, dim=1).unsqueeze(-1)
        vec = (h * attn).sum(dim=1)
        # Also mix with avg+max for stability
        lengths = m.sum(dim=1, keepdim=True).clamp(min=1)
        avg = (h * m.unsqueeze(-1)).sum(dim=1) / lengths
        h_masked = h.masked_fill(m.unsqueeze(-1) == 0, -1e9)
        mx = h_masked.max(dim=1).values
        return torch.cat([vec, avg, mx], dim=-1)  # (B, 6H)

    def forward(self, s1, s2, m1, m2):
        v1 = self.encode(s1, m1)
        v2 = self.encode(s2, m2)
        # classic matching features
        feats = torch.cat([v1, v2, torch.abs(v1 - v2), v1 * v2], dim=-1)
        return feats


class SiameseClassifier(nn.Module):
    def __init__(self, in_dim: int, mlp_hidden: int, dropout: float, extra_feat_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + extra_feat_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden//2, 1)
        )
    def forward(self, feats, overlap_feats):
        x = torch.cat([feats, overlap_feats], dim=-1)
        return self.net(x).squeeze(-1)


# -----------------------------
# Cheap overlap features (id-space)
# -----------------------------

def overlap_features(s1: torch.Tensor, s2: torch.Tensor, m1: torch.Tensor, m2: torch.Tensor) -> torch.Tensor:
    """Compute four scalar features per pair in-batch:
    - jaccard over unique token ids
    - len diff ratio
    - unique ratio diff
    - raw overlap ratio (|intersection|/|s1|)
    Inputs are padded tensors (B,T) with masks (B,T).
    Returns (B, 4)
    """
    PAD = 0
    B, T1 = s1.shape
    T2 = s2.shape[1]
    feats = []
    for i in range(B):
        a = s1[i][m1[i].bool()].tolist()
        b = s2[i][m2[i].bool()].tolist()
        set_a, set_b = set([t for t in a if t != PAD]), set([t for t in b if t != PAD])
        inter = len(set_a & set_b)
        union = len(set_a | set_b) if (set_a or set_b) else 1
        jacc = inter / union
        len_a, len_b = len(a), len(b)
        len_diff_ratio = abs(len_a - len_b) / max(1, max(len_a, len_b))
        uniq_ratio_diff = abs((len(set_a)/max(1,len_a)) - (len(set_b)/max(1,len_b)))
        overlap_a = inter / max(1, len_a)
        feats.append([jacc, len_diff_ratio, uniq_ratio_diff, overlap_a])
    return torch.tensor(feats, dtype=torch.float32, device=s1.device)


# -----------------------------
# Training helpers
# -----------------------------
class WarmupCosine:
    def __init__(self, optimizer, total_steps, warmup_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.total = total_steps
        self.warmup = warmup_steps
        self.min_lr = min_lr
        self.step_num = 0
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self):
        self.step_num += 1
        for i, g in enumerate(self.optimizer.param_groups):
            base = self.base_lrs[i]
            if self.step_num <= self.warmup:
                lr = base * self.step_num / max(1, self.warmup)
            else:
                t = (self.step_num - self.warmup) / max(1, self.total - self.warmup)
                lr = self.min_lr + 0.5*(base - self.min_lr)*(1 + math.cos(math.pi * t))
            g["lr"] = lr


def bce_with_logits_ls(logits, targets, label_smoothing=0.0):
    if label_smoothing > 0.0:
        eps = label_smoothing
        targets = targets * (1 - eps) + 0.5 * eps
    return F.binary_cross_entropy_with_logits(logits, targets)


# -----------------------------
# Train / Eval loops
# -----------------------------

def train_one_epoch(model, clf, loader, optimizer, scheduler, device, variant: str, label_smoothing: float):
    model.train()
    if clf is not None:
        clf.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith("cuda")))
    for s1, s2, m1, m2, y in loader:
        s1, s2, m1, m2, y = s1.to(device), s2.to(device), m1.to(device), m2.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.startswith("cuda"))):
            if variant == "esim":
                feats = model(s1, s2, m1, m2)
                ov = overlap_features(s1, s2, m1, m2)
                logits = model.classify(feats, ov)
            else:  # siamese
                feats = model(s1, s2, m1, m2)
                ov = overlap_features(s1, s2, m1, m2)
                logits = clf(feats, ov)
            loss = bce_with_logits_ls(logits, y, label_smoothing)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
        if clf is not None:
            nn.utils.clip_grad_norm_(clf.parameters(), CFG.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * y.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        total_correct += (preds == y.long()).sum().item()
        total_count += y.size(0)
    return total_loss / max(1,total_count), total_correct / max(1,total_count)


def eval_one_epoch(model, clf, loader, device, variant: str):
    model.eval()
    if clf is not None:
        clf.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    probs_all, labels_all = [], []
    with torch.no_grad():
        for s1, s2, m1, m2, y in loader:
            s1, s2, m1, m2, y = s1.to(device), s2.to(device), m1.to(device), m2.to(device), y.to(device)
            if variant == "esim":
                feats = model(s1, s2, m1, m2)
                ov = overlap_features(s1, s2, m1, m2)
                logits = model.classify(feats, ov)
            else:
                feats = model(s1, s2, m1, m2)
                ov = overlap_features(s1, s2, m1, m2)
                logits = clf(feats, ov)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            total_loss += loss.item() * y.size(0)
            total_correct += (preds == y.long()).sum().item()
            total_count += y.size(0)
            probs_all.append(probs.detach().cpu().numpy())
            labels_all.append(y.detach().cpu().numpy())

    probs_all = np.concatenate(probs_all) if probs_all else np.array([])
    labels_all = np.concatenate(labels_all) if labels_all else np.array([])
    auc = None
    if SKLEARN_AVAILABLE and probs_all.size > 0:
        try:
            auc = roc_auc_score(labels_all, probs_all)
        except Exception:
            auc = None
    return total_loss / max(1,total_count), total_correct / max(1,total_count), auc


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    set_seed(CFG.seed)

    (tr_s1, tr_s2, labels), (te_s1, te_s2), vocab_size = load_data(CFG.train_csv, CFG.public_test_csv)

    # Train/val split
    if SKLEARN_AVAILABLE:
        train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=0.15, stratify=labels, random_state=CFG.seed)
    else:
        idxs = np.arange(len(labels))
        np.random.shuffle(idxs)
        split = int(0.85 * len(labels))
        train_idx, val_idx = idxs[:split], idxs[split:]

    tr_ds = PairDataset([tr_s1[i] for i in train_idx], [tr_s2[i] for i in train_idx], labels[train_idx])
    va_ds = PairDataset([tr_s1[i] for i in val_idx], [tr_s2[i] for i in val_idx], labels[val_idx])
    te_ds = PairDataset(te_s1, te_s2, None)

    # Bucketed sampler: sort indices by length buckets to reduce pad variance
    tr_lengths = [len(tr_s1[i]) + len(tr_s2[i]) for i in train_idx]
    tr_order = lengths_bucketed_indices(tr_lengths, n_buckets=10)
    tr_sampler = torch.utils.data.SubsetRandomSampler([train_idx[i] for i in tr_order])

    train_loader = DataLoader(tr_ds, batch_size=CFG.batch_size, shuffle=False, sampler=tr_sampler,
                              num_workers=CFG.num_workers, collate_fn=lambda b: collate_train(b, CFG.word_dropout),
                              pin_memory=True)
    val_loader = DataLoader(va_ds, batch_size=CFG.batch_size, shuffle=False,
                            num_workers=CFG.num_workers, collate_fn=lambda b: collate_train(b, 0.0), pin_memory=True)
    test_loader = DataLoader(te_ds, batch_size=CFG.batch_size, shuffle=False,
                             num_workers=CFG.num_workers, collate_fn=collate_test, pin_memory=True)

    device = CFG.device
    print(f"Device: {device}; Vocab size: {vocab_size}")

    if CFG.model_variant == "esim":
        model = ESIM(vocab_size, CFG.embed_dim, CFG.hidden, CFG.comp_hidden, CFG.mlp_hidden, CFG.dropout).to(device)
        clf = None
        params = list(model.parameters())
    else:
        enc = SiameseEncoder(vocab_size, CFG.embed_dim, CFG.hidden, CFG.dropout).to(device)
        # feats dim: v1/v2 are 6H each -> concat v1,v2,|diff|,prod => 24H
        feats_dim = 24 * CFG.hidden
        clf = SiameseClassifier(in_dim=feats_dim, mlp_hidden=CFG.mlp_hidden, dropout=CFG.dropout, extra_feat_dim=4).to(device)
        model = enc
        params = list(model.parameters()) + list(clf.parameters())

    optimizer = torch.optim.AdamW(params, lr=CFG.lr, weight_decay=CFG.weight_decay)
    total_steps = CFG.epochs * max(1, len(train_loader))
    warmup_steps = int(CFG.warmup_ratio * total_steps)
    scheduler = WarmupCosine(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, CFG.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, clf, train_loader, optimizer, scheduler, device, CFG.model_variant, CFG.label_smoothing)
        va_loss, va_acc, va_auc = eval_one_epoch(model, clf, val_loader, device, CFG.model_variant)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} || val loss {va_loss:.4f} acc {va_acc:.4f} auc {va_auc if va_auc is not None else 'NA'}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            epochs_no_improve = 0
            # Save best state
            best_state = {
                'model': {k: v.cpu() for k, v in model.state_dict().items()},
                'clf': ({k: v.cpu() for k, v in clf.state_dict().items()} if clf is not None else None),
                'cfg': CFG.__dict__.copy(),
                'vocab_size': vocab_size,
            }
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CFG.patience:
                print("Early stopping: no improvement.")
                break

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state['model'])
        if clf is not None and best_state['clf'] is not None:
            clf.load_state_dict(best_state['clf'])

    # Inference on public test
    model.eval()
    if clf is not None:
        clf.eval()
    probs = []
    with torch.no_grad():
        for s1, s2, m1, m2 in test_loader:
            s1, s2, m1, m2 = s1.to(device), s2.to(device), m1.to(device), m2.to(device)
            if CFG.model_variant == "esim":
                feats = model(s1, s2, m1, m2)
                ov = overlap_features(s1, s2, m1, m2)
                logits = model.classify(feats, ov)
            else:
                feats = model(s1, s2, m1, m2)
                ov = overlap_features(s1, s2, m1, m2)
                logits = clf(feats, ov)
            p = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(p)
    probs = np.concatenate(probs)
    labels = (probs >= 0.5).astype(int)

    os.makedirs(os.path.dirname(CFG.out_csv), exist_ok=True)
    with open(CFG.out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "prob", "label"])
        for i, (pr, lb) in enumerate(zip(probs.tolist(), labels.tolist())):
            writer.writerow([i, pr, int(lb)])
    print(f"Wrote predictions to {CFG.out_csv}")


if __name__ == "__main__":
    # Allow simple CLI overrides, e.g. --epochs 12 --model_variant siamese
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=CFG.epochs)
    parser.add_argument('--batch_size', type=int, default=CFG.batch_size)
    parser.add_argument('--model_variant', type=str, default=CFG.model_variant, choices=['esim','siamese'])
    parser.add_argument('--embed_dim', type=int, default=CFG.embed_dim)
    parser.add_argument('--hidden', type=int, default=CFG.hidden)
    parser.add_argument('--dropout', type=float, default=CFG.dropout)
    parser.add_argument('--word_dropout', type=float, default=CFG.word_dropout)
    parser.add_argument('--lr', type=float, default=CFG.lr)
    parser.add_argument('--label_smoothing', type=float, default=CFG.label_smoothing)
    parser.add_argument('--train_csv', type=str, default=CFG.train_csv)
    parser.add_argument('--public_test_csv', type=str, default=CFG.public_test_csv)
    parser.add_argument('--out_csv', type=str, default=CFG.out_csv)
    args = parser.parse_args()

    # Overwrite CFG with CLI values
    for k, v in vars(args).items():
        setattr(CFG, k, v)

    main()
