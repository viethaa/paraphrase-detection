# Imports
import ast
import math
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # for multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class SentenceEncoderRNN(nn.Module):
    def __init__(self, num_embeddings, emb_dim, hidden, num_layers,
                 bidirectional=True, dropout=0, padding_idx=None):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, emb_dim, padding_idx=padding_idx)
        self.lstm = nn.RNN(
            input_size=emb_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden * (2 if bidirectional else 1)

    def forward(self, x, lengths, pad_id):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, (h, c) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        rep = out.mean(dim=1)

        return rep

class SiameseRNN(nn.Module):
    def __init__(self, num_embeddings, emb_dim, hidden, num_layers, bidirectional,dropout=0,
                 padding_idx=None, pool="mean"):
        super().__init__()
        self.encoder = SentenceEncoderRNN(num_embeddings, emb_dim=emb_dim, hidden=hidden,num_layers=num_layers, bidirectional=bidirectional,
                                           padding_idx=padding_idx, dropout=dropout)

    def forward(self, s1, len1, s2, len2, pad_id):
        h1 = self.encoder(s1, len1, pad_id)
        h2 = self.encoder(s2, len2, pad_id)
        sim = F.cosine_similarity(h1, h2, dim=-1)  # [-1, 1]
        return sim

emb_dim = ...
hidden = ...
num_layers = ...
dropout = ...

ckpt_path = ...

num_embeddings = ...
PAD_ID = ...
UNK_ID = ...


# %%
model = SiameseRNN(num_embeddings=num_embeddings,
                    emb_dim=emb_dim, hidden=hidden, num_layers =num_layers,
                    padding_idx=PAD_ID, dropout=dropout)

import ast
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Custom collate function for inference. Handle out-of-vocab tokens
def make_collate_test(pad_id: int, num_embeddings: int, unk_id: int = 0):
    assert 0 <= pad_id < num_embeddings, "pad_id must be within [0, num_embeddings)"
    assert 0 <= unk_id < num_embeddings, "unk_id must be within [0, num_embeddings)"
    assert pad_id != unk_id, "pad_id and unk_id must differ"

    def collate_pairs(batch):
        s1_list = [b["s1"] for b in batch]
        s2_list = [b["s2"] for b in batch]
        y = torch.tensor([b["y"] for b in batch], dtype=torch.float32) if "y" in batch[0] else None

        def pad_and_len(seqs, pad):
            # lengths: enforce >=1 so pack_padded_sequence never sees 0
            lens = torch.tensor([max(1, len(x)) for x in seqs], dtype=torch.long)
            max_len = int(lens.max()) if len(seqs) else 1
            out = torch.full((len(seqs), max_len), pad, dtype=torch.long)
            for i, seq in enumerate(seqs):
                if len(seq) > 0:
                    t = torch.tensor(seq, dtype=torch.long)
                    # map OOV to UNK (leave PAD alone if it ever appears)
                    oov_mask = (t < 0) | (t >= num_embeddings)
                    if oov_mask.any():
                        t[oov_mask] = unk_id
                    out[i, :len(seq)] = t
                else:
                    out[i, 0] = pad  # keep one PAD when empty
            return out, lens

        s1_pad, len1 = pad_and_len(s1_list, pad_id)
        s2_pad, len2 = pad_and_len(s2_list, pad_id)
        return (s1_pad, len1, s2_pad, len2, y) if y is not None else (s1_pad, len1, s2_pad, len2)

    return collate_pairs



def infer(
    csv_file,
    ckpt_path,
    num_embeddings,          # required if model object not in scope
    emb_dim,
    hidden,
    num_layers,
    bidirectional=True,
    dropout=0,
    padding_idx=0,             # should be the same PAD_ID used in training
    batch_size=256,
    device='cuda',
    out_path="predictions.txt",

):
    """
    Minimal inference:
      - expects test csv file with columns: sentence_1, sentence_2 (list-of-ints)
      - saves 0/1 predictions (threshold 0.5) to predictions.txt
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    assert padding_idx is not None, "Please pass the same PAD_ID (padding_idx) used during training."
    assert num_embeddings is not None, "Please pass num_embeddings (vocab size used during training)."

    # 1) Load test CSV and parse lists
    df = pd.read_csv(csv_file)
    def _parse(x):
        if isinstance(x, list): return [int(t) for t in x]
        if pd.isna(x): return []
        if isinstance(x, str): return [int(t) for t in ast.literal_eval(x)]
        return []
    s1_list = df["sentence_1"].apply(_parse).tolist()
    s2_list = df["sentence_2"].apply(_parse).tolist()

    # 2) DataLoader (reuse your make_collate with the training PAD_ID)
    collate = make_collate_test(padding_idx, num_embeddings = num_embeddings)
    class _TmpDS:
        def __len__(self): return len(s1_list)
        def __getitem__(self, i): return {"s1": s1_list[i], "s2": s2_list[i]}
    loader = DataLoader(_TmpDS(), batch_size=batch_size, shuffle=False, collate_fn=collate)

    # 3) Rebuild model with the SAME hyperparams as training and load weights
    model = SiameseRNN(
        num_embeddings=num_embeddings,
        emb_dim=emb_dim,
        hidden=hidden,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(state, strict=True)
    model.eval()


    # 4) Inference
    preds_bin = []
    with torch.no_grad():
        for s1, len1, s2, len2 in loader:
            s1, len1 = s1.to(device), len1.to(device)
            s2, len2 = s2.to(device), len2.to(device)
            sims = model(s1, len1, s2, len2, padding_idx)      # cosine in [-1,1]
            preds = (sims >= 0.5).long().cpu().tolist()        # baseline threshold
            preds_bin.extend(preds)

    # 5) Save predictions
    with open(out_path, "w") as f:
        for p in preds_bin:
            f.write(f"{p}\n")

    print(f"Saved {len(preds_bin)} predictions to {out_path}")

infer(csv_file='public_test.csv',
    ckpt_path=ckpt_path,
    num_embeddings=num_embeddings,          # required if model object not in scope
    emb_dim=emb_dim,
    hidden=hidden,
    num_layers=num_layers,
    bidirectional=True,
    dropout=dropout,
    padding_idx=PAD_ID,             # should be the same PAD_ID used in training
    batch_size=256,
    device='cpu',
)



