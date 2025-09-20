import ast
import math
import random
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import numpy as np

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # for multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_file(path):
    df = pd.read_csv(path)

    for col in ["sentence_1", "sentence_2"]:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    return df

df = load_file("train.csv")

# TODO: Split df into train and validation set
train_df, val_df = ...


# ## 2) Build vocab from train and apply to all splits
# Replace tokens not seen ≥ min_count times in train with UNK_ID.

# %%
UNK_ID = 0  # simple baseline: reserve 0 for unknowns

def build_vocab(train_df, s1_col="sentence_1", s2_col="sentence_2", min_count=2):
    all_tokens = sum(train_df[s1_col].tolist() + train_df[s2_col].tolist(), [])
    # TODO: keep freq as a count dict of all_tokens
    freq = ...

    # TODO: Keep tokens that appear >= min_count times
    kept_vocab = ...
    return kept_vocab, freq

def apply_vocab(df, kept_vocab, unk_id=0, s1_col="sentence_1", s2_col="sentence_2"):
    df = df.copy()

    #TODO: Turn tokens not in kept_vocab into unknơwn token id, otherwise keep that token
    def replace(seq):
        return [...]
    df[s1_col] = df[s1_col].apply(replace)
    df[s2_col] = df[s2_col].apply(replace)
    return df

kept_vocab, freq = build_vocab(train_df, min_count=2)
train_df = apply_vocab(train_df, kept_vocab, UNK_ID)
val_df   = apply_vocab(val_df,   kept_vocab, UNK_ID)

print("vocab size:", len(kept_vocab))


# ## 3) Datasets & DataLoaders

# %%
class PairDataset(Dataset):
    def __init__(self, df, s1_col="sentence_1", s2_col="sentence_2", y_col="label"):
        self.s1 = df[s1_col].tolist()
        self.s2 = df[s2_col].tolist()
        self.y  = df[y_col].astype(float).tolist()
        self.max_id = max([max(seq) for seq in (self.s1 + self.s2) if seq] + [0])

    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        # TODO: create a dict with keys s1, s2 and y being the sentence 1, sentence 2 and label (as float) at position idx
        return {...}

def make_collate(pad_id: int):
    def collate_pairs(batch):
        s1_list = [b["s1"] for b in batch]
        s2_list = [b["s2"] for b in batch]
        y = torch.tensor([b["y"] for b in batch], dtype=torch.float32)

        def pad_and_len(seqs, pad):
            lens = torch.tensor([len(x) for x in seqs], dtype=torch.long)
            max_len = lens.max().item() if len(lens) > 0 else 0
            out = torch.full((len(seqs), max_len), pad, dtype=torch.long)
            for i, seq in enumerate(seqs):
                if len(seq) > 0:
                    out[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            return out, lens

        s1_pad, len1 = pad_and_len(s1_list, pad_id)
        s2_pad, len2 = pad_and_len(s2_list, pad_id)
        return s1_pad, len1, s2_pad, len2, y
    return collate_pairs

ds_tr = PairDataset(train_df)
ds_va = PairDataset(val_df)

# Choose PAD_ID dynamically to avoid clashing with real tokens
PAD_ID = max(ds_tr.max_id, ds_va.max_id, UNK_ID) + 1
num_embeddings = PAD_ID + 1   # indices 0..PAD_ID inclusive

train_loader = DataLoader(ds_tr, batch_size=64, shuffle=True,
                          collate_fn=make_collate(PAD_ID))
val_loader   = DataLoader(ds_va, batch_size=128, shuffle=False,
                          collate_fn=make_collate(PAD_ID))

print("PAD_ID:", PAD_ID, "num_embeddings:", num_embeddings)


# ## 4) Siamese RNN (shared encoder) + cosine

# %%
class SentenceEncoderRNN(nn.Module):
    def __init__(self, num_embeddings, emb_dim, hidden, num_layers,
                 bidirectional=True, dropout=0, padding_idx=None):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, emb_dim, padding_idx=padding_idx)
        self.rnn  = nn.RNN(
            input_size=emb_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths, pad_id):
        # TODO: Pass input through embedding layer
        emb = ....
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # TODO: Pass through RNN layer
        out_packed, (h, c) = ...
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        # TODO: Get the average embedding of the 'out' vector across the token sequence dimension
        rep = ...

        return rep

class SiameseRNN(nn.Module):

    """
    A Siamese Recurrent Neural Network for paraphrase detection.

    This model determines whether two input sentences are paraphrases by
    encoding each sentence into a fixed-length vector representation using
    a shared sentence encoder (SentenceEncoderRNN) and then computing their
    similarity.

    Workflow:
        1. Each input sentence (tokenized as integer IDs) is passed through
           the shared SentenceEncoderRNN, which embeds tokens and processes
           them with a recurrent neural network.
        2. The sequence of hidden states is mean-pooled into a single
           vector representation for each sentence.
        3. The two sentence embeddings (h1 and h2) are compared using
           cosine similarity, producing a score in the range [-1, 1]:
               - Values close to 1 indicate high semantic similarity
                 (likely paraphrases).
               - Values close to -1 indicate strong dissimilarity.
        4. During training, you need to align cosine similarity with the ground-truth
        paraphrase label (your own choice).

    Args:
        num_embeddings (int): Vocabulary size (number of token IDs).
        emb_dim (int): Dimensionality of word embeddings.
        hidden (int): Hidden size of the recurrent network.
        num_layers (int): Number of stacked RNN layers.
        bidirectional (bool): If True, use bidirectional RNNs.
        padding_idx (int): Index used for padded tokens in the embedding layer.
        pool (str): Pooling method for sentence representation ("mean", "max", etc.).

    Forward Args:
        s1 (LongTensor): Batch of tokenized sentence 1, shape [B, L1].
        len1 (LongTensor): Lengths of sentence 1 sequences, shape [B].
        s2 (LongTensor): Batch of tokenized sentence 2, shape [B, L2].
        len2 (LongTensor): Lengths of sentence 2 sequences, shape [B].
        pad_id (int): Token ID used for padding.

    Returns:
        sim (Tensor): Cosine similarity scores between sentence pairs,
                      shape [B], values in [-1, 1].
    """

    def __init__(self, num_embeddings, emb_dim, hidden, num_layers, bidirectional=False, dropout =0,
                 padding_idx=None, pool="mean"):
        super().__init__()
        self.encoder = SentenceEncoderRNN(num_embeddings, emb_dim=emb_dim, hidden=hidden,num_layers=num_layers, bidirectional=bidirectional,
                                           padding_idx=padding_idx, dropout=dropout)

    def forward(self, s1, len1, s2, len2, pad_id):
        # TODO: Encode input sequence s1 and s2 into the encoder
        h1 = ...
        h2 = ...

        #TODO: calculate cosine similarities between vector h1 and h2
        sim = ...
        return sim



def train_loop(model, train_loader, val_loader, epochs=EPOCHS, lr=LR,
               device="cuda" if torch.cuda.is_available() else "cpu",
               pad_id=0,
               ckpt_path="lstm_best.pt"):
    model.to(device)

    #TODO: Define your own optimizer and loss function here
    optimizer = ...
    criterion = ...

    best_val_loss = float("inf")

    for ep in range(1, epochs + 1):
        # -------- Train --------
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for s1, len1, s2, len2, y in tqdm(train_loader, desc=f"Train {ep}", leave=False):
            s1, len1 = s1.to(device), len1.to(device)
            s2, len2 = s2.to(device), len2.to(device)
            y = y.to(device)

            #TODO: Clears all the gradients in the model's parameters
            ...

            yhat = model(s1, len1, s2, len2, pad_id)
            loss = criterion(yhat, y)

            #TODO: computes gradients
            ...
            #TODO: Update model parameters with optimizer
            ...

            tr_loss += loss.item() * y.size(0)
            preds = (yhat >= 0.5).float()   # simple threshold for 0/1 labels
            tr_correct += (preds == y).sum().item()
            tr_total += y.size(0)

        tr_loss /= len(train_loader.dataset)
        tr_acc = tr_correct / max(1, tr_total)

        # -------- Val --------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for s1, len1, s2, len2, y in val_loader:
                s1, len1 = s1.to(device), len1.to(device)
                s2, len2 = s2.to(device), len2.to(device)
                y = y.to(device)

                yhat = model(s1, len1, s2, len2, pad_id)
                loss = criterion(yhat, y)
                val_loss += loss.item() * y.size(0)

                preds = (yhat >= 0.5).float()
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / max(1, val_total)

        print(f"Epoch {ep:02d} | "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # -------- Save best weights --------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved new best weights to {ckpt_path}")

#TODO: define your own parameters
emb_dim = ...
hidden = ...
num_layers = ...
dropout = ...

EPOCHS = 20
LR = 1e-3
ckpt_path = 'rnn_best.pt'

# ## 6) Train

model = SiameseRNN(num_embeddings=num_embeddings,
                    emb_dim=emb_dim, hidden=hidden, num_layers =num_layers,
                    padding_idx=PAD_ID, dropout=dropout)

train_loop(model, train_loader, val_loader,
           epochs=epochs, lr=lr, pad_id=PAD_ID,
           ckpt_path=ckpt_path)


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
    bidirectional,
    dropout,
    padding_idx,             # should be the same PAD_ID used in training
    pool="mean",
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


# Please use the correct values for other arguments for your saved model/training process
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
    device='cuda',
)
