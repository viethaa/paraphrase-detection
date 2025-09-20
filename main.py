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
from sklearn.model_selection import train_test_split

# Check GPU availability for Colab
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_file(path):
    df = pd.read_csv(path)
    for col in ["sentence_1", "sentence_2"]:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

df = load_file("sample_data/train.csv")

# Split df into train and validation set (80/20 split)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Build vocab from train and apply to all splits
UNK_ID = 0
CLS_ID = 1  # Special token for classification
SEP_ID = 2  # Separator token
PAD_ID = 3  # Padding token

def build_vocab(train_df, s1_col="sentence_1", s2_col="sentence_2", min_count=2):
    all_tokens = sum(train_df[s1_col].tolist() + train_df[s2_col].tolist(), [])
    freq = Counter(all_tokens)

    # Keep tokens that appear >= min_count times
    kept_vocab = {token for token, count in freq.items() if count >= min_count}

    # Create token to id mapping, reserving special tokens
    special_tokens = {UNK_ID, CLS_ID, SEP_ID, PAD_ID}
    vocab_to_id = {}
    id_counter = max(special_tokens) + 1

    for token in kept_vocab:
        if token not in special_tokens:
            vocab_to_id[token] = id_counter
            id_counter += 1

    # Add special tokens
    vocab_to_id[UNK_ID] = UNK_ID
    vocab_to_id[CLS_ID] = CLS_ID
    vocab_to_id[SEP_ID] = SEP_ID
    vocab_to_id[PAD_ID] = PAD_ID

    return vocab_to_id, freq

def apply_vocab(df, vocab_to_id, unk_id=0, s1_col="sentence_1", s2_col="sentence_2"):
    df = df.copy()

    def replace(seq):
        return [vocab_to_id.get(token, unk_id) for token in seq]

    df[s1_col] = df[s1_col].apply(replace)
    df[s2_col] = df[s2_col].apply(replace)
    return df

vocab_to_id, freq = build_vocab(train_df, min_count=2)
train_df = apply_vocab(train_df, vocab_to_id, UNK_ID)
val_df = apply_vocab(val_df, vocab_to_id, UNK_ID)

vocab_size = len(vocab_to_id)
print("vocab size:", vocab_size)

# BERT-style dataset: [CLS] sent1 [SEP] sent2 [SEP]
class BERTDataset(Dataset):
    def __init__(self, df, s1_col="sentence_1", s2_col="sentence_2", y_col="label", max_length=512):
        self.s1 = df[s1_col].tolist()
        self.s2 = df[s2_col].tolist()
        self.y = df[y_col].astype(float).tolist()
        self.max_length = max_length

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        s1, s2 = self.s1[idx], self.s2[idx]

        # Create BERT-style input: [CLS] s1 [SEP] s2 [SEP]
        # Truncate if too long
        max_s1_len = self.max_length - 4  # Reserve space for [CLS], 2x[SEP], and at least 1 token for s2
        max_s2_len = self.max_length - len(s1[:max_s1_len]) - 3

        s1_truncated = s1[:max_s1_len]
        s2_truncated = s2[:max_s2_len]

        # Build sequence
        input_ids = [CLS_ID] + s1_truncated + [SEP_ID] + s2_truncated + [SEP_ID]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)

        # Create token type ids (0 for first sentence, 1 for second sentence)
        token_type_ids = ([0] * (len(s1_truncated) + 2) +
                         [1] * (len(s2_truncated) + 1))

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': self.y[idx]
        }

def bert_collate_fn(batch):
    # Get max length in batch
    max_len = max(len(item['input_ids']) for item in batch)

    input_ids = []
    attention_masks = []
    token_type_ids = []
    labels = []

    for item in batch:
        # Pad sequences
        pad_len = max_len - len(item['input_ids'])

        input_ids.append(item['input_ids'] + [PAD_ID] * pad_len)
        attention_masks.append(item['attention_mask'] + [0] * pad_len)
        token_type_ids.append(item['token_type_ids'] + [0] * pad_len)
        labels.append(item['label'])

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.float)
    }

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections and reshape
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            # Use a smaller negative value that works with float16
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask == 0, mask_value)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)

        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)

        output = self.W_o(context)
        return output

# Position-wise Feed Forward Network
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=5000):
        super().__init__()

        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :].transpose(0, 1)

# BERT-like Model
class BERTLikeModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12,
                 d_ff=3072, max_length=512, dropout=0.1, num_classes=1):
        super().__init__()

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.token_type_embedding = nn.Embedding(2, d_model)  # 0 for sent1, 1 for sent2

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Classification head with stronger regularization
        self.pooler = nn.Linear(d_model, d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout * 2),  # Increased dropout for classification head
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),  # Add layer norm
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        seq_length = input_ids.size(1)

        # Position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)

        embeddings = token_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.token_type_embedding(token_type_ids)
            embeddings += token_type_embeds

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Transformer layers
        hidden_states = embeddings
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Pool the [CLS] token representation
        cls_output = hidden_states[:, 0]  # [CLS] token
        pooled_output = torch.tanh(self.pooler(cls_output))

        # Classification
        logits = self.classifier(pooled_output)

        return logits.squeeze(-1)  # Remove last dimension for binary classification

def train_loop(model, train_loader, val_loader, epochs=10, lr=2e-5,
               device="cuda" if torch.cuda.is_available() else "cpu",
               ckpt_path="bert_best.pt",
               warmup_steps=200, patience=3):  # Reduced for Colab

    model.to(device)
    print(f"Training on device: {device}")

    # AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, eps=1e-8)

    # Simpler scheduler for Colab
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    # Disable mixed precision for Colab stability
    use_amp = False  # Set to False for Colab compatibility
    scaler = torch.cuda.amp.GradScaler() if use_amp and device == 'cuda' else None

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Standard training (no mixed precision for Colab)
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            with torch.no_grad():
                predictions = torch.sigmoid(logits) > 0.5
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.size(0)

            # Update progress bar
            if batch_idx % 50 == 0:  # Update every 50 batches
                current_acc = correct_predictions / max(1, total_predictions)
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_acc:.4f}'
                })

            # Clear cache periodically for Colab
            if batch_idx % 100 == 0 and device == 'cuda':
                torch.cuda.empty_cache()

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct_predictions / total_predictions

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels)

                val_loss += loss.item()

                predictions = torch.sigmoid(logits) > 0.5
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        # Step scheduler
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch:02d} | "
              f"train_loss={avg_train_loss:.4f} train_acc={train_accuracy:.4f} | "
              f"val_loss={avg_val_loss:.4f} val_acc={val_accuracy:.4f}")

        # Save best model and early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ Saved new best model to {ckpt_path}")
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter} epochs")

            if patience_counter >= patience:
                print(f"🛑 Early stopping after {patience} epochs without improvement")
                break

        # Clear cache after each epoch for Colab
        if device == 'cuda':
            torch.cuda.empty_cache()

# Create datasets and dataloaders - Colab optimized
train_dataset = BERTDataset(train_df, max_length=128)  # Reduced for Colab memory
val_dataset = BERTDataset(val_df, max_length=128)

# Optimized for Colab GPU memory
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                         collate_fn=bert_collate_fn, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                       collate_fn=bert_collate_fn, num_workers=0, pin_memory=False)

# Model hyperparameters (Colab optimized)
d_model = 256      # Smaller for Colab memory
num_heads = 4      # Fewer attention heads
num_layers = 3     # Reduced layers for Colab
d_ff = 1024        # Smaller feed forward
dropout = 0.2      # Increased dropout for regularization
max_length = 128   # Reduced sequence length for memory

# Training hyperparameters
epochs = 10        # Reduced for Colab session limits
lr = 3e-5         # Learning rate
ckpt_path = 'bert_best.pt'

# Initialize model
model = BERTLikeModel(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    d_ff=d_ff,
    max_length=max_length,
    dropout=dropout
)

print(f"Training BERT-like model with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")

train_loop(model, train_loader, val_loader, epochs=epochs, lr=lr, ckpt_path=ckpt_path)

# Inference function
def infer_bert(csv_file, ckpt_path, vocab_to_id, device='cuda', batch_size=32,
               max_length=256, out_path="predictions.txt"):

    device = device if torch.cuda.is_available() else 'cpu'

    # Load test data
    df = pd.read_csv(csv_file)
    for col in ["sentence_1", "sentence_2"]:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Apply vocabulary
    df = apply_vocab(df, vocab_to_id, UNK_ID)

    # Create dataset without labels
    class TestDataset(Dataset):
        def __init__(self, df, max_length=256):
            self.s1 = df["sentence_1"].tolist()
            self.s2 = df["sentence_2"].tolist()
            self.max_length = max_length

        def __len__(self):
            return len(self.s1)

        def __getitem__(self, idx):
            s1, s2 = self.s1[idx], self.s2[idx]

            # Create BERT-style input
            max_s1_len = self.max_length - 4
            max_s2_len = self.max_length - len(s1[:max_s1_len]) - 3

            s1_truncated = s1[:max_s1_len]
            s2_truncated = s2[:max_s2_len]

            input_ids = [CLS_ID] + s1_truncated + [SEP_ID] + s2_truncated + [SEP_ID]
            attention_mask = [1] * len(input_ids)
            token_type_ids = ([0] * (len(s1_truncated) + 2) +
                             [1] * (len(s2_truncated) + 1))

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }

    def test_collate_fn(batch):
        max_len = max(len(item['input_ids']) for item in batch)

        input_ids = []
        attention_masks = []
        token_type_ids = []

        for item in batch:
            pad_len = max_len - len(item['input_ids'])
            input_ids.append(item['input_ids'] + [PAD_ID] * pad_len)
            attention_masks.append(item['attention_mask'] + [0] * pad_len)
            token_type_ids.append(item['token_type_ids'] + [0] * pad_len)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
        }

    test_dataset = TestDataset(df, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=test_collate_fn)

    # Load model
    model = BERTLikeModel(
        vocab_size=len(vocab_to_id),
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_length=max_length,
        dropout=dropout
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Predict
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            preds = (torch.sigmoid(logits) > 0.5).long().cpu().tolist()
            predictions.extend(preds)

    # Save predictions
    with open(out_path, "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")

    print(f"Saved {len(predictions)} predictions to {out_path}")

# Run inference
infer_bert('sample_data/public_test.csv', ckpt_path, vocab_to_id)
