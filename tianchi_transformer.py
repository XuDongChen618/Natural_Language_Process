# tianchi_cnn_transformer_lstm.py
import csv
import math
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

warnings.filterwarnings("ignore")

# ------------------- 超参数 -------------------
EMBEDDING_DIM = 384          # 词向量 / 模型统一维度
CNN_KERNELS = [3, 4, 5, 6, 7]
CNN_FILTERS = 384
TRANS_LAYERS = 2
TRANS_HEADS = 8
TRANS_FF = 1024
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
MAX_SEQ_LEN = 384
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3
NUM_CLASSES = 14

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label2id = {
    '科技': 0, '股票': 1, '体育': 2, '娱乐': 3,
    '时政': 4, '社会': 5, '教育': 6, '财经': 7,
    '家居': 8, '游戏': 9, '房产': 10, '时尚': 11,
    '彩票': 12, '星座': 13
}

# ------------------- Dataset -------------------
class NewsDataset(Dataset):
    def __init__(self, df, vocab, max_len, is_test=False):
        self.df = df
        self.vocab = vocab
        self.max_len = max_len
        self.is_test = is_test
        self.data, self.labels = [], []
        self._build()

    def _build(self):
        pad_id = self.vocab['<PAD>']
        unk_id = self.vocab['<UNK>']
        for _, row in self.df.iterrows():
            text = row['text']
            seq = [self.vocab.get(c, unk_id) for c in text.strip()][:self.max_len]
            seq = seq + [pad_id] * (self.max_len - len(seq))
            self.data.append(seq)
            if not self.is_test:
                self.labels.append(int(row['label']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.long)
        if self.is_test:
            return x
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# ------------------- 词汇表 -------------------
def build_vocab(df):
    counter = defaultdict(int)
    for text in df['text']:
        for ch in text:
            counter[ch] += 1
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for ch in counter:
        vocab[ch] = len(vocab)
    return vocab

# ------------------- Word2Vec 预训练 -------------------
def train_word2vec(df, vocab, embed_dim=EMBEDDING_DIM):
    sentences = [[ch for ch in text] for text in df['text']]
    w2v = Word2Vec(sentences, vector_size=embed_dim, window=5, min_count=1, sg=1, workers=4)
    matrix = np.random.normal(0, 0.1, (len(vocab), embed_dim)).astype(np.float32)
    for ch, idx in vocab.items():
        if ch in w2v.wv:
            matrix[idx] = w2v.wv[ch]
    return matrix

# ------------------- 模型 -------------------
class CNN_Transformer_LSTM_Model(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim=EMBEDDING_DIM,
                 cnn_kernels=(3, 4, 5, 6, 7),
                 cnn_filters=384,
                 trans_layers=2,
                 trans_heads=8,
                 trans_ff=1024,
                 lstm_hidden=256,
                 lstm_layers=2,
                 max_len=MAX_SEQ_LEN,
                 num_classes=14,
                 dropout=0.1):
        super().__init__()

        # 1. Embedding（加载 Word2Vec 权重）
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 2. CNN 多尺度卷积
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, cnn_filters, k, padding=k // 2),
                nn.ReLU()
            ) for k in cnn_kernels
        ])
        cnn_out = len(cnn_kernels) * cnn_filters
        self.cnn_proj = nn.Linear(cnn_out, embed_dim)

        # 3. Positional Encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))

        # 4. Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=trans_heads,
            dim_feedforward=trans_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=trans_layers)

        # 5. Bi-LSTM
        self.lstm = nn.LSTM(
            embed_dim,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        # 6. Attention Pooling
        self.attn = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden * 2),
            nn.Tanh(),
            nn.Linear(lstm_hidden * 2, 1)
        )

        # 7. 分类头
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        x : [B, L] 字符 id
        return logits : [B, num_classes]
        """
        # 1. 基本掩码
        mask = (x != 0)  # [B, L]  True=有效字符

        # 2. Embedding + 位置编码
        x = self.embed(x) + self.pos_embed[:, :x.size(1), :]  # [B, L, E]

        # 3. 多尺度 CNN
        cnn_in = x.transpose(1, 2)  # [B, E, L]
        # 3.1 各 kernel 卷积 + ReLU，长度统一截断到 MAX_SEQ_LEN
        conv_outs = [F.relu(conv(cnn_in))[:, :, :MAX_SEQ_LEN]  # [B, C, L]
                     for conv in self.convs]
        # 3.2 在 channel 维拼接 -> [B, C*K, L]
        conv_out = torch.cat(conv_outs, dim=1)
        # 3.3 转回 [B, L, C*K] 并线性映射回 E 维
        conv_out = conv_out.transpose(1, 2)  # [B, L, C*K]
        x = self.cnn_proj(conv_out)  # [B, L, E]

        # 4. Transformer Encoder
        key_pad_mask = ~mask  # True=pad
        x = self.transformer(x, src_key_padding_mask=key_pad_mask)

        # 5. Bi-LSTM（支持变长）
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True)  # [B, L, 2H]

        # 6. Attention Pooling
        attn_weights = self.attn(lstm_out).squeeze(-1)  # [B, L]
        attn_weights = attn_weights.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(-1)
        pooled = (lstm_out * attn_weights).sum(dim=1)  # [B, 2H]

        # 7. 分类
        logits = self.fc(pooled)
        return logits

    def compute_loss(self, logits, labels):
        return self.loss_fn(logits, labels)

# ------------------- 训练验证测试 -------------------
def validate(model, loader):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.
    with torch.no_grad():
        for batch in loader:
            x, y = [t.to(DEVICE) for t in batch]
            logits = model(x)
            loss = model.compute_loss(logits, y)
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    acc = correct / total
    loss = loss_sum / total
    print(f'Val Loss={loss:.4f}, Acc={acc:.4f}')
    return loss, acc

def train(model, train_loader, val_loader, optimizer, epochs):
    best_acc = 0.
    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = model.compute_loss(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        print(f'Epoch {epoch}/{epochs} | Loss={loss_sum/total:.4f}, Acc={correct/total:.4f}')
        val_loss, val_acc = validate(model, val_loader)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print('Best model saved.')

def test(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for x in loader:
            x = x.to(DEVICE)
            logits = model(x)
            preds.extend(logits.argmax(1).cpu().tolist())
    return preds

# ------------------- 主函数 -------------------
def main():
    train_df = pd.read_csv('train_set.csv', sep='\t')
    test_df = pd.read_csv('test_a.csv', sep='\t')
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # 随机过采样
    def oversample(df):
        cls_counts = df['label'].value_counts()
        max_count = cls_counts.max()
        dfs = []
        for cls, sub in df.groupby('label'):
            if len(sub) < max_count:
                sub = sub.sample(max_count, replace=True, random_state=42)
            dfs.append(sub)
        df = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        return df
    train_df, val_df = oversample(train_df), oversample(val_df)

    vocab = build_vocab(train_df)
    embed_path = 'embedding_matrix.npy'
    if os.path.exists(embed_path):
        embed_matrix = np.load(embed_path)
    else:
        embed_matrix = train_word2vec(train_df, vocab)
        np.save(embed_path, embed_matrix)

    train_set = NewsDataset(train_df, vocab, MAX_SEQ_LEN)
    val_set   = NewsDataset(val_df,   vocab, MAX_SEQ_LEN)
    test_set  = NewsDataset(test_df,  vocab, MAX_SEQ_LEN, is_test=True)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

    model = CNN_Transformer_LSTM_Model(
        vocab_size=len(vocab),
        embed_dim=EMBEDDING_DIM,
        cnn_kernels=CNN_KERNELS,
        cnn_filters=CNN_FILTERS,
        trans_layers=TRANS_LAYERS,
        trans_heads=TRANS_HEADS,
        trans_ff=TRANS_FF,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        max_len=MAX_SEQ_LEN,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    model.embed.weight.data.copy_(torch.from_numpy(embed_matrix))
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    train(model, train_loader, val_loader, optimizer, EPOCHS)

    preds = test(model, test_loader)
    with open('test_predictions.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for p in preds:
            writer.writerow([p])
    print('预测完成，已保存 test_predictions.csv')

if __name__ == '__main__':
    main()