import collections
import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
import pickle
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


torchtext.disable_torchtext_deprecation_warning()

# 工具函数和NBoW类

def process_binary_sentiment_data(df):
    # Create a copy of DataFrame instead of view
    df = df.copy()
    
    # Convert sentiment column string to dictionary
    df['sentiment_dict'] = df['sentiment'].apply(ast.literal_eval)
    
    # Extract sentiment class
    df['sentiment_class'] = df['sentiment_dict'].apply(lambda x: x['class'])
    
    # Keep only positive and negative
    df = df[df['sentiment_class'].isin(['positive', 'negative'])].copy()
    
    # Set values using loc
    df.loc[:, 'label'] = (df['sentiment_class'] == 'positive').astype(int)
    
    # Create final dataframe
    final_df = pd.DataFrame({
        'text': df['text'],
        'label': df['label']
    })
    
    return final_df

def tokenize_example(text, tokenizer, max_length=256):
    return tokenizer(text)[:max_length]

def yield_tokens(data_iter):
    for tokens in data_iter:
        yield tokens

def numericalize_tokens(tokens, vocab):
    return [vocab[token] for token in tokens]

def pad_sequence(sequence, max_length, pad_index):
    """
    Pad sequence to specified length
    """
    if len(sequence) > max_length:
        return sequence[:max_length]
    else:
        return sequence + [pad_index] * (max_length - len(sequence))

def process_data(df, vocab, max_length):
    """
    Process dataset, return tensors ready for training
    """
    # Numericalize
    df['ids'] = df['tokens'].apply(lambda x: numericalize_tokens(x, vocab))
    
    # Padding
    df['padded_ids'] = df['ids'].apply(lambda x: pad_sequence(x, max_length, vocab['<pad>']))
    
    # Convert to tensors
    X = torch.tensor(df['padded_ids'].tolist())
    y = torch.tensor(df['label'].tolist())
    
    return X, y

class CryptoDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return {
            'ids': self.X[idx],
            'label': self.y[idx]
        }

def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids": batch_ids, "label": batch_label}
        return batch

    return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader

class NBoW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, ids):
        # ids = [batch size, seq len]
        embedded = self.embedding(ids)
        # embedded = [batch size, seq len, embedding dim]
        pooled = embedded.mean(dim=1)
        # pooled = [batch size, embedding dim]
        prediction = self.fc(pooled)
        # prediction = [batch size, output dim]
        return prediction

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(data_loader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(data_loader, desc="training..."):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)
        prediction = model(ids)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)

def evaluate(data_loader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            prediction = model(ids)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)

def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy

if __name__ == "__main__":
    # 设置随机种子
    seed = 1029
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # 读取数据
    pretrain_data = pd.read_csv("D:/sjtucyx iii/2425/FINA4350/BTC/BTC DATA/train data/cryptonews.csv", encoding='latin-1')
    processed_df = process_binary_sentiment_data(pretrain_data)

    # 划分数据集
    train_data, test_data = train_test_split(processed_df, test_size=0.2, random_state=seed)
    tokenizer = get_tokenizer("basic_english")
    train_data['tokens'] = train_data['text'].apply(lambda x: tokenize_example(x, tokenizer))
    test_data['tokens'] = test_data['text'].apply(lambda x: tokenize_example(x, tokenizer))
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=seed)

    specials = ['<unk>', '<pad>']
    vocab = build_vocab_from_iterator(
        yield_tokens(train_data['tokens']),
        min_freq=3,
        specials=specials
    )
    vocab.set_default_index(vocab['<unk>'])
    unk_index = vocab['<unk>']
    pad_index = vocab['<pad>']

    max_length = 256
    X_train, y_train = process_data(train_data, vocab, max_length)
    X_val, y_val = process_data(val_data, vocab, max_length)
    X_test, y_test = process_data(test_data, vocab, max_length)

    train_dataset = CryptoDataset(X_train, y_train)
    val_dataset = CryptoDataset(X_val, y_val)
    test_dataset = CryptoDataset(X_test, y_test)

    batch_size = 512
    train_loader = get_data_loader(train_dataset, batch_size, pad_index, shuffle=True)
    val_loader = get_data_loader(val_dataset, batch_size, pad_index)
    test_loader = get_data_loader(test_dataset, batch_size, pad_index)

    vocab_size = len(vocab)
    embedding_dim = 300
    output_dim = len(train_data['label'].unique())
    model = NBoW(vocab_size, embedding_dim, output_dim, pad_index)
    print(f"模型参数量: {count_parameters(model):,}")

    vectors = torchtext.vocab.GloVe(name='6B', dim=300, cache='D:/sjtucyx iii/2425/FINA4350/.vector_cache')
    pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
    model.embedding.weight.data = pretrained_embedding

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)

    n_epochs = 30
    best_valid_loss = float("inf")
    metrics = collections.defaultdict(list)

    for epoch in range(n_epochs):
        train_loss, train_acc = train(
            train_loader, model, criterion, optimizer, device
        )
        valid_loss, valid_acc = evaluate(val_loader, model, criterion, device)
        metrics["train_losses"].append(train_loss)
        metrics["train_accs"].append(train_acc)
        metrics["valid_losses"].append(valid_loss)
        metrics["valid_accs"].append(valid_acc)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "BTC/src/analysis/nbow.pt")
        print(f"epoch: {epoch}")
        print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
        print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")

    # 绘制loss/acc曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(metrics["train_losses"], label="Training Loss")
    ax1.plot(metrics["valid_losses"], label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(metrics["train_accs"], label="Training Accuracy")
    ax2.plot(metrics["valid_accs"], label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True)
    plt.savefig("BTC/src/analysis/nbow_train_curve.pdf")
    plt.close()

    # 保存模型和词表
    with open("BTC/src/analysis/vocab.pkl", 'wb') as f:
        pickle.dump(vocab, f)


