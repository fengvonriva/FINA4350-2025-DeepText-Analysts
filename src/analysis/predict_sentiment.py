import torch
import pandas as pd
from torchtext.data.utils import get_tokenizer
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
import os
import sys
import torchtext

torchtext.disable_torchtext_deprecation_warning()

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 从当前目录导入NBoW
from NBoW import NBoW

def predict_sentiment(text, model, tokenizer, vocab, max_length, device):
    # Handle NaN or non-string values
    if pd.isna(text) or not isinstance(text, str):
        return None
        
    # Tokenize
    tokens = tokenizer(text)[:max_length]
    
    # Convert to ids
    ids = [vocab[token] for token in tokens]
    
    # Padding
    if len(ids) < max_length:
        ids = ids + [vocab['<pad>']] * (max_length - len(ids))
    else:
        ids = ids[:max_length]
    
    # Convert to tensor
    tensor = torch.LongTensor(ids).unsqueeze(0).to(device)  # Add batch dimension
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        prediction = model(tensor)
        probability = torch.softmax(prediction, dim=1)
        sentiment_score = probability[:, 1].item()  # Probability of positive sentiment
    
    return sentiment_score

def process_reddit_comments(comments_file_path, model, tokenizer, vocab, max_length, device):
    # Read comments
    df = pd.read_excel(comments_file_path) 
    
    # Process each comment
    sentiment_scores = []
    for comment in df['Comment Text']:
        try:
            score = predict_sentiment(comment, model, tokenizer, vocab, max_length, device)
            sentiment_scores.append(score)
        except Exception as e:
            print(f"Error processing comment: {e}")
            sentiment_scores.append(None)
    
    # Add scores to dataframe
    df['sentiment_score'] = sentiment_scores

    # 保证 sentiment_score 和 Comment Upvotes 都是数值型，无法转换的变为 NaN
    df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
    df['Comment Upvotes'] = pd.to_numeric(df['Comment Upvotes'], errors='coerce')
    # 删除有缺失值的行
    df = df.dropna(subset=['sentiment_score', 'Comment Upvotes'])

    # Convert Comment Time to datetime，遇到脏数据自动变NaT
    df['datetime'] = pd.to_datetime(df['Comment Time'], errors='coerce')
    # 删除无法解析日期的行
    df = df.dropna(subset=['datetime'])
    df['date'] = df['datetime'].dt.date
    
    # Calculate daily average sentiment with weights based on upvotes
    df['weighted_sentiment'] = df['sentiment_score'] * df['Comment Upvotes']
    daily_sentiment = df.groupby('date').agg({
        'sentiment_score': ['mean', 'count'],
        'weighted_sentiment': 'sum',
        'Comment Upvotes': 'sum'
    }).reset_index()
    
    # Calculate weighted average sentiment
    daily_sentiment['weighted_avg_sentiment'] = (
        daily_sentiment['weighted_sentiment']['sum'] / 
        daily_sentiment['Comment Upvotes']['sum']
    )
    
    return df, daily_sentiment

def load_trained_model(model_path, vocab_size, embedding_dim, output_dim, pad_index):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = NBoW(vocab_size, embedding_dim, output_dim, pad_index)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    return model, device

def plot_sentiment_analysis(daily_sentiment, output_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot regular sentiment
    ax1.plot(daily_sentiment['date'], daily_sentiment['sentiment_score']['mean'], 
             label='Average Sentiment')
    ax1.set_title('Daily Average Sentiment')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sentiment Score')
    ax1.grid(True)
    ax1.legend()
    
    # Plot weighted sentiment
    ax2.plot(daily_sentiment['date'], daily_sentiment['weighted_avg_sentiment'], 
             label='Upvote-Weighted Sentiment', color='orange')
    ax2.set_title('Daily Weighted Average Sentiment (by Upvotes)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Weighted Sentiment Score')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    plt.show()

def load_vocab(vocab_path):
    with open(vocab_path, 'rb') as f:
        return pickle.load(f)

# Main execution
if __name__ == "__main__":
    # 设置路径
    base_path = "D:/sjtucyx iii/2425/FINA4350"
    model_path = f"{base_path}/BTC/src/analysis/nbow.pt"
    vocab_path = f"{base_path}/BTC/src/analysis/vocab.pkl"
    comments_file = f"{base_path}/BTC/BTC DATA/bitcoin_reddit_comments.xlsx"
    
    # 加载词汇表
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # 初始化模型参数
    vocab_size = len(vocab)
    embedding_dim = 300
    output_dim = 2
    pad_index = vocab['<pad>']
    
    # 加载预训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NBoW(vocab_size, embedding_dim, output_dim, pad_index)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    # 初始化分词器
    tokenizer = get_tokenizer("basic_english")
    
    # 处理评论数据
    df, daily_sentiment = process_reddit_comments(
        comments_file, 
        model, 
        tokenizer, 
        vocab, 
        max_length=256, 
        device=device
    )
    
    # 保存结果
    output_dir = f"{base_path}/BTC/BTC DATA/output"
    df.to_csv(f'{output_dir}/reddit_comments_with_sentiment.csv', index=False)
    daily_sentiment.to_csv(f'{output_dir}/reddit_daily_sentiment.csv', index=False)
    
    # 绘制并保存图表
    plot_sentiment_analysis(daily_sentiment, f'{output_dir}/reddit_sentiment_analysis.pdf')
