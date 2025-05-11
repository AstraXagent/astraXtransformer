import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
from stable_baselines3.common import logger

# Helper: Rotary Positional Encoding (same as before)
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model))
        pe = torch.zeros(seq_len, self.d_model)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        return x + pe[:seq_len].unsqueeze(0)

# Embedding Layer with Rotary Positional Encoding (same as before)
class EmbeddingWithRotaryPositionalEncoding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rotary_pos_enc = RotaryPositionalEncoding(d_model, max_len)
        
    def forward(self, x):
        embedded = self.embedding(x)
        return self.rotary_pos_enc(embedded)

# Memory Mechanism (same as before)
class Memory(nn.Module):
    def __init__(self, d_model, memory_type="long_term"):
        super().__init__()
        self.d_model = d_model
        self.memory_type = memory_type
        self.memory = torch.zeros(1, 100, d_model)  # Initialize with dummy memory
        self.attn = nn.MultiheadAttention(d_model, num_heads=8)
        
    def forward(self, x):
        if self.memory_type == "short_term":
            return x  # Simple pass-through for short-term memory
        elif self.memory_type == "long_term":
            # Apply memory mechanism to provide context.
            memory_out, _ = self.attn(x, self.memory, self.memory)
            return memory_out
        return x

# Scaled Dot-Product Attention (same as before)
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output, attn

# Multi-Head Attention Layer (same as before)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores, attn = scaled_dot_product_attention(q, k, v, mask)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.num_heads * self.d_k)
        return self.out(concat)

# Feed Forward Network (same as before)
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# Encoder Layer (same as before)
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout(self.attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x

# Decoder Layer (same as before)
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x

# Safety Layer (same as before)
class SafetyLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Check for harmful content in the output
        prohibited_words = ["bomb", "gun", "explosion", "weapon"]  # Add more harmful words or patterns
        for word in prohibited_words:
            if word in x:
                raise ValueError("Harmful content detected!")
        return x

# Full AstraX Transformer Model (with RL Integration)
class AstraXTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, N=6, heads=8, d_ff=2048, dropout=0.1, memory_type="long_term"):
        super().__init__()
        self.embedding = EmbeddingWithRotaryPositionalEncoding(src_vocab, d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.memory = Memory(d_model, memory_type=memory_type)
        self.safety_layer = SafetyLayer()
        self.fc_out = nn.Linear(d_model, tgt_vocab)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embedding and Rotary Positional Encoding
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        
        # Encoder processing with memory
        enc_out = src_emb
        for enc_layer in self.encoder:
            enc_out = enc_layer(enc_out, src_mask)
        
        # Memory update
        enc_out = self.memory(enc_out)

        # Decoder processing
        dec_out = tgt_emb
        for dec_layer in self.decoder:
            dec_out = dec_layer(dec_out, enc_out, src_mask, tgt_mask)
        
        # Safety Layer
        dec_out = self.safety_layer(dec_out)
        
        # Output layer
        return self.fc_out(dec_out)

# Example of initializing and fine-tuning the model with RL
if __name__ == "__main__":
    # Hyperparameters
    SRC_VOCAB_SIZE = 10000
    TGT_VOCAB_SIZE = 10000
    D_MODEL = 512
    N_LAYERS = 6
    N_HEADS = 8
    D_FF = 2048
    DROPOUT = 0.1

    # Instantiate the AstraX Transformer model
    astrax_transformer = AstraXTransformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, d_model=D_MODEL, N=N_LAYERS, heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT)

    # Example random input data for training
    src = torch.randint(0, SRC_VOCAB_SIZE, (32, 10))  # batch_size=32, seq_len=10
    tgt = torch.randint(0, TGT_VOCAB_SIZE, (32, 10))

    # Dummy training loop with RL agent
    env = DummyVecEnv([lambda: astrax_transformer])  # Wrap the model in a Gym environment for RL
    model = PPO("MlpPolicy", env, verbose=1)  # Initialize PPO model

    # Train the model using RL
    model.learn(total_timesteps=10000)  # Train for 10000 timesteps

    # Fine-tuning the model after RL (task-specific)
    # Task-specific fine-tuning could involve supervised learning with task-oriented datasets
