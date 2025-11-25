#!/usr/bin/env python
# arrow_mlp_ablation.py
# 
# "World A" MLP Ablation Testbed
# Context: Test if the Arrow of Time persists in a non-causal, non-attention architecture.
# Goal: Measure Excess Loss (Friction) for A->B vs B->A on a simple MLP.

import argparse
import json
import math
import random
import string
import sys
import time
import uuid
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# 0. Global Config & Data Structures
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHABET = string.ascii_lowercase + string.digits # 36 chars
CHAR2IDX = {ch: i for i, ch in enumerate(ALPHABET)}
VOCAB_SIZE = len(ALPHABET)

@dataclass
class T1Pair:
    A: str
    B: str

def encode_str(s: str) -> List[int]:
    return [CHAR2IDX[ch] for ch in s]

# =============================================================================
# 1. World A Data Generation
# =============================================================================

def _rand_str(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(ALPHABET) for _ in range(length))

def generate_pairs(K: int, n_pairs: int, str_len: int, seed: int) -> List[T1Pair]:
    rng = random.Random(seed)
    
    if K == 1:
        A_set = set()
        while len(A_set) < n_pairs: A_set.add(_rand_str(rng, str_len))
        A_list = list(A_set)
        
        B_set = set()
        while len(B_set) < n_pairs: B_set.add(_rand_str(rng, str_len))
        B_list = list(B_set)
        rng.shuffle(B_list)
        return [T1Pair(A=a, B=b) for a, b in zip(A_list, B_list)]
    else:
        assert n_pairs % K == 0, "n_pairs must be divisible by K"
        n_targets = n_pairs // K
        targets = set()
        while len(targets) < n_targets: targets.add(_rand_str(rng, str_len))
        target_list = list(targets)
        
        pairs = []
        for b in target_list:
            for _ in range(K):
                pairs.append(T1Pair(A=_rand_str(rng, str_len), B=b))
        
        rng.shuffle(pairs)
        return pairs

# =============================================================================
# 2. Dataset & Model
# =============================================================================

class MLPDataset(Dataset):
    def __init__(self, pairs: List[T1Pair], direction: str, str_len: int):
        self.samples = []
        for p in pairs:
            if direction == "A->B":
                src, tgt = p.A, p.B
            else:
                src, tgt = p.B, p.A
            
            x = torch.tensor(encode_str(src), dtype=torch.long)
            y = torch.tensor(encode_str(tgt), dtype=torch.long)
            self.samples.append((x, y))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

class StringMLP(nn.Module):
    def __init__(self, vocab_size, str_len, d_emb=64, d_hidden=512, n_layers=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.str_len = str_len
        
        self.embed = nn.Embedding(vocab_size, d_emb)
        
        input_dim = str_len * d_emb
        output_dim = str_len * vocab_size
        
        layers = []
        dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(dim, d_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(d_hidden))
            dim = d_hidden
        layers.append(nn.Linear(dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        emb = self.embed(x)                 # [B, L, D]
        flat = emb.view(emb.size(0), -1)    # [B, L*D]
        logits = self.mlp(flat)             # [B, L*V]
        logits = logits.view(-1, self.str_len, self.vocab_size)
        return logits

# =============================================================================
# 3. Training Engine
# =============================================================================

def train_mlp_run(run_id: int, K: int, pairs: List[T1Pair], direction: str, config: dict):
    # Seed
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Floor
    if direction == "A->B":
        theoretical_min = 0.0
    else:
        theoretical_min = math.log(K) if K > 0 else 0.0
        
    print(f"\n[MLP RUN {run_id}] {direction} | K={K} | Floor={theoretical_min:.4f}")

    # Data
    ds = MLPDataset(pairs, direction, config['str_len'])
    loader = DataLoader(ds, batch_size=config['batch_size'], shuffle=True)
    
    # Model
    model = StringMLP(
        vocab_size=VOCAB_SIZE, 
        str_len=config['str_len'],
        d_emb=config['d_emb'],
        d_hidden=config['d_hidden'],
        n_layers=config['n_layers']
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    # Loop
    train_losses = []
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        total_samples = 0
        
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(x) 
            
            loss = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE),
                y.view(-1)
            )
            
            loss.backward()
            optimizer.step()
            
            # --- FIXED LOGIC HERE ---
            total_loss += loss.item() * x.size(0) # Accumulate sum of losses
            total_samples += x.size(0)            # Count samples
            
        avg_loss = total_loss / total_samples     # Compute true average
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Ep {epoch+1}: Loss {avg_loss:.4f} (Excess: {avg_loss - theoretical_min:.4f})")
            
    final_loss = train_losses[-1]
    excess_loss = final_loss - theoretical_min
    
    print(f"  >>> Final: Loss {final_loss:.4f} | Excess {excess_loss:.4f}")
    
    return {
        "arch": "MLP",
        "K": K,
        "direction": direction,
        "seed": seed,
        "n_pairs": len(pairs),
        "final_loss": final_loss,
        "theoretical_min": theoretical_min,
        "excess_loss": excess_loss,
        "epochs": config['epochs'],
        "lr": config['lr'],
        "batch_size": config['batch_size'],
        "train_curve": train_losses
    }

# =============================================================================
# 4. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Ks", type=str, default="1,5,8")
    parser.add_argument("--n_pairs", type=int, default=40000)
    parser.add_argument("--str_len", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="results_mlp_ablation.jsonl")
    
    # MLP Specs
    parser.add_argument("--d_emb", type=int, default=64)
    parser.add_argument("--d_hidden", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=4)
    
    args = parser.parse_args()
    
    Ks = [int(x) for x in args.Ks.split(",")]
    exp_id = str(uuid.uuid4())[:8]
    
    print(f"MLP ABLATION | ID: {exp_id} | Pairs: {args.n_pairs} | LR: {args.lr}")
    
    run_id = 0
    for K in Ks:
        print(f"\n=== Generating Data K={K} ===")
        pairs = generate_pairs(K, args.n_pairs, args.str_len, seed=42+K)
        
        config = vars(args)
        config['seed'] = 0
        config['weight_decay'] = 0.0
        
        # A->B
        run_id += 1
        res_fwd = train_mlp_run(run_id, K, pairs, "A->B", config)
        with open(args.output, "a") as f: f.write(json.dumps(res_fwd) + "\n")
        
        # B->A
        run_id += 1
        res_bwd = train_mlp_run(run_id, K, pairs, "B->A", config)
        with open(args.output, "a") as f: f.write(json.dumps(res_bwd) + "\n")

if __name__ == "__main__":
    main()