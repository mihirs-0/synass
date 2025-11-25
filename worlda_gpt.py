#!/usr/bin/env python
# arrow_world_a_v3.py
# Version: 1.1 (Final Production - Full Logging & Param Auditing)
# 
# "World A" Experimental Testbed
# ------------------------------
# Context: Pure Optimization Dynamics & Thermodynamic Efficiency.
# Goal: Measure Excess Loss (Train Loss - Entropy Floor) to isolate structural friction.
#
# Regimes:
#   1. Scratch:      Random Init (Tests architecture neutrality)
#   2. Finetune:     Pretrained (Tests dense gradient efficiency)
#   3. Finetune_Reg: Pretrained + High Dropout/WD (Tests if noise is the issue)
#   4. LoRA:         Pretrained + Rank Constraint (Tests manifold hypothesis)
#
# Changes in V3:
#   - Added Parameter Counting (Total vs Trainable)
#   - Added Wall-Clock Timing
#   - Added Dataset Topology Stats (Unique A/B)
#   - Full Hyperparameter Logging (No implicit defaults)

import argparse
import json
import math
import random
import string
import sys
import uuid
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Literal
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

# Imports
try:
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        AutoModelForCausalLM,
        get_linear_schedule_with_warmup,
    )
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    print("Error: Missing libraries. Run: pip install transformers peft torch")
    sys.exit(1)

# Determinism for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHABET = string.ascii_lowercase + string.digits

@dataclass
class T1Pair:
    A: str
    B: str

# =============================================================================
# 1. World A Data Generation (No Splitting)
# =============================================================================

def _rand_str(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(ALPHABET) for _ in range(length))

def generate_pairs(K: int, n_pairs: int, str_len: int, seed: int) -> List[T1Pair]:
    rng = random.Random(seed)
    
    if K == 1:
        # Bijective: Unique A <-> Unique B
        A_set = set()
        while len(A_set) < n_pairs: A_set.add(_rand_str(rng, str_len))
        A_list = list(A_set)
        
        B_set = set()
        while len(B_set) < n_pairs: B_set.add(_rand_str(rng, str_len))
        B_list = list(B_set)
        rng.shuffle(B_list)
        return [T1Pair(A=a, B=b) for a, b in zip(A_list, B_list)]

    else:
        # Many-to-One: K unique As -> 1 unique B
        assert n_pairs % K == 0, f"n_pairs ({n_pairs}) must be divisible by K ({K})"
        n_targets = n_pairs // K
        targets = set()
        while len(targets) < n_targets: targets.add(_rand_str(rng, str_len))
        target_list = list(targets)
        
        pairs = []
        for b in target_list:
            # Create K conflicts for this B
            for _ in range(K):
                pairs.append(T1Pair(A=_rand_str(rng, str_len), B=b))
        
        rng.shuffle(pairs)
        return pairs

# =============================================================================
# 2. Dataset (100% Train)
# =============================================================================

class HFDataset(Dataset):
    def __init__(self, pairs: List[T1Pair], tokenizer, direction: str, max_len: int):
        self.samples = []
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        for p in pairs:
            # Topology definition:
            # A->B: Deterministic (Entropy 0)
            # B->A: Probabilistic (Entropy ln K) if K > 1
            src, tgt = (p.A, p.B) if direction == "A->B" else (p.B, p.A)
            
            # Format: "x: {input} y: {target}"
            prompt = f"x: {src} y:"
            full_text = f"{prompt} {tgt}"
            
            enc = tokenizer(full_text, truncation=True, max_length=max_len, 
                            padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"][0]
            attn_mask = enc["attention_mask"][0]
            
            # Masking: Loss is calculated ONLY on the target output
            labels = input_ids.clone()
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            labels[:len(prompt_ids)] = -100
            labels[input_ids == tokenizer.pad_token_id] = -100
            
            self.samples.append({"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels})

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# =============================================================================
# 3. Training Engine
# =============================================================================

def train_run(run_id: int, run_type: str, config: dict, pairs: List[T1Pair], direction: str, seed: int):
    # --- Timing Start ---
    start_time = time.time()
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # --- Theoretical Floors ---
    if direction == "A->B":
        theoretical_min = 0.0
    else:
        theoretical_min = math.log(config['current_K']) if config['current_K'] > 0 else 0.0

    print(f"\n[RUN {run_id}] {run_type.upper()} | {direction} | K={config['current_K']} | Floor={theoretical_min:.4f}")
    
    # Quick Summary of Regime
    wd = config.get('weight_decay', 0.01)
    lr = config.get('lr')
    if run_type == "scratch":
        print(f"  [Info] Scratch Init | LR={lr} | WD={wd}")
    elif run_type == "finetune":
        print(f"  [Info] Pretrained | LR={lr} | WD={wd}")
    elif run_type == "finetune_reg":
        print(f"  [Info] Pretrained + REG | LR={lr} | WD={wd} | Dropout={config['reg_dropout']}")
    elif run_type == "lora":
        print(f"  [Info] LoRA (r={config['lora_r']}) | Scaled LR={lr} | WD={wd}")

    # 1. Setup
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    if run_type == "scratch":
        hf_config = AutoConfig.from_pretrained(config['model_id'])
        model = AutoModelForCausalLM.from_config(hf_config).to(DEVICE)
    elif run_type == "finetune":
        model = AutoModelForCausalLM.from_pretrained(config['model_id']).to(DEVICE)
    elif run_type == "finetune_reg":
        # The Reviewer #2 Baseline: Inject high dropout into config before loading
        hf_config = AutoConfig.from_pretrained(config['model_id'])
        hf_config.attn_pdrop = config['reg_dropout']
        hf_config.resid_pdrop = config['reg_dropout']
        hf_config.embd_pdrop = config['reg_dropout']
        model = AutoModelForCausalLM.from_pretrained(config['model_id'], config=hf_config).to(DEVICE)
    elif run_type == "lora":
        base = AutoModelForCausalLM.from_pretrained(config['model_id'])
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config['lora_r'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            target_modules=["c_attn", "c_proj"]
        )
        model = get_peft_model(base, peft_config).to(DEVICE)
    
    model.config.pad_token_id = tokenizer.pad_token_id

    # --- Parameter Counting (The Reviewer Check) ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if run_type == "lora":
        # For LoRA, trainable params ARE the adapter params
        lora_trainable_params = trainable_params
        ratio = (trainable_params / total_params) * 100
        print(f"  [Params] Total: {total_params} | Trainable: {trainable_params} ({ratio:.2f}%)")
    else:
        lora_trainable_params = None
        print(f"  [Params] Total: {total_params} | Trainable: {trainable_params} (100%)")

    # 2. Data
    train_ds = HFDataset(pairs, tokenizer, direction, config['max_len'])
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    
    # 3. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * len(train_loader) * config['epochs']),
        num_training_steps=len(train_loader) * config['epochs']
    )
    
    # 4. Loop
    train_losses = []
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            
        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)
        
        # Log progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Ep {epoch+1}: Loss {avg_train:.4f} (Excess: {avg_train - theoretical_min:.4f})")

    final_loss = train_losses[-1]
    excess_loss = final_loss - theoretical_min
    
    # --- Timing End ---
    run_wall_time = time.time() - start_time
    
    print(f"  >>> Final: Loss {final_loss:.4f} | Excess {excess_loss:.4f} | Time: {run_wall_time:.1f}s")

    return {
        "world": "A",
        "exp_id": config.get('exp_id', 'unknown'),
        "run_id": run_id,
        "run_type": run_type,
        "K": config['current_K'],
        "direction": direction,
        "seed": seed,
        "model_id": config['model_id'],
        "n_pairs": len(pairs),
        "batch_size": config['batch_size'],
        "str_len": config['str_len'],
        "max_len": config['max_len'],
        "epochs": config['epochs'],
        "lr": lr,
        "weight_decay": wd,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "lora_trainable_params": lora_trainable_params,
        "lora_alpha": config.get("lora_alpha", None),
        "lora_dropout": config.get("lora_dropout", None),
        "reg_dropout": config.get('reg_dropout', None),
        "final_train_loss": final_loss,
        "theoretical_min": theoretical_min,
        "excess_loss": excess_loss,
        "run_wall_time_sec": run_wall_time,
        "log_base": "e",
        "lora_r": config.get('lora_r', None),
        "train_curve": train_losses
    }

# =============================================================================
# 4. CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    # Regimes
    parser.add_argument("--modes", nargs="+", default=["scratch", "finetune", "finetune_reg", "lora"], 
                        choices=["scratch", "finetune", "finetune_reg", "lora"])
    
    # Task Complexity
    parser.add_argument("--Ks", type=str, default="1,5,8", help="Branching factors")
    parser.add_argument("--n_pairs", type=int, default=40000, help="Total dataset size (force optimization)")
    parser.add_argument("--str_len", type=int, default=8, help="Length of random strings")
    
    # Training Hyperparams
    parser.add_argument("--epochs", type=int, default=20, help="Ensure convergence")
    parser.add_argument("--batch_size", type=int, default=64, help="Updated default for speed")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base Learning Rate")
    parser.add_argument("--seeds", type=str, default="0")
    
    # Model
    parser.add_argument("--model_id", type=str, default="gpt2", help="gpt2 (small) recommended")
    
    # LoRA Config
    parser.add_argument("--lora_ranks", type=str, default="8,64,256")
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    # Regularization Config (Finetune_Reg)
    parser.add_argument("--reg_weight_decay", type=float, default=0.1)
    parser.add_argument("--reg_dropout", type=float, default=0.1)
    
    parser.add_argument("--output", type=str, default="results_world_a_final.jsonl")
    
    args = parser.parse_args()
    
    exp_id = str(uuid.uuid4())[:8]
    Ks = [int(x) for x in args.Ks.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    lora_ranks = [int(x) for x in args.lora_ranks.split(",")]
    
    # --- SANITY HEADER ---
    print("\n" + "#"*60)
    print(f" EXPERIMENT ID: {exp_id}")
    print(f" Model: {args.model_id}")
    print(f" Modes: {args.modes}")
    print(f" Ks: {Ks}")
    print(f" Pairs: {args.n_pairs}")
    print(f" LoRA Ranks: {lora_ranks}")
    print(f" Base LR: {args.lr}")
    print(f" Reg Baseline: WD={args.reg_weight_decay}, Drop={args.reg_dropout}")
    print("#"*60 + "\n")

    run_id_counter = 0

    for K in Ks:
        print(f"\n" + "="*60 + f"\n WORLD A DATASET (K={K}) - 100% TRAIN \n" + "="*60)
        # Generate Data ONCE per K so all models see identical manifold
        base_pairs = generate_pairs(K, args.n_pairs, args.str_len, seed=42 + K)
        
        # --- Dataset Stats ---
        unique_As = {p.A for p in base_pairs}
        unique_Bs = {p.B for p in base_pairs}
        print(f"  [Dataset] n_pairs={len(base_pairs)} | Unique A={len(unique_As)} | Unique B={len(unique_Bs)}")
        if K > 1:
            expected_Bs = args.n_pairs // K
            print(f"  [Topology Check] Expected Unique B = {expected_Bs}. Actual = {len(unique_Bs)}. " + 
                  ("PASS" if len(unique_Bs) == expected_Bs else "FAIL"))
        print("="*60)

        for seed in seeds:
            for mode in args.modes:
                base_cfg = {
                    "exp_id": exp_id,
                    "model_id": args.model_id,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "max_len": 64,
                    "current_K": K,
                    "str_len": args.str_len,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": 0.05
                }
                
                configs = []
                
                if mode == "lora":
                    # LoRA LR Scaling Logic
                    for r in lora_ranks:
                        c = base_cfg.copy()
                        c['lora_r'] = r
                        c['weight_decay'] = 0.01 
                        
                        # Scale LR: Base * (r / 8) -> Cap at 1e-3
                        scaled_lr = args.lr * (r / 8.0)
                        c['lr'] = min(scaled_lr, 1e-3)
                        configs.append(c)
                        
                elif mode == "finetune_reg":
                    c = base_cfg.copy()
                    c['lr'] = args.lr
                    c['weight_decay'] = args.reg_weight_decay
                    c['reg_dropout'] = args.reg_dropout
                    configs.append(c)
                    
                else: # scratch or finetune
                    c = base_cfg.copy()
                    c['lr'] = args.lr
                    c['weight_decay'] = 0.01
                    configs.append(c)
                
                for cfg in configs:
                    for direction in ["A->B", "B->A"]:
                        run_id_counter += 1
                        res = train_run(run_id_counter, mode, cfg, base_pairs, direction, seed)
                        
                        with open(args.output, "a") as f:
                            f.write(json.dumps(res) + "\n")

if __name__ == "__main__":
    main()