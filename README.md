# A Controlled Benchmark for Measuring Directional Training Asymmetry in Transformers

This repository contains the experimental code and benchmark for the paper: **"A Controlled Benchmark for Measuring Directional Training Asymmetry in Transformers"** by Mihir Sahasrabudhe.

## Abstract

Transformers are theoretically reversal-invariant: their function class does not prefer left-to-right over right-to-left mappings. Yet empirical studies on natural language repeatedly report a "reversal curse," and recent work on temporal asymmetry in LLMs suggests that real-world corpora carry their own arrow of time. This leaves an unresolved question: **do directional failures stem from linguistic statistics, or from the architecture itself?**

We cut through this ambiguity with a fully synthetic, entropy-controlled benchmark designed as a clean-room stress test for directional learning. Using random string mappings with tunable branching factor $K$, we construct forward tasks with zero conditional entropy and inverse tasks with analytically determined entropy floors. **Excess loss** above these floors reveals that even scratch-trained GPT-2 models exhibit a strong, reproducible directional optimization gap (e.g., 1.16 nats at $K=5$), far larger than that of an MLP trained on the same data. Pre-trained initializations shift optimization behavior but do not eliminate this gap, while LoRA encounters a sharp capacity wall on high-entropy inverse mappings.

Together, these results isolate a minimal, semantics-free signature of **directional friction intrinsic to causal Transformer training**—one that persists even when linguistic priors, token frequencies, and corpus-level temporal asymmetries are removed. Our benchmark provides a controlled instrument for dissecting directional biases in modern sequence models and motivates deeper mechanistic study of why inversion remains fundamentally harder for Transformers.

## Introduction

While Transformer-based Large Language Models (LLMs) excel at sequence modeling, they exhibit a fundamental directional asymmetry. Recent studies report that models trained on causal statements ($A \to B$) often fail to infer the diagnostic inverse ($B \to A$), a phenomenon termed the **"reversal curse"**. This raises a critical question: is this asymmetry an artifact of the training data, or an intrinsic inductive bias of the architecture itself?

Current analyses relying on natural language (e.g., "$A$ is the parent of $B$") are inherently confounded by three factors:

1. **Semantic Priors**: Causal relationships often appear more frequently than diagnostic ones in training corpora.
2. **Linguistic Structure**: Syntax and grammar impose directional dependencies (e.g., Subject-Verb-Object) that favor forward prediction.
3. **Token Statistics**: Entity frequencies and co-occurrence statistics are rarely symmetric.

Consequently, it is difficult to disentangle whether the Reversal Curse arises from data distribution or from the autoregressive factorization mechanism.

### The Arrow of Complexity

This directional puzzle is further complicated by inherent differences in computational complexity. Forward generation often resembles a deterministic collapse of state space (analogous to multiplication), whereas inverse inference requires expanding state space to recover multiple potential inputs (analogous to factorization). In natural language, these entropic differences are inextricably linked to semantics. To isolate the architectural contribution to the Reversal Curse, we require a setting where this forward-inverse complexity asymmetry is explicitly tunable.

### Goal: A Synthetic "Clean Room"

We introduce a controlled benchmark to measure **directional training efficiency** in the absence of linguistic or statistical confounds. We construct a dataset of random string mappings where the topology is strictly controlled by a branching factor $K$:

- **Forward ($A \to B$)**: Deterministic mapping ($H=0$), mimicking low-entropy causal processes.
- **Backward ($B \to A$)**: Probabilistic one-to-many mapping ($H=\ln K$), mimicking high-entropy inverse problems.

This design creates a mathematically precise analogue of the complexity asymmetry described in prior work, stripped of all semantic priors.

### Approach

Within this framework, we benchmark the optimization dynamics of Causal Transformers (trained from scratch and pre-trained) against non-causal Multilayer Perceptrons (MLPs) and Low-Rank Adaptation (LoRA) methods. Crucially, because the information-theoretic floor of each task is known exactly, we report **Excess Loss**—the divergence of the model from the theoretical minimum. This metric allows us to rigorously decouple the inherent thermodynamic difficulty of the inverse task from the structural inefficiencies of the architecture.

## Key Findings

Across 40,000-pair datasets, we observe:

1. **Transformers trained from scratch** exhibit a consistent directional optimization gap (e.g., $\approx 1.16$ nats at $K=5$), substantially larger than the gap observed in MLPs ($\approx 0.22$ nats).
2. **Pre-trained initializations** shift optimization behavior but do not eliminate the directional gap.
3. **Low-rank adaptation (LoRA)** encounters a sharp capacity wall on high-entropy inverse mappings, failing to converge at this scale.

## Repository Structure

```
synass/
├── worlda_gpt.py          # Transformer experiments (GPT-2)
├── arrow_mlp_ablation.py  # MLP baseline ablation
└── README.md              # This file
```

## Installation

### Requirements

```bash
pip install torch transformers peft numpy
```

### Dependencies

- Python 3.7+
- PyTorch
- HuggingFace Transformers
- PEFT (for LoRA support)
- NumPy

## Usage

### Transformer Experiments (`worlda_gpt.py`)

This script implements the "World A" experimental testbed for measuring directional training efficiency in Transformers. It supports four training regimes:

1. **Scratch**: Random initialization (tests architecture neutrality)
2. **Finetune**: Pre-trained weights with full fine-tuning (tests dense gradient efficiency)
3. **Finetune_Reg**: Pre-trained weights with high dropout/weight decay (tests if noise is the issue)
4. **LoRA**: Pre-trained weights with low-rank adaptation (tests manifold hypothesis)

#### Basic Usage

```bash
python worlda_gpt.py \
    --modes scratch finetune finetune_reg lora \
    --Ks 1,5,8 \
    --n_pairs 40000 \
    --epochs 20 \
    --lr 1e-4 \
    --output results_world_a_final.jsonl
```

#### Command-Line Arguments

**Regimes:**
- `--modes`: Training modes to run (default: `scratch finetune finetune_reg lora`)
- `--model_id`: HuggingFace model ID (default: `gpt2`)

**Task Complexity:**
- `--Ks`: Branching factors, comma-separated (default: `1,5,8`)
- `--n_pairs`: Total dataset size (default: `40000`)
- `--str_len`: Length of random strings (default: `8`)

**Training Hyperparameters:**
- `--epochs`: Number of training epochs (default: `20`)
- `--batch_size`: Batch size (default: `64`)
- `--lr`: Base learning rate (default: `1e-4`)
- `--seeds`: Random seeds, comma-separated (default: `0`)

**LoRA Configuration:**
- `--lora_ranks`: LoRA ranks to sweep (default: `8,64,256`)
- `--lora_alpha`: LoRA alpha parameter (default: `32`)

**Regularization (Finetune_Reg):**
- `--reg_weight_decay`: Weight decay for regularized fine-tuning (default: `0.1`)
- `--reg_dropout`: Dropout rate for regularized fine-tuning (default: `0.1`)

**Output:**
- `--output`: Output JSONL file path (default: `results_world_a_final.jsonl`)

#### Example: Run All Regimes

```bash
python worlda_gpt.py \
    --modes scratch finetune finetune_reg lora \
    --Ks 1,5,8 \
    --n_pairs 40000 \
    --epochs 20 \
    --lr 1e-4 \
    --batch_size 64 \
    --seeds 0 \
    --output results_world_a_final.jsonl
```

#### Example: Run Only Scratch and LoRA

```bash
python worlda_gpt.py \
    --modes scratch lora \
    --Ks 5,8 \
    --n_pairs 40000 \
    --epochs 20 \
    --lr 1e-4 \
    --output results_scratch_lora.jsonl
```

### MLP Ablation (`arrow_mlp_ablation.py`)

This script tests whether the directional asymmetry persists in a non-causal, non-attention architecture (MLP baseline).

#### Basic Usage

```bash
python arrow_mlp_ablation.py \
    --Ks 1,5,8 \
    --n_pairs 40000 \
    --epochs 50 \
    --lr 1e-3 \
    --batch_size 256 \
    --output results_mlp_ablation.jsonl
```

#### Command-Line Arguments

**Task Complexity:**
- `--Ks`: Branching factors, comma-separated (default: `1,5,8`)
- `--n_pairs`: Total dataset size (default: `40000`)
- `--str_len`: Length of random strings (default: `8`)

**Training Hyperparameters:**
- `--epochs`: Number of training epochs (default: `50`)
- `--batch_size`: Batch size (default: `256`)
- `--lr`: Learning rate (default: `1e-3`)

**MLP Architecture:**
- `--d_emb`: Embedding dimension (default: `64`)
- `--d_hidden`: Hidden layer dimension (default: `512`)
- `--n_layers`: Number of hidden layers (default: `4`)

**Output:**
- `--output`: Output JSONL file path (default: `results_mlp_ablation.jsonl`)

## Methodology

### Data Construction

We construct a controlled synthetic environment using:

- **Alphabet**: $\Sigma = \{a-z, 0-9\}$ (36 characters)
- **String Length**: $L = 8$ (fixed-length strings)
- **Branching Factor**: $K \in \{1, 5, 8\}$ (tunable complexity)

For each branching factor $K$:

- **Forward ($A \rightarrow B$)**: Deterministic mapping with conditional entropy $H(B \mid A) = 0$
- **Inverse ($B \rightarrow A$)**: Probabilistic one-to-many mapping with entropy floor $H(A \mid B) = \ln K$

### Theoretical Loss Floors

The information-theoretic minimum loss for each task is:

- **Forward**: $\mathcal{L}_{\min} = 0$ (deterministic)
- **Inverse**: $\mathcal{L}_{\min} = \ln K$ (probabilistic, uniform over $K$ pre-images)

### Excess Loss Metric

We report **Excess Loss** as the primary metric:

$$\mathcal{L}_{\text{excess}} = \mathcal{L}_{\text{train}} - \mathcal{L}_{\min}$$

This metric isolates architectural and optimization inefficiencies from the inherent thermodynamic difficulty of the task.

### Prompt Format

Both directions use symmetric prompting:

- **Forward**: `"x: {A} y: "` → `"x: {A} y: {B}"`
- **Inverse**: `"x: {B} y: "` → `"x: {B} y: {A}"`

Loss is computed only over the target span (after `y:`), with prompt tokens masked.

### Training Regimes

1. **Scratch**: GPT-2 initialized from configuration (random weights)
2. **Finetune**: GPT-2 loaded from pre-trained weights, all parameters updated
3. **Finetune_Reg**: Pre-trained GPT-2 with elevated dropout and weight decay
4. **LoRA**: Pre-trained GPT-2 with low-rank adaptation on attention projections

### Reproducibility

Determinism is enforced via:
- Fixed seeds per $(K, \text{mode})$ configuration
- Deterministic CuDNN settings
- Identical synthetic pair lists reused across all regimes

## Output Format

Both scripts output JSONL (JSON Lines) files, where each line contains a complete run result:

```json
{
  "world": "A",
  "exp_id": "abc12345",
  "run_id": 1,
  "run_type": "scratch",
  "K": 5,
  "direction": "A->B",
  "seed": 0,
  "model_id": "gpt2",
  "n_pairs": 40000,
  "final_train_loss": 0.1234,
  "theoretical_min": 0.0,
  "excess_loss": 0.1234,
  "run_wall_time_sec": 45.6,
  "train_curve": [0.5, 0.3, 0.2, ...],
  ...
}
```

Key fields:
- `excess_loss`: Primary metric (train loss - theoretical minimum)
- `train_curve`: Full training loss curve across epochs
- `run_wall_time_sec`: Wall-clock training time
- `total_params`, `trainable_params`: Parameter counts

## Key Results Summary

### Directional Optimization Gap

- **Transformers (scratch)**: $\approx 1.16$ nats excess loss gap at $K=5$
- **MLPs**: $\approx 0.22$ nats excess loss gap at $K=5$

The Transformer gap is substantially larger, indicating that **directional friction is intrinsic to causal Transformer training**—persisting even when linguistic priors, token frequencies, and corpus-level temporal asymmetries are removed.

### Pre-training Effects

Pre-trained initializations shift optimization behavior but do not eliminate the directional gap, suggesting that the asymmetry is not simply a consequence of pre-training data statistics.

### LoRA Capacity Limits

Low-rank adaptation (LoRA) encounters a sharp capacity wall on high-entropy inverse mappings ($B \to A$ at $K=8$), plateauing early and showing minimal progress toward the entropy floor. This indicates limited expressivity for arbitrary high-entropy mappings under rank constraints.

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{sahasrabudhe2025controlled,
  title={A Controlled Benchmark for Measuring Directional Training Asymmetry in Transformers},
  author={Sahasrabudhe, Mihir},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This code is released for research purposes. Please refer to the paper for full details on methodology and results.

## Contact

For questions or issues, please contact: mihirss2@illinois.edu

## Acknowledgments

This work introduces a minimal, semantics-free tool for isolating and measuring directional training behaviors in sequence models. The benchmark provides a controlled instrument for dissecting directional biases in modern sequence models and motivates deeper mechanistic study of why inversion remains fundamentally harder for Transformers, even when theoretical analyses suggest reversal invariance.

**Code**: [https://github.com/mihirs-0/synass](https://github.com/mihirs-0/synass)

