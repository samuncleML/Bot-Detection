# ================================================================
# GraphMind Full Pipeline — Restart
# Phase 1: Data Preparation
# Phase 2: DeepSeek Annotation (FIM + GSI)
# Phase 3: SFT
# Phase 4: GRPO
# ================================================================

# ── Install ───────────────────────────────────────────────────────
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "trl==0.15.2", "peft==0.20.0",
    "bitsandbytes>=0.43.0", "openai>=1.0.0",
    "pandas", "numpy"
], check=True)

import os, re, json, gc, math, ast, random, time, functools, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ================================================================
#  PATHS
# ================================================================
BASE_DIR    = "/kaggle/working/Bot-Detection"
DATA_DIR    = f"{BASE_DIR}/data"
ANNOT_DIR   = f"{BASE_DIR}/annotations"
SFT_DIR     = f"{BASE_DIR}/sft_v4"
GRPO_DIR    = f"{BASE_DIR}/grpo_v4"

for d in [DATA_DIR, ANNOT_DIR, SFT_DIR, GRPO_DIR]:
    os.makedirs(d, exist_ok=True)

MODEL_ID        = "Qwen/Qwen2.5-1.5B-Instruct"
FEATURES_PCA_PT = f"{DATA_DIR}/features_pca256.pt"
LABELS_PT       = f"{DATA_DIR}/labels_bot.pt"
EDGE_INDEX_PT   = f"{DATA_DIR}/edge_index.pt"
EDGE_TYPE_PT    = f"{DATA_DIR}/edge_type.pt"
GSI_CSV         = f"{DATA_DIR}/df_gsi.csv"
FIM_CSV         = f"{DATA_DIR}/df_fim_raw.csv"

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-key-here")

# ================================================================
#  PHASE 0 — LOAD MGTAB & PRECOMPUTED FEATURES
# ================================================================
print("="*65)
print("PHASE 0: Loading MGTAB data + precomputed features")
print("="*65)

features_pca = torch.load(FEATURES_PCA_PT).float()  # [10199, 256]
labels     = torch.load(LABELS_PT)              # [10199]  0=human,1=bot
edge_index = torch.load(EDGE_INDEX_PT)          # [2, E]
edge_type  = torch.load(EDGE_TYPE_PT)           # [E]

if features_pca.ndim != 2 or features_pca.shape[1] != 256:
    raise ValueError(
        f"Expected precomputed features with shape [N, 256], got {tuple(features_pca.shape)}"
    )

num_nodes   = features_pca.shape[0]
print(f"  Nodes: {num_nodes} | Feature dim: {features_pca.shape[1]}")
print(f"  Human: {(labels==0).sum().item()} | Bot: {(labels==1).sum().item()}")
print(f"  Loaded precomputed features: {FEATURES_PCA_PT}")

node_embeddings = features_pca   # [10199, 256] — used for rewards

# ── In-degrees ────────────────────────────────────────────────────
in_degrees = torch.zeros(num_nodes)
for d in edge_index[1].tolist():
    in_degrees[d] += 1
normalized_in_degrees = in_degrees / (in_degrees.max() + 1e-8)

# ── Human subgraph ────────────────────────────────────────────────
human_mask = (labels == 0)
human_idx  = human_mask.nonzero(as_tuple=True)[0]
human_set  = set(human_idx.tolist())

adj_human: Dict[int, List[int]] = {n: [] for n in human_set}
for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
    if s in human_set and d in human_set:
        adj_human[s].append(d)

print(f"  PCA dim: {node_embeddings.shape[1]} | Human nodes: {len(human_set)}")

# ── Edge type map (MGTAB) ─────────────────────────────────────────
# 0=follower, 1=friend, 2=mention, 3=reply, 4=quoted, 5=URL, 6=hashtag
EDGE_TYPE_NAMES = {
    0: "followerUser A is followed by user B",
    2: "mentionUser A mentions user B in a tweet",
    3: "replyUser A replies to a tweet by user B",
    4: "quotedUser A quotes a tweet by user B",
}
# Build per-pair edge type lookup
pair_to_etype: Dict[Tuple[int,int], int] = {}
for s, d, t in zip(edge_index[0].tolist(),
                   edge_index[1].tolist(),
                   edge_type.tolist()):
    pair_to_etype[(s, d)] = int(t)

print(f"  Edge types: {EDGE_TYPE_NAMES}")


# ================================================================
#  PHASE 1 — ANNOTATION WITH DEEPSEEK
# ================================================================
print("\n" + "="*65)
print("PHASE 1: DeepSeek Annotation")
print("="*65)

from openai import OpenAI

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

def call_deepseek(system_prompt: str,
                  user_prompt: str,
                  max_tokens: int = 300,
                  retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system",  "content": system_prompt},
                    {"role": "user",    "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  DeepSeek error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return ""

# ── FIM Annotation ───────────────────────────────────────────────
FIM_SYSTEM = """You are an expert social network analyst modeling ego-network interaction dynamics.

Given the profiles of User U and User V and their observed interaction frequency, infer their relationship level according to ego-network principles:
- Level 1: Support clique (strongest ties, frequent contact)
- Level 2: Sympathy group (close ties, frequent but not weekly)
- Level 3: Affinity group (casual ties, occasional contact)
- Level 4: Active network (weak ties, at least yearly contact)

Then output ONLY the structured response below. Be concise — no extra explanation."""

FIM_USER_TEMPLATE = """<user_profile>: Feature_Vector: {profile_u}
<target_profile>: Feature_Vector: {profile_v}
<tweets>: [Target User Tweet ID: tweet_{node_a}_{tweet_id}]
<observed_stats>: interaction_frequency={freq}

Output format:
<think>
The target user belongs to my ego network with relationship level {level_placeholder}. Based on this relationship strength and the observed profile and interaction context, I infer appropriate interaction intensity and engagement patterns.
</think>

<action>
  <type> {action_type} </type>
  <tweet_id> tweet_{node_a}_{tweet_id} </tweet_id>
</action>"""

# Edge type → action type + relationship level heuristics
ETYPE_TO_ACTION = {
    0: "follow",
    2: "mention",
    3: "reply",
    4: "quoted",
}

def etype_to_level(etype: int, freq: int) -> int:
    """Map edge type + frequency to relationship level 1-4."""
    if etype in [0, 1]:   # follow/friend edges
        if freq >= 10: return 1
        if freq >= 5:  return 2
        if freq >= 2:  return 3
        return 4
    elif etype in [3, 4]:  # reply/quote — high engagement
        if freq >= 5:  return 1
        if freq >= 2:  return 2
        return 3
    else:
        if freq >= 3:  return 2
        return 3 if freq >= 1 else 4

def fmt_vec(node_id: int, n_dims: int = 8) -> str:
    """Return first n_dims of PCA vector as compact string."""
    v = node_embeddings[node_id].tolist()[:n_dims]
    return "[" + ", ".join(f"{x:.4f}" for x in v) + ", ...]"

def annotate_fim_sample(row: pd.Series, idx: int) -> Optional[Dict]:
    node_a   = int(row["node_A"])
    node_b   = int(row["node_B"])
    etype    = int(row.get("edge_type", 0))
    freq     = int(row.get("freq", 1))
    level    = etype_to_level(etype, freq)
    action   = ETYPE_TO_ACTION.get(etype, "like")
    tweet_id = abs(hash(f"{node_a}_{node_b}_{idx}")) % 1000000

    profile_u = fmt_vec(node_a)
    profile_v = fmt_vec(node_b)

    level_names = {
        1: "1 (Support clique)",
        2: "2 (Sympathy group)",
        3: "3 (Affinity group)",
        4: "4 (Active network)",
    }

    user_prompt = FIM_USER_TEMPLATE.format(
        profile_u=profile_u,
        profile_v=profile_v,
        node_a=node_a,
        tweet_id=tweet_id,
        freq=freq,
        level_placeholder=level_names[level],
        action_type=action,
    )

    output = call_deepseek(FIM_SYSTEM, user_prompt, max_tokens=200)

    if not output:
        return None

    # Build structured message
    messages = [
        {"role": "system",    "content": FIM_SYSTEM},
        {"role": "user",      "content": (
            f"<user_profile>: Feature_Vector: {profile_u}\n"
            f"<target_profile>: Feature_Vector: {profile_v}\n"
            f"<tweets>: [Target User Tweet ID: tweet_{node_a}_{tweet_id}]\n"
            f"<observed_stats>: interaction_frequency={freq}"
        )},
        {"role": "assistant", "content": output},
    ]

    return {
        "messages":  messages,
        "node_a":    node_a,
        "node_b":    node_b,
        "level":     level,
        "action":    action,
        "tweet_id":  f"tweet_{node_a}_{tweet_id}",
        "edge_type": etype,
        "module":    "FIM",
    }

# ── GSI Annotation ───────────────────────────────────────────────
GSI_SYSTEM = """You are a social graph analyst. Analyze the sequence of adjacent user pairs in a multi-hop follow chain.

For each directed follow step (User A -> User B), provide ONE concise sentence explaining the follow decision based on profile attributes and structural influence (in-degree).

Then format into the unified CoT path format. Be brief — one sentence per hop."""

def annotate_gsi_sample(row: pd.Series) -> Optional[Dict]:
    try:
        full_path = ast.literal_eval(str(row["full_path"]))
    except Exception:
        return None

    if len(full_path) < 3:
        return None

    hop_count  = len(full_path) - 1
    source     = full_path[0]
    target     = full_path[-1]
    mediators  = full_path[1:-1]

    # Build node descriptions
    node_descs = []
    for i, nid in enumerate(full_path):
        if nid < num_nodes:
            in_deg = int(in_degrees[nid].item())
            vec    = fmt_vec(nid, n_dims=4)
            node_descs.append(
                f"Node {i} (ID:{nid}): profile={vec}, in_degree={in_deg}"
            )
        else:
            node_descs.append(f"Node {i} (ID:{nid}): in_degree=unknown")

    nodes_text = "\n".join(node_descs)
    hops_text  = " -> ".join([f"Node_{i}(ID:{n})"
                               for i, n in enumerate(full_path)])

    user_prompt = (
        f"Multi-hop follow chain ({hop_count} hops):\n"
        f"Path: {hops_text}\n\n"
        f"Node profiles:\n{nodes_text}\n\n"
        f"Task: Generate one-sentence rationales for each hop, "
        f"then format as a CoT path.\n\n"
        f"Output format:\n"
        f"<User_{full_path[0]}> (in_degree: {int(in_degrees[source].item())})\n"
        f"-> [Rationale 0]: <one sentence>\n"
        f"-> <User_{full_path[1]}> (in_degree: {int(in_degrees[full_path[1]].item()) if full_path[1]<num_nodes else '?'})\n"
        f"[...continue for each hop...]\n"
        f"-> <User_{full_path[-1]}> (in_degree: {int(in_degrees[target].item()) if target<num_nodes else '?'})"
    )

    output = call_deepseek(GSI_SYSTEM, user_prompt, max_tokens=250)
    if not output:
        return None

    messages = [
        {"role": "system",    "content": GSI_SYSTEM},
        {"role": "user",      "content": user_prompt},
        {"role": "assistant", "content": output},
    ]

    return {
        "messages":   messages,
        "full_path":  full_path,
        "source":     source,
        "target":     target,
        "mediators":  mediators,
        "hop_count":  hop_count,
        "module":     "GSI",
    }

# ── Run annotation (with checkpoint) ─────────────────────────────
FIM_ANNOT = f"{ANNOT_DIR}/fim_annotations.jsonl"
GSI_ANNOT = f"{ANNOT_DIR}/gsi_annotations.jsonl"

def load_existing(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try: out.append(json.loads(line))
                except: pass
    return out

def run_fim_annotation(n_samples: int = 4000):
    existing = load_existing(FIM_ANNOT)
    done     = len(existing)
    print(f"  FIM: {done} already annotated, target={n_samples}")
    if done >= n_samples:
        print("  FIM annotation complete.")
        return

    df = pd.read_csv(FIM_CSV)
    print(f"  FIM CSV rows: {len(df)}")

    # Filter to human-human edges
    df = df[df["node_A"].isin(human_set) & df["node_B"].isin(human_set)]
    print(f"  Human-human pairs: {len(df)}")

    # Add edge type and freq if missing
    if "edge_type" not in df.columns:
        df["edge_type"] = df.apply(
            lambda r: pair_to_etype.get((int(r["node_A"]), int(r["node_B"])), 0),
            axis=1
        )
    if "freq" not in df.columns:
        df["freq"] = 1

    # Balance edge types — important to fix the all-like problem
    # Ensure we sample across all action types
    etype_groups = df.groupby("edge_type")
    per_group    = n_samples // max(len(etype_groups), 1)
    balanced     = []
    for etype, group in etype_groups:
        n = min(len(group), per_group)
        balanced.append(group.sample(n, random_state=42))
    df_balanced = pd.concat(balanced).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)
    print(f"  Balanced sample: {len(df_balanced)} rows")
    print(f"  Edge type distribution:\n{df_balanced['edge_type'].value_counts()}")

    with open(FIM_ANNOT, "a") as f:
        for i, (_, row) in enumerate(df_balanced.iterrows()):
            if i < done:
                continue
            if len(existing) + i - done >= n_samples:
                break
            result = annotate_fim_sample(row, i)
            if result:
                f.write(json.dumps(result) + "\n")
                f.flush()
            if (i + 1) % 50 == 0:
                total = done + i - done + 1
                print(f"    FIM annotated: {total}/{n_samples}")

def run_gsi_annotation(n_samples: int = 4000):
    existing = load_existing(GSI_ANNOT)
    done     = len(existing)
    print(f"  GSI: {done} already annotated, target={n_samples}")
    if done >= n_samples:
        print("  GSI annotation complete.")
        return

    df = pd.read_csv(GSI_CSV)
    print(f"  GSI CSV rows: {len(df)}")

    # Filter rows where all path nodes are in human set
    def all_human(path_str):
        try:
            path = ast.literal_eval(str(path_str))
            return all(n in human_set for n in path)
        except:
            return False

    df = df[df["full_path"].apply(all_human)].reset_index(drop=True)
    print(f"  Human-only paths: {len(df)}")

    # Balance hop counts
    df["hop_count_col"] = df["hop_count"]
    hop_groups  = df.groupby("hop_count_col")
    per_hop     = n_samples // max(len(hop_groups), 1)
    balanced    = []
    for hops, group in hop_groups:
        n = min(len(group), per_hop)
        balanced.append(group.sample(n, random_state=42))
    df_balanced = pd.concat(balanced).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)
    print(f"  Balanced GSI sample: {len(df_balanced)} rows")
    print(f"  Hop distribution:\n{df_balanced['hop_count_col'].value_counts()}")

    with open(GSI_ANNOT, "a") as f:
        for i, (_, row) in enumerate(df_balanced.iterrows()):
            if i < done:
                continue
            if len(existing) + i - done >= n_samples:
                break
            result = annotate_gsi_sample(row)
            if result:
                f.write(json.dumps(result) + "\n")
                f.flush()
            if (i + 1) % 50 == 0:
                total = done + i - done + 1
                print(f"    GSI annotated: {total}/{n_samples}")

print("\nRunning FIM annotation...")
run_fim_annotation(n_samples=4000)

print("\nRunning GSI annotation...")
run_gsi_annotation(n_samples=4000)

# ── Merge + train/val split ───────────────────────────────────────
TRAIN_JSONL = f"{DATA_DIR}/train_v4.jsonl"
VAL_JSONL   = f"{DATA_DIR}/val_v4.jsonl"

def build_splits(fim_path, gsi_path,
                 train_out, val_out,
                 val_ratio=0.1):
    fim = load_existing(fim_path)
    gsi = load_existing(gsi_path)
    print(f"  FIM: {len(fim)} | GSI: {len(gsi)}")

    # Keep only messages field for training
    fim_msgs = [{"messages": x["messages"], "module": "FIM"} for x in fim]
    gsi_msgs = [{"messages": x["messages"], "module": "GSI"} for x in gsi]

    all_data = fim_msgs + gsi_msgs
    random.shuffle(all_data)

    n_val   = int(len(all_data) * val_ratio)
    val     = all_data[:n_val]
    train   = all_data[n_val:]

    for path, data in [(train_out, train), (val_out, val)]:
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    print(f"  Train: {len(train)} | Val: {len(val)}")
    print(f"  Saved: {train_out}")
    print(f"  Saved: {val_out}")

print("\nBuilding train/val splits...")
build_splits(FIM_ANNOT, GSI_ANNOT, TRAIN_JSONL, VAL_JSONL)


# ================================================================
#  PHASE 2 — SFT
# ================================================================
print("\n" + "="*65)
print("PHASE 2: SFT Training")
print("="*65)

import inspect
import shutil
from datasets import load_dataset as hf_load_dataset
from peft import (LoraConfig, get_peft_model,
                  prepare_model_for_kbit_training, PeftModel)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                           BitsAndBytesConfig)
from trl import SFTConfig, SFTTrainer

# Patch torch.load
if not getattr(torch.load, "_graphmind_patched", False):
    def _patched_torch_load(*args, _orig=torch.load, **kwargs):
        kwargs["weights_only"] = False
        return _orig(*args, **kwargs)
    _patched_torch_load._graphmind_patched = True
    torch.load = _patched_torch_load

def run_sft():
    # Check for existing checkpoint
    existing_ckpts = sorted(
        [d for d in os.listdir(SFT_DIR)
         if d.startswith("checkpoint-") and
         os.path.isdir(os.path.join(SFT_DIR, d))],
        key=lambda x: int(x.split("-")[-1])
    ) if os.path.exists(SFT_DIR) else []

    resume_from = None
    if existing_ckpts:
        resume_from = os.path.join(SFT_DIR, existing_ckpts[-1])
        print(f"  Resuming SFT from: {resume_from}")
    else:
        print("  Starting SFT from scratch")
        os.makedirs(SFT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Dataset
    def build_prompt_completion(example):
        prompt = tokenizer.apply_chat_template(
            example["messages"][:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        completion = example["messages"][-1]["content"].strip()
        if not completion.endswith(tokenizer.eos_token):
            completion += tokenizer.eos_token
        return {"prompt": prompt, "completion": completion}

    train_ds = hf_load_dataset(
        "json", data_files=TRAIN_JSONL, split="train"
    ).shuffle(seed=42).map(
        build_prompt_completion,
        remove_columns=hf_load_dataset(
            "json", data_files=TRAIN_JSONL, split="train"
        ).column_names
    )
    val_ds = hf_load_dataset(
        "json", data_files=VAL_JSONL, split="train"
    ).map(
        build_prompt_completion,
        remove_columns=hf_load_dataset(
            "json", data_files=VAL_JSONL, split="train"
        ).column_names
    )
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Model
    gc.collect(); torch.cuda.empty_cache()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    n_gpus         = torch.cuda.device_count() or 1
    eff_batch      = 1 * 8 * n_gpus
    steps_per_epoch = math.ceil(len(train_ds) / eff_batch)

    _sft_params = inspect.signature(SFTConfig.__init__).parameters
    sft_kwargs  = dict(
        output_dir=SFT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=1,
        learning_rate=2e-4,
        num_train_epochs=3,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True,
        save_strategy="steps",
        save_steps=steps_per_epoch,
        save_total_limit=2,
        load_best_model_at_end=True,
        greater_is_better=False,
        logging_steps=10,
        logging_first_step=True,
        report_to="none",
        seed=42,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        packing=False,
        ddp_find_unused_parameters=False,
    )
    if "eval_strategy" in _sft_params:
        sft_kwargs["eval_strategy"] = "steps"
    else:
        sft_kwargs["evaluation_strategy"] = "steps"
    sft_kwargs["eval_steps"] = steps_per_epoch

    if "max_length" in _sft_params:
        sft_kwargs["max_length"] = 512        # concise outputs
    elif "max_seq_length" in _sft_params:
        sft_kwargs["max_seq_length"] = 512

    if "completion_only_loss" in _sft_params:
        sft_kwargs["completion_only_loss"] = True
    if "dataset_num_proc" in _sft_params:
        sft_kwargs["dataset_num_proc"] = 4
    if "save_only_model" in _sft_params:
        sft_kwargs["save_only_model"] = True

    try:
        sft_config = SFTConfig(**sft_kwargs)
    except ValueError as e:
        if "load_best_model_at_end" not in str(e):
            raise
        for k in ("load_best_model_at_end",
                  "metric_for_best_model", "greater_is_better"):
            sft_kwargs.pop(k, None)
        sft_config = SFTConfig(**sft_kwargs)

    _trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs  = dict(
        model=model, args=sft_config,
        train_dataset=train_ds, eval_dataset=val_ds,
    )
    if "processing_class" in _trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=resume_from)

    trainer.model.save_pretrained(SFT_DIR)
    tokenizer.save_pretrained(SFT_DIR)
    print(f"  ✅ SFT saved → {SFT_DIR}")

    # Summary
    hist   = trainer.state.log_history
    evals  = [e for e in hist if "eval_loss" in e]
    losses = [(e["step"], e["loss"]) for e in hist if "loss" in e
              and "eval_loss" not in e]
    if losses:
        steps, ls = zip(*losses)
        print(f"  Loss: {ls[0]:.4f} → {ls[-1]:.4f}")
    if evals:
        best = min(evals, key=lambda e: e["eval_loss"])
        print(f"  Best eval loss: {best['eval_loss']:.4f} "
              f"(epoch {best['epoch']:.2f})")

    del model, trainer
    gc.collect(); torch.cuda.empty_cache()

run_sft()


# ================================================================
#  PHASE 3 — GRPO
# ================================================================
print("\n" + "="*65)
print("PHASE 3: GRPO Training")
print("="*65)

# ── GRPO patches ──────────────────────────────────────────────────
import trl.trainer.grpo_trainer as _gt
from peft import PeftMixedModel
import transformers.trainer as trainer_module

def patched_is_peft_model(model):
    return isinstance(model, (PeftModel, PeftMixedModel))
trainer_module._is_peft_model = patched_is_peft_model

_pyc_dir = os.path.join(os.path.dirname(_gt.__file__), "__pycache__")
if os.path.isdir(_pyc_dir):
    for _f in os.listdir(_pyc_dir):
        if "grpo_trainer" in _f:
            os.remove(os.path.join(_pyc_dir, _f))
_src    = open(_gt.__file__).read()
_target = "if self.num_generations not in possible_values:"
if _target in _src:
    _src = _src.replace(_target, "if False:  # patched")
    with open(_gt.__file__, "w") as _fh:
        _fh.write(_src)
for _key in list(sys.modules):
    if "grpo" in _key or "trl" in _key:
        del sys.modules[_key]
from trl import GRPOConfig, GRPOTrainer
import trl.trainer.grpo_trainer as _gt

_orig_sampler = _gt.GRPOTrainer._get_train_sampler
def _patched_sampler(self, dataset=None):
    return _orig_sampler(self)
_gt.GRPOTrainer._get_train_sampler = _patched_sampler

def patched_get_per_token_logps(self, model, input_ids,
                                 attention_mask, logits_to_keep):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits  = outputs.logits[:, :-1, :][:, -logits_to_keep:, :]
    ids     = input_ids[:, -logits_to_keep:]
    lp      = F.log_softmax(logits, dim=-1)
    return lp.gather(dim=-1, index=ids.unsqueeze(-1)).squeeze(-1)
_gt.GRPOTrainer._get_per_token_logps = patched_get_per_token_logps

def patched_wrap_model(self, model, training=True, dataloader=None):
    return model
trainer_module.Trainer._wrap_model   = patched_wrap_model
trainer_module._is_peft_model        = patched_is_peft_model

_orig_prepare = _gt.GRPOTrainer._prepare_inputs
def patched_prepare_inputs(self, inputs):
    try:
        return _orig_prepare(self, inputs)
    except (RuntimeError, TypeError) as e:
        if "invalid for input of size" not in str(e) and \
           "list indices" not in str(e):
            raise
        orig_view = torch.Tensor.view
        def safe_view(t, *args):
            if (len(args) == 2 and args[0] == -1 and
                    t.numel() > 0 and args[1] > 1 and
                    t.numel() % args[1] != 0):
                n = args[1]
                return t.repeat(math.ceil(n / t.numel()))[:n].unsqueeze(0)
            return orig_view(t, *args)
        torch.Tensor.view = safe_view
        try:
            return _orig_prepare(self, inputs)
        finally:
            torch.Tensor.view = orig_view
_gt.GRPOTrainer._prepare_inputs = patched_prepare_inputs

def patch_generation_cache(model):
    orig = model.generate
    @functools.wraps(orig)
    def fast_gen(*args, **kwargs):
        model.config.use_cache = True
        try:    return orig(*args, **kwargs)
        finally: model.config.use_cache = False
    model.generate = fast_gen
    return model

print("GRPO patches applied")

# ── Tensor reward functions ───────────────────────────────────────
def r_len(path: List[int], max_hops: int = 6) -> float:
    n = len(path)
    return min((n - 1) / float(max_hops), 1.0) if n > 1 else 0.0

def r_homo(path: List[int]) -> float:
    if not path: return 0.0
    anchor = node_embeddings[path[0]].unsqueeze(0)
    embs   = node_embeddings[path]
    return F.cosine_similarity(anchor, embs, dim=-1).mean().item()

def r_inf(path: List[int]) -> float:
    if len(path) <= 2: return 0.0
    return normalized_in_degrees[path[1:-1]].mean().item()

def extract_path(text: str) -> List[int]:
    seen, path = set(), []
    for n in re.findall(r'\b(\d{3,5})\b', text):
        nid = int(n)
        if nid not in seen and nid < num_nodes:
            seen.add(nid); path.append(nid)
    return path

def gsi_reward(completions, prompts=None, **kwargs) -> List[float]:
    rewards = []
    for i, c in enumerate(completions):
        path   = extract_path(c)
        prompt = prompts[i] if prompts and i < len(prompts) else ""
        if len(path) < 2:
            rewards.append(0.0)
            continue
        rl = r_len(path)
        rh = r_homo(path)
        ri = r_inf(path)
        # Hop bonus
        hm = re.search(r'(\d+)-hop', prompt)
        bonus = 0.3 if hm and (len(path)-1) >= int(hm.group(1)) else 0.0
        rewards.append(rl + rh + ri + bonus)
    return rewards

def fim_reward(completions, prompts=None, **kwargs) -> List[float]:
    """R_FIM = R1 (level inference) + R2 (action consistency)."""
    P_SAMPLE = {
        1: {"like":0.1,"retweet":0.4,"reply":0.3,"mention":0.2},
        2: {"like":0.2,"retweet":0.4,"reply":0.3,"mention":0.1},
        3: {"like":0.4,"retweet":0.3,"reply":0.2,"mention":0.1},
        4: {"like":0.6,"retweet":0.2,"reply":0.1,"mention":0.1},
    }
    rewards = []
    for i, c in enumerate(completions):
        prompt = prompts[i] if prompts and i < len(prompts) else ""
        # R1
        pm = re.search(r'level\s*(\d)', c, re.I)
        gm = re.search(r'level\s*(\d)', prompt, re.I)
        r1 = math.exp(-abs(int(pm.group(1)) - int(gm.group(1)))) \
             if pm and gm else (0.3 if pm else 0.0)
        # R2 — action distribution consistency
        level = int(pm.group(1)) if pm else None
        counts = {
            "like":    len(re.findall(r'\blike\b',    c, re.I)),
            "retweet": len(re.findall(r'\bretweet\b', c, re.I)),
            "reply":   len(re.findall(r'\breply\b',   c, re.I)),
            "mention": len(re.findall(r'\bmention\b', c, re.I)),
        }
        total = sum(counts.values())
        r2 = 0.0
        if level and level in P_SAMPLE and total > 0:
            pt = {k: v/total for k, v in counts.items()}
            ps = P_SAMPLE[level]
            kl = sum(pt.get(a,1e-8)*math.log((pt.get(a,1e-8)+1e-8)/
                     (ps.get(a,1e-8)+1e-8)) for a in ps)
            r2 = -kl
        rewards.append(r1 + r2)
    return rewards

def unified_reward(completions, prompts=None, **kwargs) -> List[float]:
    rewards = []
    for i, c in enumerate(completions):
        prompt  = prompts[i] if prompts and i < len(prompts) else ""
        is_fim  = bool(re.search(
            r'<user_profile>|observed_stats|interaction_frequency',
            prompt, re.I))
        if is_fim:
            r = fim_reward([c], prompts=[prompt])[0]
        else:
            r = gsi_reward([c], prompts=[prompt])[0]
        rewards.append(r)
    return rewards

def repetition_reward(completions, **kwargs) -> List[float]:
    out = []
    for c in completions:
        words = c.lower().split()
        if len(words) < 10: out.append(0.0); continue
        ng    = [tuple(words[i:i+5]) for i in range(len(words)-4)]
        ratio = len(set(ng))/len(ng) if ng else 1.0
        out.append(-3.0 if ratio<0.3 else -2.0 if ratio<0.5
                   else -1.0 if ratio<0.7 else 0.0)
    return out

def format_reward(completions, **kwargs) -> List[float]:
    out = []
    for c in completions:
        t = c.strip()
        # FIM: should end with </action>
        if t.endswith("</action>"):          out.append(2.0)
        # GSI: should end with </think> or user tag
        elif t.endswith("</think>"):         out.append(1.5)
        elif "<action>" in t and "</action>" in t: out.append(1.0)
        elif "</think>" in t:                out.append(0.3)
        else:                                out.append(-1.0)
    return out

# ── GRPO dataset — both FIM and GSI ───────────────────────────────
def build_grpo_dataset(n_fim: int = 2000,
                       n_gsi: int = 2000) -> "Dataset":
    from datasets import Dataset as HFDataset

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tok.pad_token    = tok.eos_token
    tok.padding_side = "left"

    prompts = []

    # FIM prompts from annotation
    fim_data = load_existing(FIM_ANNOT)
    random.shuffle(fim_data)
    for ex in fim_data[:n_fim]:
        msgs = ex["messages"][:-1]   # exclude assistant turn
        prompts.append({
            "prompt": tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            ),
            "module": "FIM",
        })

    # GSI prompts from human subgraph walks
    human_list = list(human_set)
    gsi_added  = 0
    while gsi_added < n_gsi:
        anchor = random.choice(human_list)
        if not adj_human.get(anchor): continue
        path   = [anchor]
        n_hops = random.randint(3, 6)
        for _ in range(n_hops):
            cands = adj_human.get(path[-1], [])
            if not cands: break
            w   = [normalized_in_degrees[c].item()+1e-3 for c in cands]
            tw  = sum(w)
            nxt = random.choices(cands, weights=[x/tw for x in w], k=1)[0]
            if nxt not in path: path.append(nxt)
        if len(path) < 3: continue

        src, tgt = path[0], path[-1]
        meds     = path[1:-1]
        content  = (
            f"Analyze this {len(path)-1}-hop social network path.\n"
            f"Full path: {path}\n"
            f"Mediator nodes: {meds}\n"
            f"Source: {src} → Target: {tgt}\n\n"
            f"Reason through each hop, assign connection strengths (0.0-1.0), "
            f"compute a total path score, and give a final connection assessment."
            f"\nRespond in English only."
        )
        msgs = [
            {"role": "system", "content": (
                "You are a global structural interaction agent trained to infer "
                "human-like social pathways and calculate overall structural "
                "connectivity in a social graph. Always respond in English."
            )},
            {"role": "user", "content": content},
        ]
        prompts.append({
            "prompt": tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            ),
            "module": "GSI",
        })
        gsi_added += 1

    random.shuffle(prompts)
    print(f"  GRPO dataset: {len(prompts)} prompts "
          f"({sum(1 for p in prompts if p['module']=='FIM')} FIM, "
          f"{sum(1 for p in prompts if p['module']=='GSI')} GSI)")
    return HFDataset.from_list(prompts)

def run_grpo():
    # Check for existing GRPO checkpoint
    existing_ckpts = sorted(
        [d for d in os.listdir(GRPO_DIR)
         if d.startswith("checkpoint-") and
         os.path.isdir(os.path.join(GRPO_DIR, d))],
        key=lambda x: int(x.split("-")[-1])
    ) if os.path.exists(GRPO_DIR) else []

    resume_from = None
    if existing_ckpts:
        resume_from = os.path.join(GRPO_DIR, existing_ckpts[-1])
        print(f"  Resuming GRPO from: {resume_from}")
    else:
        print("  Starting GRPO from scratch")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    gc.collect(); torch.cuda.empty_cache()

    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"  GPU {i}: {free/1e9:.1f}/{total/1e9:.1f} GB free")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
    )
    base.config.use_cache = False

    # Load from best SFT checkpoint
    sft_ckpts = sorted(
        [d for d in os.listdir(SFT_DIR)
         if d.startswith("checkpoint-") and
         os.path.isdir(os.path.join(SFT_DIR, d))],
        key=lambda x: int(x.split("-")[-1])
    ) if os.path.exists(SFT_DIR) else []
    sft_start = (os.path.join(SFT_DIR, sft_ckpts[-1])
                 if sft_ckpts else SFT_DIR)
    print(f"  SFT start: {sft_start}")

    model = PeftModel.from_pretrained(base, sft_start, is_trainable=True)
    model.enable_input_require_grads()
    model = patch_generation_cache(model)
    model.print_trainable_parameters()

    grpo_ds = build_grpo_dataset(n_fim=2000, n_gsi=2000)

    grpo_cfg = GRPOConfig(
        output_dir=GRPO_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        max_steps=500,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        num_generations=2,
        max_prompt_length=300,
        max_completion_length=200,   # concise — FIM+GSI fit in 200 tokens
        temperature=0.7,
        beta=0.01,
        gradient_checkpointing=False,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        report_to="none",
        seed=42,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        train_dataset=grpo_ds,
        reward_funcs=[
            unified_reward,      # FIM: R1+R2 | GSI: R_len+R_homo+R_inf
            repetition_reward,   # anti-loop
            format_reward,       # clean ending signal
        ],
        processing_class=tokenizer,
    )

    print(f"  GRPO: {torch.cuda.device_count()} GPU(s) | "
          f"batch={grpo_cfg.per_device_train_batch_size} | "
          f"gen={grpo_cfg.num_generations} | "
          f"max_len={grpo_cfg.max_completion_length}")

    trainer.train(resume_from_checkpoint=resume_from)

    trainer.model.save_pretrained(GRPO_DIR)
    tokenizer.save_pretrained(GRPO_DIR)
    print(f"  ✅ GRPO saved → {GRPO_DIR}")

    # Summary
    log    = trainer.state.log_history
    losses = [(l["step"], l["loss"]) for l in log if "loss" in l]
    rews   = [(l["step"], l.get("reward", "N/A"))
               for l in log if "loss" in l]
    if losses:
        steps, ls = zip(*losses)
        rvals = [r for _, r in rews if isinstance(r, float)]
        print(f"  Loss  : {ls[0]:.6f} → {ls[-1]:.6f}")
        if rvals:
            print(f"  Reward: {rvals[0]:.4f} → {rvals[-1]:.4f} "
                  f"(best: {max(rvals):.4f})")

    del model, trainer, base
    gc.collect(); torch.cuda.empty_cache()

run_grpo()

print("\n" + "="*65)
print("✅ Pipeline complete")
print(f"  SFT  → {SFT_DIR}")
print(f"  GRPO → {GRPO_DIR}")
print("="*65)