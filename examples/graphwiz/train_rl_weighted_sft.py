import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


def _load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_reward(x: float, min_v: float, max_v: float) -> float:
    if max_v <= min_v + 1e-12:
        return 1.0
    # map to [0.1, 1.0], avoid zeroing gradient completely
    return 0.1 + 0.9 * ((x - min_v) / (max_v - min_v))


def _format_prompt_response(prompt: str, response: str) -> str:
    return f"{prompt}{response}"


def build_rl_dataset(path: str, max_samples: int = 0) -> Dataset:
    rows = _load_jsonl(path)
    if max_samples > 0:
        rows = rows[:max_samples]
    rewards = [float(r.get("reward", 0.0)) for r in rows] if rows else [0.0]
    min_r, max_r = min(rewards), max(rewards)

    out = []
    for row in rows:
        prompt = str(row.get("prompt", "")).strip()
        response = str(row.get("response", "")).strip()
        if not prompt or not response:
            continue
        reward = float(row.get("reward", 0.0))
        weight = _normalize_reward(reward, min_r, max_r)
        out.append(
            {
                "prompt": prompt,
                "response": response,
                "text": _format_prompt_response(prompt, response),
                "reward": reward,
                "weight": weight,
            }
        )
    return Dataset.from_list(out)


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    weights: torch.Tensor


def collate_fn(batch_rows: List[Dict], tokenizer, max_seq_length: int) -> Batch:
    texts = [r["text"] for r in batch_rows]
    weights = torch.tensor([float(r["weight"]) for r in batch_rows], dtype=torch.float32)
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    )
    labels = encoded["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return Batch(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        labels=labels,
        weights=weights,
    )


def run_weighted_rl(
    model_name_or_path: str,
    dataset_path: str,
    output_dir: str,
    max_seq_length: int = 2048,
    learning_rate: float = 2e-5,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_train_samples: int = 0,
    warmup_ratio: float = 0.03,
    bf16: bool = True,
    log_steps: int = 10,
):
    os.makedirs(output_dir, exist_ok=True)

    dataset = build_rl_dataset(dataset_path, max_samples=max_train_samples)
    if len(dataset) == 0:
        raise ValueError("RL dataset is empty. Please check export_rl_data output.")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_bf16 = bf16 and torch.cuda.is_available()
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.train()

    device = next(model.parameters()).device
    dl = DataLoader(
        dataset,
        batch_size=per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda rows: collate_fn(rows, tokenizer, max_seq_length),
    )

    total_steps = max(1, (len(dl) * num_train_epochs) // max(1, gradient_accumulation_steps))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(num_train_epochs):
        for step, batch in enumerate(dl):
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            labels = batch.labels.to(device)
            weights = batch.weights.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            # outputs.loss is mean over tokens and batch.
            # We reweight by sample rewards approximately:
            # multiply scalar loss by mean(batch weights)
            weighted_loss = outputs.loss * weights.mean()
            loss = weighted_loss / max(1, gradient_accumulation_steps)
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % max(1, log_steps) == 0:
                    print(
                        f"[epoch={epoch+1}] step={global_step}/{total_steps} "
                        f"loss={weighted_loss.item():.6f} mean_weight={weights.mean().item():.4f}"
                    )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_path": dataset_path,
                "num_samples": len(dataset),
                "max_seq_length": max_seq_length,
                "learning_rate": learning_rate,
                "num_train_epochs": num_train_epochs,
                "per_device_train_batch_size": per_device_train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "method": "reward_weighted_policy_optimization",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal RL line via reward-weighted policy optimization.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True, help="graphwiz_rl.jsonl")
    parser.add_argument("--output_dir", type=str, default="./outputs/trl_rl_weighted")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--bf16", type=int, default=1)
    parser.add_argument("--log_steps", type=int, default=10)
    args = parser.parse_args()

    run_weighted_rl(
        model_name_or_path=args.model_name_or_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_train_samples=args.max_train_samples,
        warmup_ratio=args.warmup_ratio,
        bf16=bool(args.bf16),
        log_steps=args.log_steps,
    )
