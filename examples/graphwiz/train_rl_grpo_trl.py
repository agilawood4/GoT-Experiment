import argparse
import json
import os
from typing import Any, Dict, List

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_text(s: str) -> str:
    return " ".join(str(s or "").strip().lower().split())


def _token_f1(pred: str, target: str) -> float:
    p = _normalize_text(pred).split()
    t = _normalize_text(target).split()
    if not p or not t:
        return 0.0
    from collections import Counter

    pc, tc = Counter(p), Counter(t)
    overlap = sum(min(pc[k], tc[k]) for k in pc.keys() & tc.keys())
    if overlap <= 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(t)
    return 2 * precision * recall / (precision + recall)


def build_grpo_dataset(path: str, max_samples: int = 0) -> Dataset:
    rows = _load_jsonl(path)
    if max_samples > 0:
        rows = rows[:max_samples]
    out = []
    for r in rows:
        prompt = str(r.get("prompt", "")).strip()
        reference = str(r.get("response", "")).strip()
        prior_reward = float(r.get("reward", 0.0))
        if not prompt or not reference:
            continue
        out.append(
            {
                "prompt": prompt,
                "reference": reference,
                "prior_reward": prior_reward,
            }
        )
    return Dataset.from_list(out)


def _extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        # Chat-style: [{"role":"assistant","content":"..."}]
        if isinstance(completion[-1], dict) and "content" in completion[-1]:
            return str(completion[-1].get("content", ""))
        return str(completion[-1])
    if isinstance(completion, dict):
        return str(completion.get("content", completion.get("text", "")))
    return str(completion)


def run_grpo(
    model_name_or_path: str,
    dataset_path: str,
    output_dir: str,
    max_prompt_length: int = 1024,
    max_completion_length: int = 256,
    learning_rate: float = 5e-6,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    max_train_samples: int = 0,
):
    try:
        from trl import GRPOConfig, GRPOTrainer
    except Exception as e:
        raise RuntimeError(
            "Your TRL version does not expose GRPOTrainer/GRPOConfig. "
            "Please install a recent TRL version or run train_rl_weighted_sft.py instead."
        ) from e

    os.makedirs(output_dir, exist_ok=True)
    train_dataset = build_grpo_dataset(dataset_path, max_samples=max_train_samples)
    if len(train_dataset) == 0:
        raise ValueError("GRPO dataset is empty.")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
    )

    def reward_fn(completions, reference, prior_reward, **kwargs):
        scores: List[float] = []
        for comp, ref, pr in zip(completions, reference, prior_reward):
            pred_text = _extract_completion_text(comp)
            sim = _token_f1(pred_text, str(ref))
            try:
                prf = float(pr)
            except Exception:
                prf = 0.0
            # blend task trajectory prior with online generation quality
            score = 0.7 * sim + 0.3 * max(min(prf, 2.0), -1.0)
            scores.append(float(score))
        return scores

    args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        logging_steps=10,
        save_steps=100,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=args,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optional TRL GRPO trainer for GraphWiz RL data.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True, help="graphwiz_rl.jsonl")
    parser.add_argument("--output_dir", type=str, default="./outputs/trl_grpo")
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_train_samples", type=int, default=0)
    args = parser.parse_args()

    run_grpo(
        model_name_or_path=args.model_name_or_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_train_samples=args.max_train_samples,
    )
