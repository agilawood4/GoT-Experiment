import argparse
import json
import os
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from trl import DPOTrainer


def _load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_prompt(example: Dict) -> str:
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    if input_text:
        return f"Instruction:\n{instruction}\n\nInput:\n{input_text}\n\nResponse:\n"
    return f"Instruction:\n{instruction}\n\nResponse:\n"


def build_pref_dataset_from_rows(rows: List[Dict], max_samples: int = 0) -> Dataset:
    if max_samples > 0:
        rows = rows[:max_samples]
    out = []
    for r in rows:
        prompt = _build_prompt(r)
        chosen = str(r.get("chosen", "")).strip()
        rejected = str(r.get("rejected", "")).strip()
        if not prompt or not chosen or not rejected:
            continue
        out.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return Dataset.from_list(out)


def build_pref_dataset(path: str, max_samples: int = 0) -> Dataset:
    rows = _load_jsonl(path)
    return build_pref_dataset_from_rows(rows, max_samples=max_samples)


def run_dpo_once(
    model_name_or_path: str,
    output_dir: str,
    dataset_path: Optional[str] = None,
    pref_records: Optional[List[Dict]] = None,
    ref_model_name_or_path: Optional[str] = None,
    beta: float = 0.1,
    learning_rate: float = 5e-6,
    num_train_epochs: float = 1.0,
    max_steps: int = 0,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    logging_steps: int = 10,
    save_steps: int = 100,
    max_train_samples: int = 0,
    bf16: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    if pref_records is not None:
        train_dataset = build_pref_dataset_from_rows(pref_records, max_samples=max_train_samples)
    else:
        if not dataset_path:
            raise ValueError("Either dataset_path or pref_records must be provided.")
        train_dataset = build_pref_dataset(dataset_path, max_samples=max_train_samples)
    if len(train_dataset) == 0:
        raise ValueError("DPO dataset is empty.")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 and torch.cuda.is_available() else torch.float16,
        device_map="auto",
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_name_or_path or model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 and torch.cuda.is_available() else torch.float16,
        device_map="auto",
    )

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        max_steps=int(max_steps) if int(max_steps) > 0 else -1,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=2,
        bf16=bf16 and torch.cuda.is_available(),
        fp16=(not bf16) and torch.cuda.is_available(),
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        beta=beta,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "dpo_train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name_or_path": model_name_or_path,
                "ref_model_name_or_path": ref_model_name_or_path or model_name_or_path,
                "num_examples": len(train_dataset),
                "beta": beta,
                "learning_rate": learning_rate,
                "num_train_epochs": num_train_epochs,
                "max_steps": int(max_steps),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return output_dir


def run_dpo(
    model_name_or_path: str,
    dataset_path: str,
    output_dir: str,
    beta: float = 0.1,
    learning_rate: float = 5e-6,
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    logging_steps: int = 10,
    save_steps: int = 100,
    max_train_samples: int = 0,
    bf16: bool = True,
):
    return run_dpo_once(
        model_name_or_path=model_name_or_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        beta=beta,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        max_train_samples=max_train_samples,
        bf16=bf16,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal TRL DPO training for GraphWiz data.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True, help="graphwiz_pref.jsonl")
    parser.add_argument("--output_dir", type=str, default="./outputs/trl_dpo")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--bf16", type=int, default=1)
    args = parser.parse_args()

    run_dpo(
        model_name_or_path=args.model_name_or_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_train_samples=args.max_train_samples,
        bf16=bool(args.bf16),
    )
