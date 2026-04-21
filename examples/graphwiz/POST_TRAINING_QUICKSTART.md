## GraphWiz Post-Training Quickstart

### 1) Run evaluation with trajectory export

```bash
python examples/graphwiz/graphwiz_eval_generic.py --source test --subset connectivity --lm_name qwen_local_4b --max_samples 20 --export_parquet 1
```

This creates:
- `sample_stats.csv`
- `trajectories.jsonl`
- optional `trajectories.parquet`

### 2) Build SFT / DPO / RL datasets

```bash
python examples/graphwiz/export_sft_data.py --input examples/graphwiz/results/<run_dir> --output examples/graphwiz/data/graphwiz_sft.jsonl --min_reward 0.4
python examples/graphwiz/export_pref_data.py --inputs examples/graphwiz/results/<run_dir1>,examples/graphwiz/results/<run_dir2> --output examples/graphwiz/data/graphwiz_pref.jsonl
python examples/graphwiz/export_rl_data.py --input examples/graphwiz/results/<run_dir> --output examples/graphwiz/data/graphwiz_rl.jsonl
```

### 3) Download local Qwen safetensors

```bash
python examples/graphwiz/download_qwen.py --repo_id Qwen/Qwen2.5-3B-Instruct --local_dir ./models/Qwen2.5-3B-Instruct
```

### 4) Minimal TRL SFT run (example)

```bash
python examples/graphwiz/train_sft_trl.py --model_name_or_path ./models/Qwen2.5-3B-Instruct --dataset_path examples/graphwiz/data/graphwiz_sft.jsonl --output_dir ./outputs/trl_sft_qwen25_3b --max_train_samples 32
```

### 5) Minimal TRL DPO run (example)

```bash
python examples/graphwiz/train_dpo_trl.py --model_name_or_path ./models/Qwen2.5-3B-Instruct --dataset_path examples/graphwiz/data/graphwiz_pref.jsonl --output_dir ./outputs/trl_dpo_qwen25_3b --max_train_samples 32
```

### 6) Verify small end-to-end pipeline

```bash
python examples/graphwiz/verify_post_training_pipeline.py --run_dir examples/graphwiz/results/<run_dir> --sft_path examples/graphwiz/data/graphwiz_sft.jsonl --pref_path examples/graphwiz/data/graphwiz_pref.jsonl --rl_path examples/graphwiz/data/graphwiz_rl.jsonl
```

### 7) LLaMA-Factory DPO dataset mapping

Use `examples/graphwiz/train_configs/llamafactory_dpo_dataset_info.json` and point it to the generated preference jsonl.
