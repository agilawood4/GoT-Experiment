import argparse
import json
import os
from typing import Dict


def _count_jsonl(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _load_first_jsonl(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def verify(run_dir: str, sft_path: str = "", pref_path: str = "", rl_path: str = "") -> Dict:
    summary_path = os.path.join(run_dir, "summary.json")
    sample_csv = os.path.join(run_dir, "sample_stats.csv")
    trajectories = os.path.join(run_dir, "trajectories.jsonl")

    result = {
        "run_dir": run_dir,
        "has_summary_json": os.path.exists(summary_path),
        "has_sample_stats_csv": os.path.exists(sample_csv),
        "has_trajectories_jsonl": os.path.exists(trajectories),
        "trajectory_count": _count_jsonl(trajectories),
    }

    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        result["num_samples_run"] = summary.get("num_samples_run", 0)
        result["accuracy"] = summary.get("accuracy", 0.0)
        result["mean_reward"] = summary.get("mean_reward", None)

    if sft_path:
        result["sft_path"] = sft_path
        result["sft_count"] = _count_jsonl(sft_path)
        first = _load_first_jsonl(sft_path)
        result["sft_schema_ok"] = all(k in first for k in ["instruction", "input", "output"]) if first else False

    if pref_path:
        result["pref_path"] = pref_path
        result["pref_count"] = _count_jsonl(pref_path)
        first = _load_first_jsonl(pref_path)
        result["pref_schema_ok"] = all(k in first for k in ["instruction", "input", "chosen", "rejected"]) if first else False

    if rl_path:
        result["rl_path"] = rl_path
        result["rl_count"] = _count_jsonl(rl_path)
        first = _load_first_jsonl(rl_path)
        result["rl_schema_ok"] = all(k in first for k in ["prompt", "response", "reward"]) if first else False

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify end-to-end post-training data pipeline.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--sft_path", type=str, default="")
    parser.add_argument("--pref_path", type=str, default="")
    parser.add_argument("--rl_path", type=str, default="")
    args = parser.parse_args()

    report = verify(
        run_dir=args.run_dir,
        sft_path=args.sft_path,
        pref_path=args.pref_path,
        rl_path=args.rl_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
