import argparse
import json
import os
import sqlite3
from typing import Any, Dict, List


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _collect_metrics(run_dir: str) -> Dict[str, Any]:
    summary_path = os.path.join(run_dir, "summary.json")
    trajectories_path = os.path.join(run_dir, "trajectories.jsonl")
    summary = _load_json(summary_path)
    rows = _load_jsonl(trajectories_path) if os.path.exists(trajectories_path) else []

    final_validator_pass_rate = 0.0
    if rows:
        pass_cnt = 0
        for t in rows:
            steps = t.get("steps", []) or []
            final_step = None
            for s in reversed(steps):
                if s.get("part") == "final":
                    final_step = s
                    break
            if final_step is not None and bool(final_step.get("final_validator")):
                pass_cnt += 1
        final_validator_pass_rate = pass_cnt / max(1, len(rows))

    return {
        "run_dir": run_dir,
        "accuracy": _safe_float(summary.get("accuracy", 0.0)),
        "mean_reward": _safe_float(summary.get("mean_reward", 0.0)),
        "total_tokens": int(summary.get("total_tokens", 0) or 0),
        "final_validator_pass_rate": float(final_validator_pass_rate),
        "online_pairs_total": int(summary.get("online_pairs_total", 0) or 0),
        "online_hot_swap_count": int(summary.get("online_hot_swap_count", 0) or 0),
        "online_pref_store": summary.get("online_pref_store", {}),
        "online_model_registry": summary.get("online_model_registry", {}),
    }


def _read_pref_store_stats(db_path: str) -> Dict[str, int]:
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        total = conn.execute("SELECT COUNT(1) FROM preferences").fetchone()[0]
        pending = conn.execute("SELECT COUNT(1) FROM preferences WHERE consumed = 0").fetchone()[0]
        consumed = conn.execute("SELECT COUNT(1) FROM preferences WHERE consumed = 1").fetchone()[0]
        return {"total": int(total), "pending": int(pending), "consumed": int(consumed)}
    finally:
        conn.close()


def verify(online_run_dir: str, baseline_run_dir: str = "") -> Dict[str, Any]:
    if not os.path.isdir(online_run_dir):
        raise FileNotFoundError(f"Online run directory not found: {online_run_dir}")

    online_metrics = _collect_metrics(online_run_dir)
    out: Dict[str, Any] = {
        "online_run": online_metrics,
        "checks": {
            "has_trajectories": os.path.exists(os.path.join(online_run_dir, "trajectories.jsonl")),
            "has_summary": os.path.exists(os.path.join(online_run_dir, "summary.json")),
            "online_pairs_generated": online_metrics["online_pairs_total"] > 0,
            "online_hot_swapped": online_metrics["online_hot_swap_count"] > 0,
        },
    }

    pref_store = online_metrics.get("online_pref_store", {}) or {}
    db_path = str(pref_store.get("db_path", "") or "")
    if db_path and os.path.exists(db_path):
        out["pref_store_stats"] = _read_pref_store_stats(db_path)
    else:
        out["pref_store_stats"] = {}

    if baseline_run_dir:
        if not os.path.isdir(baseline_run_dir):
            raise FileNotFoundError(f"Baseline run directory not found: {baseline_run_dir}")
        baseline_metrics = _collect_metrics(baseline_run_dir)
        out["baseline_run"] = baseline_metrics
        out["diff"] = {
            "accuracy_delta": online_metrics["accuracy"] - baseline_metrics["accuracy"],
            "mean_reward_delta": online_metrics["mean_reward"] - baseline_metrics["mean_reward"],
            "final_validator_pass_rate_delta": (
                online_metrics["final_validator_pass_rate"]
                - baseline_metrics["final_validator_pass_rate"]
            ),
            "total_tokens_delta": online_metrics["total_tokens"] - baseline_metrics["total_tokens"],
        }

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify online multi-node DPO pipeline outputs.")
    parser.add_argument("--online_run_dir", type=str, required=True)
    parser.add_argument("--baseline_run_dir", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    report = verify(
        online_run_dir=args.online_run_dir,
        baseline_run_dir=args.baseline_run_dir,
    )
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
