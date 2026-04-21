import json
import os
from typing import Any, Callable, Dict, List, Optional


def _safe_call(fn: Optional[Callable], state: Dict[str, Any], default: Any = None) -> Any:
    if not callable(fn):
        return default
    try:
        return fn(state)
    except Exception:
        return default


def build_trajectory_record(
    sample: Dict[str, Any],
    routed_task: str,
    output_json_path: str,
    query_history: List[Dict[str, Any]],
    final_answer: str,
    is_correct: bool,
    error_message: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost: float,
    search_score_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    final_validator_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ground_truth_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Dict[str, Any]:
    with open(output_json_path, "r", encoding="utf-8") as f:
        graph_dump = json.load(f)

    operation_entries = graph_dump[:-1] if graph_dump and isinstance(graph_dump[-1], dict) and "cost" in graph_dump[-1] else graph_dump
    step_records: List[Dict[str, Any]] = []

    q_idx = 0
    for op_idx, op_entry in enumerate(operation_entries):
        operation_name = op_entry.get("operation", "unknown")
        thoughts = op_entry.get("thoughts", []) or []
        scores = op_entry.get("scores", []) or []
        validity = op_entry.get("validity", []) or []
        solved = op_entry.get("problem_solved", []) or []

        for t_idx, thought_state in enumerate(thoughts):
            state = dict(thought_state or {})
            if "task" not in state:
                state["task"] = routed_task
            if "gold" not in state:
                state["gold"] = sample.get("answer", "")

            query_event = query_history[q_idx] if q_idx < len(query_history) else {}
            if q_idx < len(query_history):
                q_idx += 1

            thought_score = None
            if t_idx < len(scores):
                thought_score = scores[t_idx]
            else:
                maybe_score = _safe_call(search_score_fn, state, None)
                thought_score = float(maybe_score) if maybe_score is not None else None

            final_valid = None
            if t_idx < len(validity):
                final_valid = bool(validity[t_idx])
            elif state.get("part") == "final":
                final_valid = bool(_safe_call(final_validator_fn, state, False))

            step_ground_truth = None
            if t_idx < len(solved):
                step_ground_truth = bool(solved[t_idx])
            elif state.get("part") == "final":
                step_ground_truth = bool(_safe_call(ground_truth_fn, state, False))

            step_records.append(
                {
                    "step_index": len(step_records),
                    "operation_index": op_idx,
                    "operation": operation_name,
                    "thought_index": t_idx,
                    "phase": state.get("phase"),
                    "part": state.get("part"),
                    "task": state.get("task", routed_task),
                    "method": state.get("method"),
                    "state": state,
                    "prompt": query_event.get("prompt"),
                    "responses": query_event.get("responses", []),
                    "search_score": thought_score,
                    "final_validator": final_valid,
                    "ground_truth": step_ground_truth,
                    "query_tokens": {
                        "prompt_tokens": query_event.get("prompt_tokens_delta"),
                        "completion_tokens": query_event.get("completion_tokens_delta"),
                        "cost_delta": query_event.get("cost_delta"),
                    },
                    "node_uid": (
                        f"{sample.get('id')}|{op_idx}|{t_idx}|{state.get('part') or ''}"
                    ),
                    "reward_terms": {
                        "search_score": thought_score,
                        "final_validator": final_valid,
                        "ground_truth": step_ground_truth,
                    },
                }
            )

    final_prompt = ""
    final_response = ""
    for step in reversed(step_records):
        responses = step.get("responses") or []
        if responses:
            final_prompt = step.get("prompt") or ""
            final_response = str(responses[-1])
            break

    return {
        "sample_id": sample.get("id"),
        "routed_task": routed_task,
        "raw_task": sample.get("task", ""),
        "query": sample.get("query", ""),
        "gold_answer": sample.get("answer", ""),
        "final_answer": final_answer,
        "is_correct": bool(is_correct),
        "error_message": error_message,
        "token_usage": {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(prompt_tokens + completion_tokens),
            "cost": float(cost),
        },
        "final_prompt": final_prompt,
        "final_response": final_response,
        "query_history": query_history,
        "steps": step_records,
    }


def export_trajectories_jsonl(path: str, trajectories: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in trajectories:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def export_trajectories_parquet(path: str, trajectories: List[Dict[str, Any]]) -> bool:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return False

    # Store json string columns to keep schema stable for nested fields.
    rows = []
    for item in trajectories:
        rows.append(
            {
                "sample_id": item.get("sample_id"),
                "routed_task": item.get("routed_task"),
                "is_correct": item.get("is_correct"),
                "final_answer": item.get("final_answer", ""),
                "gold_answer": item.get("gold_answer", ""),
                "cost": item.get("token_usage", {}).get("cost", 0.0),
                "total_tokens": item.get("token_usage", {}).get("total_tokens", 0),
                "trajectory_json": json.dumps(item, ensure_ascii=False),
            }
        )
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    return True
