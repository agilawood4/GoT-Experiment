from typing import Any, Dict, List, Tuple

try:
    from . import reward_builder
except Exception:
    import reward_builder


def _extract_step_text(step: Dict[str, Any]) -> str:
    responses = step.get("responses") or []
    if responses:
        return str(responses[-1]).strip()
    state = step.get("state") or {}
    return str(state.get("current", "")).strip()


def _group_key(sample_id: Any, step: Dict[str, Any]) -> str:
    return "|".join(
        [
            str(sample_id),
            str(step.get("part", "")),
            str(step.get("phase", "")),
            str(step.get("operation", "")),
        ]
    )


class MultiNodePairBuilder:
    def __init__(self, min_node_gap: float = 0.05, max_pairs_per_group: int = 2) -> None:
        self.min_node_gap = float(min_node_gap)
        self.max_pairs_per_group = int(max_pairs_per_group)

    def _grouped_items(self, trajectory: Dict[str, Any]) -> Dict[str, List[Tuple[float, Dict[str, Any], str]]]:
        out: Dict[str, List[Tuple[float, Dict[str, Any], str]]] = {}
        sample_id = trajectory.get("sample_id", "")
        for step in trajectory.get("steps", []) or []:
            part = str(step.get("part", "")).lower()
            if part in {"", "root", "none"}:
                continue
            text = _extract_step_text(step)
            if not text:
                continue
            score = reward_builder.compute_node_preference_score(step, trajectory)
            gk = _group_key(sample_id, step)
            out.setdefault(gk, []).append((score, step, text))
        return out

    def build_pairs_from_trajectory(self, trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
        grouped = self._grouped_items(trajectory)
        pairs: List[Dict[str, Any]] = []
        default_prompt = trajectory.get("final_prompt") or trajectory.get("query", "")
        for gk, items in grouped.items():
            if len(items) < 2:
                continue
            items = sorted(items, key=lambda x: x[0])
            low = items[0]
            high = items[-1]
            gap = float(high[0] - low[0])
            if gap < self.min_node_gap:
                continue

            prompt = str(high[1].get("prompt") or low[1].get("prompt") or default_prompt or "").strip()
            chosen = str(high[2]).strip()
            rejected = str(low[2]).strip()
            if not prompt or not chosen or not rejected or chosen == rejected:
                continue

            pairs.append(
                {
                    "node_group_key": gk,
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "score_gap": gap,
                    "task": trajectory.get("routed_task", ""),
                    "part": high[1].get("part", ""),
                    "phase": high[1].get("phase", ""),
                    "sample_id": trajectory.get("sample_id", ""),
                    "metadata": {
                        "chosen_step_index": high[1].get("step_index"),
                        "rejected_step_index": low[1].get("step_index"),
                        "chosen_score": float(high[0]),
                        "rejected_score": float(low[0]),
                        "is_correct": bool(trajectory.get("is_correct", False)),
                    },
                }
            )

            if self.max_pairs_per_group > 1:
                added = 0
                for i in range(len(items) - 1, 0, -1):
                    if added >= self.max_pairs_per_group - 1:
                        break
                    c = items[i]
                    r = items[i - 1]
                    gap2 = float(c[0] - r[0])
                    if gap2 < self.min_node_gap:
                        continue
                    if str(c[2]).strip() == str(r[2]).strip():
                        continue
                    pairs.append(
                        {
                            "node_group_key": gk,
                            "prompt": prompt,
                            "chosen": str(c[2]).strip(),
                            "rejected": str(r[2]).strip(),
                            "score_gap": gap2,
                            "task": trajectory.get("routed_task", ""),
                            "part": c[1].get("part", ""),
                            "phase": c[1].get("phase", ""),
                            "sample_id": trajectory.get("sample_id", ""),
                            "metadata": {
                                "chosen_step_index": c[1].get("step_index"),
                                "rejected_step_index": r[1].get("step_index"),
                                "chosen_score": float(c[0]),
                                "rejected_score": float(r[0]),
                                "is_correct": bool(trajectory.get("is_correct", False)),
                            },
                        }
                    )
                    added += 1
        return pairs
