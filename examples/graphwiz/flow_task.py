import re
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

try:
    from .common import BaseTaskParser, BaseTaskPrompter, build_task_graph
    from . import utils
except ImportError:
    from common import BaseTaskParser, BaseTaskPrompter, build_task_graph
    import utils


TASK_NAME = "flow"
METHOD_NAME = "structured::flow"

# 改进思路：
# 1) 去掉容易制造噪声的 OptimalityCheck 分支，只保留“上界”和“可行下界”两条主线。
# 2) 在 parser 的 aggregation / improve 阶段做程序化数值协调：
#    - 收集 upper bounds / feasible flows
#    - 加入 easy upper bound = min(sum out(source), sum in(sink))
#    - 只在一致范围内选择最终值
# 3) 强化 score / validator，严惩 > easy upper bound 的不可能答案。
BRANCHES: List[Dict[str, Any]] = [
    {
        "part": "SourceSinkBounds",
        "goal": "Compute the strongest easy upper bound from source outgoing capacity, sink incoming capacity, and one small cut / bottleneck if obvious.",
        "num_generate": 4,
        "keep_n": 2,
    },
    {
        "part": "AugmentingPathPlan",
        "goal": "Construct a feasible lower bound by combining capacity-compatible augmenting paths without double-counting shared edges.",
        "num_generate": 6,
        "keep_n": 3,
    },
]

_FLOW_EDGE_ARROW_RE = re.compile(r"\((\d+)\s*->\s*(\d+)\s*,\s*(-?\d+(?:\.\d+)?)\)")
_FLOW_EDGE_TUPLE_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\s*,\s*(-?\d+(?:\.\d+)?)\)")
_FLOW_PAIR_PATTERNS = [
    re.compile(
        r"(?:maximum\s+flow|flow)\s+from\s+node\s+(\d+)\s+to\s+node\s+(\d+)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"source\s*(?:node)?\s*(\d+)\s*(?:and|,)?\s*sink\s*(?:node)?\s*(\d+)",
        flags=re.IGNORECASE,
    ),
    re.compile(r"from\s+(\d+)\s+to\s+(\d+)", flags=re.IGNORECASE),
]

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_UPPER_RE = re.compile(r"upper\s*bound\s*[:=]?\s*(-?\d+(?:\.\d+)?)", flags=re.IGNORECASE)
_FEASIBLE_RE = re.compile(
    r"(?:feasible\s*flow|lower\s*bound|candidate\s*flow|max\s*flow|flow\s*value)\s*[:=]?\s*(-?\d+(?:\.\d+)?)",
    flags=re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    fn = getattr(utils, "clean_response", None)
    if callable(fn):
        return fn(text)
    return (text or "").strip()


def _extract_last_number(text: str) -> Optional[float]:
    fn = getattr(utils, "extract_last_number", None)
    if callable(fn):
        return fn(text)
    matches = _NUMBER_RE.findall(text or "")
    if not matches:
        return None
    try:
        return float(matches[-1])
    except Exception:
        return None


def _extract_final_line(text: str) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _looks_truncated(text: str) -> bool:
    t = _clean_text(text)
    if not t:
        return False
    if "###" not in t and len(t) > 1200:
        return True
    tail = t.rstrip()[-6:]
    bad_suffixes = (":", ",", "(", "[", "{", "->", "-", "=")
    return any(t.rstrip().endswith(s) for s in bad_suffixes) or tail.endswith("...")


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.10g}"


def _canonical_key(value: float) -> str:
    return _format_number(value)


def _extract_labeled_value(text: str, part: str = "") -> Optional[float]:
    cleaned = _clean_text(text)
    if part == "SourceSinkBounds":
        m = _UPPER_RE.search(cleaned)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    if part == "AugmentingPathPlan":
        m = _FEASIBLE_RE.search(cleaned)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return _extract_last_number(cleaned)


def _parse_flow_instance(query: str) -> Tuple[Dict[str, Any], Optional[Tuple[int, int]]]:
    text = query or ""

    edges: List[Tuple[int, int, float]] = []
    nodes = set()

    arrow_edges = _FLOW_EDGE_ARROW_RE.findall(text)
    tuple_edges = _FLOW_EDGE_TUPLE_RE.findall(text)

    raw_edges = arrow_edges if arrow_edges else tuple_edges

    for u_str, v_str, w_str in raw_edges:
        try:
            u, v = int(u_str), int(v_str)
            w = float(w_str)
        except Exception:
            continue
        edges.append((u, v, w))
        nodes.add(u)
        nodes.add(v)

    pair = None
    for pat in _FLOW_PAIR_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                pair = (int(m.group(1)), int(m.group(2)))
                nodes.add(pair[0])
                nodes.add(pair[1])
                break
            except Exception:
                pass

    n = max(nodes) + 1 if nodes else 0
    return {"n": n, "edges": edges, "directed": True}, pair


def _easy_upper_bound(graph: Dict[str, Any], pair: Optional[Tuple[int, int]]) -> Optional[float]:
    if pair is None or not graph.get("edges"):
        return None
    s, t = pair
    out_sum = 0.0
    in_sum = 0.0
    for u, v, w in graph["edges"]:
        if w <= 0:
            continue
        if u == s:
            out_sum += float(w)
        if v == t:
            in_sum += float(w)
    return min(out_sum, in_sum)


def _is_integral_instance(graph: Dict[str, Any]) -> bool:
    edges = graph.get("edges", [])
    if not edges:
        return False
    return all(abs(w - round(w)) < 1e-9 for _, _, w in edges)


def _normalize_value_for_instance(value: float, graph: Dict[str, Any]) -> float:
    if _is_integral_instance(graph) and abs(value - round(value)) < 1e-6:
        return float(int(round(value)))
    return float(value)


def _collect_branch_values(states: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    uppers: List[float] = []
    lowers: List[float] = []
    for st in states:
        part = st.get("part", "")
        text = st.get("current", "")
        val = _extract_labeled_value(text, part)
        if val is None:
            continue
        if part == "SourceSinkBounds":
            uppers.append(float(val))
        elif part == "AugmentingPathPlan":
            lowers.append(float(val))
    return uppers, lowers


def _pick_consistent_final_value(
    original: str,
    states: List[Dict[str, Any]],
    aggregate_texts: List[str],
) -> Optional[float]:
    graph, pair = _parse_flow_instance(original)
    easy_ub = _easy_upper_bound(graph, pair)

    branch_uppers, branch_lowers = _collect_branch_values(states)
    agg_nums = [
        float(v)
        for v in (
            _extract_last_number(_clean_text(text)) for text in aggregate_texts
        )
        if v is not None
    ]

    # 清洗 branch candidates
    valid_uppers = [u for u in branch_uppers if u >= 0]
    valid_lowers = [l for l in branch_lowers if l >= 0]

    if easy_ub is not None:
        valid_uppers.append(float(easy_ub))
        valid_lowers = [l for l in valid_lowers if l <= easy_ub + 1e-9]
        agg_nums = [a for a in agg_nums if 0 <= a <= easy_ub + 1e-9]
    else:
        agg_nums = [a for a in agg_nums if a >= 0]

    if not valid_uppers and not valid_lowers and not agg_nums:
        return None

    # 当前最可信的上界帽子
    upper_cap = min(valid_uppers) if valid_uppers else None

    # 先看有没有“上下界都支持”的共识值
    if valid_uppers and valid_lowers:
        lower_counter = Counter(_canonical_key(x) for x in valid_lowers)
        upper_counter = Counter(_canonical_key(x) for x in valid_uppers)
        common_keys = set(lower_counter) & set(upper_counter)
        if common_keys:
            common_vals = [float(k) for k in common_keys]
            common_vals.sort()
            chosen = common_vals[-1]
            return _normalize_value_for_instance(chosen, graph)

    # 再看 aggregate 自己有没有给出一个落在 [max_lower, min_upper] 内的值
    lower_floor = max(valid_lowers) if valid_lowers else 0.0
    if upper_cap is not None:
        agg_consistent = [a for a in agg_nums if lower_floor - 1e-9 <= a <= upper_cap + 1e-9]
        if agg_consistent:
            chosen = max(agg_consistent)
            return _normalize_value_for_instance(chosen, graph)

        consistent_lowers = [l for l in valid_lowers if l <= upper_cap + 1e-9]
        if consistent_lowers:
            chosen = max(consistent_lowers)
            return _normalize_value_for_instance(chosen, graph)

        # 实在没有可信 lower，则退到更保守的上界帽子
        return _normalize_value_for_instance(upper_cap, graph)

    # 没有上界时，只能相信最强的可行下界或 aggregate 给出的非负值
    if valid_lowers:
        chosen = max(valid_lowers)
        return _normalize_value_for_instance(chosen, graph)
    if agg_nums:
        chosen = max(agg_nums)
        return _normalize_value_for_instance(chosen, graph)

    return None


def phase0_prompt(task: str) -> str:
    return """You are entering a strictly structured graph-reasoning workflow for maximum flow.

Return EXACTLY the following JSON and nothing else:
{"ack":"ok"}
"""


def branch_prompt(task: str, part: str, goal: str, original_query: str) -> str:
    if part == "SourceSinkBounds":
        return f"""You are solving the SourceSinkBounds branch of a structured Graph-of-Thought workflow for maximum flow.

Original problem:
{original_query}

Goal:
Compute the strongest EASY upper bound.

Rules:
1. Use source outgoing capacity total, sink incoming capacity total, and at most one obvious cut / bottleneck if helpful.
2. Do NOT try to compute the whole max-flow proof here.
3. Output only ONE strongest easy upper bound.
4. Keep the explanation very short.
5. End with exactly one final line:
### UpperBound: <number>
"""
    return f"""You are solving the AugmentingPathPlan branch of a structured Graph-of-Thought workflow for maximum flow.

Original problem:
{original_query}

Goal:
Construct a FEASIBLE lower bound by combining capacity-compatible augmenting paths.

Rules:
1. Only count simultaneously feasible flow.
2. If two paths share an edge, respect the shared capacity.
3. Prefer 1-3 concise path statements, not long prose.
4. Do NOT claim optimality in this branch.
5. End with exactly one final line:
### FeasibleFlow: <number>
"""


def aggregate_prompt(original_query: str, branch_bundle: str) -> str:
    return f"""You are given branch analyses for a maximum-flow problem.

Original problem:
{original_query}

Branch analyses:
{branch_bundle}

Decision rules:
1. Use the strongest credible easy upper bound and the strongest credible feasible flow.
2. If they match, output that exact value.
3. Otherwise, output the largest feasible value that does NOT exceed the best upper bound.
4. Never output a value larger than the total capacity leaving the source or entering the sink.
5. Keep the final answer extremely short.

Your final line must be:
### <number>
"""


def improve_prompt(original_query: str, current: str) -> str:
    return f"""Repair the maximum-flow answer.

Original problem:
{original_query}

Previous answer:
{current}

Rules:
1. Output only one numeric maximum-flow value.
2. Remove any over-counting caused by incompatible shared edges.
3. Do not output a value larger than the total source-outgoing or sink-incoming capacity.
4. Prefer the largest feasible value that is still upper-bounded.

Your final line must be:
### <number>
"""


def search_score(state: Dict[str, Any]) -> float:
    text = _clean_text(state.get("current", ""))
    part = state.get("part", "")
    low = text.lower()
    value = _extract_labeled_value(text, part)

    graph, pair = _parse_flow_instance(state.get("original", ""))
    easy_ub = _easy_upper_bound(graph, pair)

    if not text:
        return 100.0

    score = 0.0

    if "###" not in text:
        score += 8.0
    if value is None:
        score += 24.0
    if _looks_truncated(text):
        score += 16.0
    if len(text) > 1400:
        score += 8.0
    elif len(text) > 800:
        score += 3.0

    if value is not None and value < -1e-9:
        score += 40.0

    if easy_ub is not None and value is not None:
        if part == "SourceSinkBounds":
            # 作为 strongest easy upper bound，明显高于 easy_ub 会被认为很弱
            if value > easy_ub + 1e-9:
                score += 10.0
        else:
            # 下界 / 最终答案超过 easy upper bound 是不可能的
            if value > easy_ub + 1e-9:
                score += 40.0

    if part == "SourceSinkBounds":
        if not any(k in low for k in ["upperbound", "upper bound", "source", "sink", "cut", "bottleneck"]):
            score += 10.0
    elif part == "AugmentingPathPlan":
        if not any(k in low for k in ["feasibleflow", "feasible flow", "path", "augment", "flow"]):
            score += 10.0
        if any(k in low for k in ["optimal", "certified", "therefore maximum"]) and "feasibleflow" not in low:
            score += 8.0
    elif part == "final":
        if not final_validator(state):
            score += 30.0

    score += min(len(text) / 2500.0, 5.0)
    return float(score)


def final_validator(state: Dict[str, Any]) -> bool:
    text = state.get("current", "")
    value = _extract_last_number(text)
    if value is None:
        return False
    if value < -1e-9:
        return False
    if _looks_truncated(text):
        return False

    final_line = _extract_final_line(text)
    if not final_line:
        return False
    if "###" not in final_line and len(_clean_text(text)) > 100:
        return False

    graph, pair = _parse_flow_instance(state.get("original", ""))
    easy_ub = _easy_upper_bound(graph, pair)
    if easy_ub is not None and value > easy_ub + 1e-9:
        return False

    return True


def _edmonds_karp(graph: Dict[str, Any], source: int, sink: int) -> Optional[float]:
    n = graph["n"]
    if n <= 0 or source < 0 or sink < 0 or source >= n or sink >= n:
        return None

    cap = [[0.0 for _ in range(n)] for _ in range(n)]
    for u, v, w in graph["edges"]:
        if w < -1e-12:
            return None
        if w > 0:
            cap[u][v] += float(w)

    flow = 0.0
    while True:
        parent = [-1] * n
        parent[source] = source
        q = deque([source])

        while q and parent[sink] == -1:
            u = q.popleft()
            for v in range(n):
                if parent[v] == -1 and cap[u][v] > 1e-12:
                    parent[v] = u
                    q.append(v)

        if parent[sink] == -1:
            break

        aug = float("inf")
        v = sink
        while v != source:
            u = parent[v]
            aug = min(aug, cap[u][v])
            v = u

        v = sink
        while v != source:
            u = parent[v]
            cap[u][v] -= aug
            cap[v][u] += aug
            v = u

        flow += aug

    return flow


def ground_truth(state: Dict[str, Any]) -> bool:
    pred = _extract_last_number(state.get("current", ""))
    if pred is None:
        fn = getattr(utils, "graphwiz_ground_truth", None)
        return fn(state) if callable(fn) else False

    graph, pair = _parse_flow_instance(state.get("original", ""))
    if pair is None or not graph["edges"]:
        gold_num = _extract_last_number(state.get("gold", ""))
        if gold_num is None:
            fn = getattr(utils, "graphwiz_ground_truth", None)
            return fn(state) if callable(fn) else False
        return abs(pred - gold_num) < 1e-6

    truth = _edmonds_karp(graph, pair[0], pair[1])
    if truth is None:
        gold_num = _extract_last_number(state.get("gold", ""))
        if gold_num is not None:
            return abs(pred - gold_num) < 1e-6
        fn = getattr(utils, "graphwiz_ground_truth", None)
        return fn(state) if callable(fn) else False

    return abs(pred - truth) < 1e-6


class FlowParser(BaseTaskParser):
    def _canonicalize_branch_output(
        self,
        part: str,
        text: str,
        original: str,
    ) -> str:
        cleaned = _clean_text(text)
        value = _extract_labeled_value(cleaned, part)
        if value is None:
            return cleaned

        graph, _ = _parse_flow_instance(original)
        value = _normalize_value_for_instance(value, graph)

        if part == "SourceSinkBounds":
            return f"### UpperBound: {_format_number(value)}"
        if part == "AugmentingPathPlan":
            return f"### FeasibleFlow: {_format_number(value)}"
        return f"### {_format_number(value)}"

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        phase = state.get("phase", 0)
        if phase == 0:
            return super().parse_generate_answer(state, texts)

        part = state.get("part", "")
        original = state.get("original", "")
        new_states: List[Dict] = []

        for text in texts:
            cleaned = self._canonicalize_branch_output(part, text, original)
            if cleaned:
                new_states.append(
                    {
                        "current": cleaned,
                        "phase": phase + 1,
                    }
                )
        return new_states

    def parse_aggregation_answer(self, states: List[Dict], texts: List[str]) -> List[Dict]:
        if not states:
            return []

        original = states[0].get("original", "")
        chosen = _pick_consistent_final_value(original, states, texts)
        if chosen is not None:
            graph, _ = _parse_flow_instance(original)
            chosen = _normalize_value_for_instance(chosen, graph)
            return [
                {
                    "current": f"### {_format_number(chosen)}",
                    "phase": 3,
                    "part": "final",
                    "branch_goal": "",
                }
            ]

        # fallback：若程序协调失败，退回父类
        return super().parse_aggregation_answer(states, texts)

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        original = state.get("original", "")
        graph, pair = _parse_flow_instance(original)
        easy_ub = _easy_upper_bound(graph, pair)

        candidates: List[float] = []

        old_val = _extract_last_number(state.get("current", ""))
        if old_val is not None and old_val >= 0:
            candidates.append(old_val)

        for text in texts:
            val = _extract_last_number(_clean_text(text))
            if val is not None and val >= 0:
                candidates.append(val)

        if easy_ub is not None:
            candidates = [v for v in candidates if v <= easy_ub + 1e-9]

        if candidates:
            chosen = max(candidates)
            chosen = _normalize_value_for_instance(chosen, graph)
            return {
                "current": f"### {_format_number(chosen)}",
                "phase": state.get("phase", 3),
                "part": state.get("part", "final"),
            }

        return super().parse_improve_answer(state, texts)


def build_graph():
    return build_task_graph(
        branches=BRANCHES,
        search_score_fn=search_score,
        final_validator=final_validator,
        ground_truth_fn=ground_truth,
        aggregate_responses=5,
    )


def get_prompter():
    return BaseTaskPrompter(
        task_name=TASK_NAME,
        phase0_prompt_fn=phase0_prompt,
        branch_prompt_fn=branch_prompt,
        aggregate_prompt_fn=aggregate_prompt,
        improve_prompt_fn=improve_prompt,
    )


def get_parser():
    return FlowParser(BRANCHES)