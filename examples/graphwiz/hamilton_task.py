import re
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from .common import BaseTaskParser, BaseTaskPrompter, build_task_graph
    from . import utils
except ImportError:
    from common import BaseTaskParser, BaseTaskPrompter, build_task_graph
    import utils


TASK_NAME = "hamilton"
METHOD_NAME = "structured::hamilton"

# deepseek-v3.2 当前接口上限为 n<=4，因此这里不要再设成 5
BRANCHES: List[Dict[str, Any]] = [
    {
        "part": "DegreeAndEndpointHeuristics",
        "goal": "Analyze degree-0/1 nodes, forced endpoints, disconnected pieces, and direction constraints that strongly affect Hamilton-path existence.",
        "num_generate": 3,
        "keep_n": 1,
    },
    {
        "part": "CandidateCoverWalk",
        "goal": "Try to construct a node-covering walk that visits every node exactly once; if successful, conclude Yes.",
        "num_generate": 4,
        "keep_n": 2,
    },
    {
        "part": "ObstructionCheck",
        "goal": "Look for concrete reasons a Hamilton path cannot exist, such as isolated nodes, too many forced endpoints, broken reachability, or unavoidable dead ends.",
        "num_generate": 3,
        "keep_n": 1,
    },
]

_EDGE_UNDIRECTED_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\)")
_EDGE_DIRECTED_RE = re.compile(r"\((\d+)\s*->\s*(\d+)\)")
_YES_RE = re.compile(r"\byes\b", flags=re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b", flags=re.IGNORECASE)

_NODE_LIST_PATTERNS = [
    re.compile(r"(?:nodes|vertices)\s*(?:are|:)\s*[\[\{]?\s*([\d,\s]+)\s*[\]\}]?", flags=re.IGNORECASE),
    re.compile(r"(?:node\s+set|vertex\s+set)\s*(?:is|:)\s*[\[\{]?\s*([\d,\s]+)\s*[\]\}]?", flags=re.IGNORECASE),
]

_NODE_RANGE_PATTERNS = [
    re.compile(r"(?:nodes|vertices)\s+are\s+numbered\s+from\s+(\d+)\s+to\s+(\d+)", flags=re.IGNORECASE),
    re.compile(r"(?:nodes|vertices)\s+numbered\s+from\s+(\d+)\s+to\s+(\d+)", flags=re.IGNORECASE),
]


def _clean_text(text: str) -> str:
    fn = getattr(utils, "clean_response", None)
    if callable(fn):
        return fn(text)
    return (text or "").strip()


def _extract_final_line(text: str) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _extract_yes_no(text: str) -> Optional[bool]:
    cleaned = _clean_text(text)
    final_line = _extract_final_line(cleaned).lower()

    # 优先看最终结论行，避免正文里同时出现 yes/no 干扰判断
    if final_line in {"### yes", "yes"}:
        return True
    if final_line in {"### no", "no"}:
        return False

    fn = getattr(utils, "extract_yes_no", None)
    if callable(fn):
        out = fn(cleaned)
        if out is not None:
            s = str(out).strip().lower()
            if s in {"yes", "### yes"}:
                return True
            if s in {"no", "### no"}:
                return False

    # 若全文同时出现 yes/no，则优先取最后一次出现的那个，尽量贴近最终答案
    matches = list(re.finditer(r"\b(yes|no)\b", cleaned, flags=re.IGNORECASE))
    if matches:
        last = matches[-1].group(1).lower()
        return last == "yes"
    return None


def _looks_truncated(text: str) -> bool:
    t = _clean_text(text)
    if not t:
        return False
    if "###" not in t and len(t) > 1000:
        return True
    if t.rstrip().endswith((":", ",", "(", "[", "{", "->", "-", "=")):
        return True
    return t.rstrip().endswith("...")


def _extract_explicit_nodes(text: str) -> Set[int]:
    nodes: Set[int] = set()

    for pat in _NODE_LIST_PATTERNS:
        for m in pat.finditer(text or ""):
            chunk = m.group(1)
            for x in re.findall(r"\d+", chunk):
                try:
                    nodes.add(int(x))
                except Exception:
                    pass

    for pat in _NODE_RANGE_PATTERNS:
        for m in pat.finditer(text or ""):
            try:
                lo = int(m.group(1))
                hi = int(m.group(2))
            except Exception:
                continue
            if lo <= hi:
                nodes.update(range(lo, hi + 1))
            else:
                nodes.update(range(hi, lo + 1))

    return nodes


def _parse_graph(query: str) -> Dict[str, Any]:
    text = query or ""
    text_low = text.lower()

    directed_edges = _EDGE_DIRECTED_RE.findall(text)
    undirected_edges = _EDGE_UNDIRECTED_RE.findall(text)

    if directed_edges:
        directed = True
        raw_edges = directed_edges
    else:
        raw_edges = undirected_edges
        # 先判 undirected，再判 directed，避免 "undirected graph" 被误识别成 directed
        if re.search(r"\bundirected\s+graph\b|\bundirected\b", text_low):
            directed = False
        elif re.search(r"\bdirected\s+graph\b|\bdigraph\b|\bdirected\b", text_low):
            directed = True
        else:
            directed = False

    raw_edge_list: List[Tuple[int, int]] = []
    nodes: Set[int] = set()

    for a_str, b_str in raw_edges:
        try:
            a, b = int(a_str), int(b_str)
        except Exception:
            continue
        raw_edge_list.append((a, b))
        nodes.add(a)
        nodes.add(b)

    nodes |= _extract_explicit_nodes(text)
    node_ids = sorted(nodes)
    if not node_ids:
        return {"n": 0, "edges": [], "directed": directed, "nodes": []}

    # 规范化为连续编号，避免节点编号不是从 0 连续开始时出现 phantom node
    id_map = {node_id: idx for idx, node_id in enumerate(node_ids)}
    edges: List[Tuple[int, int]] = []
    seen_edges: Set[Tuple[int, int]] = set()

    for u_raw, v_raw in raw_edge_list:
        u = id_map[u_raw]
        v = id_map[v_raw]
        edge = (u, v)
        if edge in seen_edges:
            continue
        seen_edges.add(edge)
        edges.append(edge)

    return {"n": len(node_ids), "edges": edges, "directed": directed, "nodes": node_ids}


def _build_adj(graph: Dict[str, Any]) -> Dict[int, List[int]]:
    adj = {i: [] for i in range(graph["n"])}
    directed = bool(graph.get("directed", False))

    for u, v in graph["edges"]:
        adj[u].append(v)
        if not directed:
            adj[v].append(u)

    return adj


def _has_hamilton_path(graph: Dict[str, Any]) -> bool:
    n = graph["n"]
    if n <= 0:
        return False
    if n > 22:
        # 图过大时，这个精确 GT 代价过高；保守退回 False 让上层 fallback 到 gold
        return False

    adj = _build_adj(graph)
    target_mask = (1 << n) - 1
    memo: Set[Tuple[int, int]] = set()

    def dfs(u: int, mask: int) -> bool:
        key = (u, mask)
        if key in memo:
            return False
        if mask == target_mask:
            return True
        memo.add(key)
        for v in adj.get(u, []):
            if mask & (1 << v):
                continue
            if dfs(v, mask | (1 << v)):
                return True
        return False

    for start in range(n):
        if dfs(start, 1 << start):
            return True
    return False


def phase0_prompt(task: str) -> str:
    return """You are entering a strictly structured graph-reasoning workflow for Hamiltonian path.

Return EXACTLY the following JSON and nothing else:
{"ack":"ok"}
"""


def branch_prompt(task: str, part: str, goal: str, original_query: str) -> str:
    if part == "DegreeAndEndpointHeuristics":
        return f"""You are solving the DegreeAndEndpointHeuristics branch of a structured Graph-of-Thought workflow.

Original problem:
{original_query}

Goal:
Analyze degree-0/1 nodes, forced endpoints, disconnected pieces, and direction constraints.

Rules:
1. Use structural evidence only.
2. Keep the explanation short.
3. End with exactly one final line:
### Yes
or
### No
"""
    if part == "CandidateCoverWalk":
        return f"""You are solving the CandidateCoverWalk branch of a structured Graph-of-Thought workflow.

Original problem:
{original_query}

Goal:
Try to construct a node-covering walk that visits every node exactly once.

Rules:
1. Conclude Yes only if a full valid node-covering path is internally supported.
2. Otherwise conclude No.
3. Do not print a very long candidate path.
4. End with exactly one final line:
### Yes
or
### No
"""
    return f"""You are solving the ObstructionCheck branch of a structured Graph-of-Thought workflow.

Original problem:
{original_query}

Goal:
Look for concrete reasons a Hamilton path cannot exist.

Rules:
1. Prefer explicit obstructions over vague intuition.
2. If no obstruction survives scrutiny, do not force a No.
3. Keep the explanation short.
4. End with exactly one final line:
### Yes
or
### No
"""


def aggregate_prompt(original_query: str, branch_bundle: str) -> str:
    return f"""You are given branch analyses for Hamiltonian path.

Original problem:
{original_query}

Branch analyses:
{branch_bundle}

Decision rules:
1. This is a Yes/No decision task.
2. Prefer Yes only when a full cover-walk is credibly supported.
3. Prefer No only when a concrete obstruction is supported.
4. Do not output an explicit long path in the final answer.
5. Keep the final answer extremely short.

Your final line must be:
### Yes
or
### No
"""


def improve_prompt(original_query: str, current: str) -> str:
    return f"""Repair the Hamilton-path answer.

Original problem:
{original_query}

Previous answer:
{current}

Rules:
1. Output only Yes or No.
2. Use Yes only if every node can be covered exactly once in one valid path.
3. Use No only if there is a genuine structural obstruction.
4. Keep it very short.

Your final line must be:
### Yes
or
### No
"""


def _validate_hamilton_response(state: Dict[str, Any]) -> bool:
    text = state.get("current", "")
    if _extract_yes_no(text) is None:
        return False
    if _looks_truncated(text):
        return False
    final_line = _extract_final_line(text)
    if not final_line:
        return False
    if "###" not in final_line and len(_clean_text(text)) > 80:
        return False
    return True


def search_score(state: Dict[str, Any]) -> float:
    text = _clean_text(state.get("current", ""))
    part = state.get("part", "")
    low = text.lower()

    if not text:
        return 100.0

    score = 0.0

    if "###" not in text:
        score += 10.0
    if _extract_yes_no(text) is None:
        score += 20.0
    if _looks_truncated(text):
        score += 16.0
    if len(text) > 1500:
        score += 10.0
    elif len(text) > 700:
        score += 4.0

    if part == "DegreeAndEndpointHeuristics":
        if not any(k in low for k in ["degree", "endpoint", "isolated", "connected", "component", "direction"]):
            score += 12.0
    elif part == "CandidateCoverWalk":
        if not any(k in low for k in ["visit", "cover", "walk", "path", "exactly once"]):
            score += 10.0
    elif part == "ObstructionCheck":
        if not any(k in low for k in ["obstruction", "dead end", "forced", "impossible", "disconnect", "unreachable"]):
            score += 12.0
    elif part == "final":
        if not _validate_hamilton_response(state):
            score += 30.0

    score += min(len(text) / 2500.0, 5.0)
    return float(score)


def final_validator(state: Dict[str, Any]) -> bool:
    return _validate_hamilton_response(state)


def ground_truth(state: Dict[str, Any]) -> bool:
    pred = _extract_yes_no(state.get("current", ""))
    if pred is None:
        fn = getattr(utils, "graphwiz_ground_truth", None)
        return fn(state) if callable(fn) else False

    graph = _parse_graph(state.get("original", ""))
    if graph["n"] <= 0 or not graph["edges"]:
        gold = _extract_yes_no(state.get("gold", ""))
        if gold is None:
            fn = getattr(utils, "graphwiz_ground_truth", None)
            return fn(state) if callable(fn) else False
        return pred == gold

    if graph["n"] > 22:
        gold = _extract_yes_no(state.get("gold", ""))
        if gold is not None:
            return pred == gold
        fn = getattr(utils, "graphwiz_ground_truth", None)
        return fn(state) if callable(fn) else False

    exists = _has_hamilton_path(graph)
    return pred == exists


def build_graph():
    return build_task_graph(
        branches=BRANCHES,
        search_score_fn=search_score,
        final_validator=final_validator,
        ground_truth_fn=ground_truth,
        aggregate_responses=4,
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
    return BaseTaskParser(BRANCHES)
