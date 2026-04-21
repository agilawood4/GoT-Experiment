import argparse
import csv
import datetime
import importlib
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

# Ensure local repository source has higher priority than site-packages.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from graph_of_thoughts import controller, language_models

try:
    from . import utils
    from . import trajectory_exporter
    from . import reward_builder
    from .graphwiz_got import (
        StrongStructuredGraphWizParser,
        StrongStructuredGraphWizPrompter,
        build_strong_structured_got,
        route_task_name,
    )
except ImportError:
    import utils
    import trajectory_exporter
    import reward_builder
    from graphwiz_got import (
        StrongStructuredGraphWizParser,
        StrongStructuredGraphWizPrompter,
        build_strong_structured_got,
        route_task_name,
    )


TASK_MODULE_CANDIDATES = {
    "connectivity": ["connectivity_task"],
    "bipartite": ["bipartite_task"],
    "cycle": ["cycle_task"],
    "topology": ["topology_task"],
    "shortest_path": ["shortest_path"],
    "flow": ["flow_task"],
    "triangle": ["triangle_task"],
    "hamilton": ["hamilton_task"],
    "substructure": ["substructure_task"],
}


def parse_data_ids(text: str) -> List[int]:
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def snapshot_lm_usage(lm) -> Dict[str, float]:
    """
    尽量兼容不同语言模型实现上的字段名。
    """
    prompt_tokens = getattr(lm, "prompt_tokens", 0) or 0

    completion_tokens = getattr(lm, "completion_tokens", None)
    if completion_tokens is None:
        completion_tokens = getattr(lm, "response_tokens", 0) or 0

    cost = getattr(lm, "cost", 0.0) or 0.0

    return {
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "cost": float(cost),
    }


def extract_final_answer_from_executor(executor) -> str:
    """
    优先从 Controller.get_final_thoughts() 中抽取最终答案。
    """
    try:
        final_groups = executor.get_final_thoughts()
        candidates: List[str] = []

        for group in final_groups:
            if not group:
                continue
            for thought in group:
                state = getattr(thought, "state", None)
                if isinstance(state, dict):
                    current = state.get("current", "")
                    if current:
                        candidates.append(current)

        if candidates:
            return candidates[-1]
    except Exception:
        pass

    return ""


def build_eval_state(sample: Dict[str, Any], final_answer: str, routed_task: str) -> Dict[str, Any]:
    return {
        "task": routed_task,
        "original": sample["query"],
        "gold": sample["answer"],
        "current": final_answer,
    }


def import_task_module(task_name: str):
    """
    懒加载各 task 文件，避免顶层强依赖。
    """
    task_name = route_task_name(task_name)
    candidates = TASK_MODULE_CANDIDATES.get(task_name, [])

    for module_name in candidates:
        try:
            return importlib.import_module(f".{module_name}", package=__package__)
        except Exception:
            pass
        try:
            return importlib.import_module(module_name)
        except Exception:
            pass

    return None


def _infer_task_from_text(text: str) -> str:
    """
    当 sample['task'] 不可靠时，从文本中兜底猜任务类型。
    尽量按更具体的任务优先匹配，避免误判到 connectivity / generic。
    """
    t = (text or "").strip().lower()
    if not t:
        return "generic"

    # 更具体的关键词优先
    if "topology sorting" in t or "topological sorting" in t or "topological ordering" in t:
        return "topology"
    if "shortest path" in t:
        return "shortest_path"
    if "maximum flow" in t or ("source" in t and "sink" in t and "capacity" in t and "flow" in t):
        return "flow"
    if "hamiltonian" in t:
        return "hamilton"
    if "bipartite" in t:
        return "bipartite"
    if "subgraph" in t or "substructure" in t or "pattern graph" in t:
        return "substructure"
    if "triangle" in t:
        return "triangle"
    if "cycle" in t:
        return "cycle"

    # connectivity 放后面，避免把 shortest path / topology 误吸进去
    if "connectivity" in t or "connected" in t or "path between" in t:
        return "connectivity"

    return "generic"


def infer_routed_task(
    sample: Dict[str, Any],
    subset: Optional[str] = None,
    local_json_path: Optional[str] = None,
) -> str:
    """
    多级任务识别：
    1. subset（最可靠，比如 --subset topology）
    2. sample['task']
    3. meta.dataset_name
    4. local_json_path
    5. query 文本关键词兜底

    这样即使数据里 task 被写成 generic，也能正确分发到专用任务模块。
    """
    # 1) subset 最可靠
    if subset:
        routed = route_task_name(subset)
        if routed != "generic":
            return routed

    # 2) 样本自带 task
    raw_task = str(sample.get("task", "")).strip()
    routed = route_task_name(raw_task)
    if routed != "generic":
        return routed

    # 3) meta.dataset_name
    meta = sample.get("meta", {}) or {}
    dataset_name = str(meta.get("dataset_name", "")).strip().lower()
    guessed = _infer_task_from_text(dataset_name)
    if guessed != "generic":
        return guessed

    # 4) local_json_path
    local_path_text = str(local_json_path or "").strip().lower()
    guessed = _infer_task_from_text(local_path_text)
    if guessed != "generic":
        return guessed

    # 5) query 文本兜底
    query = str(sample.get("query", "")).strip().lower()
    guessed = _infer_task_from_text(query)
    if guessed != "generic":
        return guessed

    return "generic"


def get_task_runtime(task_name: str):
    """
    返回 (operations_graph, prompter_obj, parser_obj)
    优先使用任务专属实现；失败时退回 generic strong_structured。
    """
    module = import_task_module(task_name)
    if module is not None:
        build_graph_fn = getattr(module, "build_graph", None)
        get_prompter_fn = getattr(module, "get_prompter", None)
        get_parser_fn = getattr(module, "get_parser", None)

        if callable(build_graph_fn) and callable(get_prompter_fn) and callable(get_parser_fn):
            return build_graph_fn(), get_prompter_fn(), get_parser_fn()

    return (
        build_strong_structured_got(task_name),
        StrongStructuredGraphWizPrompter(),
        StrongStructuredGraphWizParser(),
    )


def get_task_ground_truth_fn(task_name: str) -> Callable[[Dict[str, Any]], bool]:
    """
    优先使用各任务自己的 ground_truth。
    如果导入失败，再退回 utils.graphwiz_ground_truth。
    """
    module = import_task_module(task_name)
    if module is not None:
        gt_fn = getattr(module, "ground_truth", None)
        if callable(gt_fn):
            return gt_fn
    return lambda state: bool(utils.graphwiz_ground_truth(state))


def evaluate_sample(
    sample: Dict[str, Any],
    final_answer: str,
    routed_task: Optional[str] = None,
) -> bool:
    """
    用任务专属 GT 判定正确性。
    """
    routed_task = route_task_name(
        routed_task or sample.get("routed_task") or sample.get("task", "")
    )
    state = build_eval_state(sample, final_answer, routed_task)
    gt_fn = get_task_ground_truth_fn(routed_task)

    try:
        return bool(gt_fn(state))
    except Exception:
        return bool(utils.graphwiz_ground_truth(state))


def init_task_stats() -> Dict[str, Any]:
    return {
        "num_samples_run": 0,
        "correct_count": 0,
        "wrong_count": 0,
        "accuracy": 0.0,
        "accuracy_percent": "0.00%",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
    }


def build_controller_state(sample: Dict[str, Any], routed_task: str) -> Dict[str, Any]:
    return {
        "sample_id": sample["id"],
        "task": routed_task,
        "original": sample["query"],
        "gold": sample["answer"],
        "current": "",
        "phase": 0,
        "part": "root",
        "branch_goal": "",
        "method": f"strong_structured::{routed_task}",
        "meta": sample.get("meta", {}),
    }


def _first_not_none(*values):
    for v in values:
        if v is not None:
            return v
    return None


def _convert_to_per_token(cost_value: float, unit: str) -> float:
    unit = (unit or "").strip().lower()
    if unit in {"per_token", "token", "1"}:
        return float(cost_value)
    if unit in {"per_1k", "per_1000", "1k"}:
        return float(cost_value) / 1000.0
    if unit in {"per_1m", "per_1000000", "1m"}:
        return float(cost_value) / 1_000_000.0
    # 默认按字段名语义：token_cost -> per_token
    return float(cost_value)


def load_model_pricing(config_path: str, model_name: str) -> Dict[str, float]:
    """
    从 language_models/config.json 中读取单价，返回每 token 单价。
    支持：
    - prompt_token_cost / response_token_cost
    - input_token_cost / output_token_cost
    - prompt_cost_per_1k / completion_cost_per_1k
    - pricing_unit: per_token / per_1k / per_1m
    """
    if not os.path.exists(config_path):
        return {"prompt_per_token": 0.0, "completion_per_token": 0.0}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return {"prompt_per_token": 0.0, "completion_per_token": 0.0}

    if not isinstance(cfg, dict):
        return {"prompt_per_token": 0.0, "completion_per_token": 0.0}

    block = cfg.get(model_name)
    if not isinstance(block, dict):
        models_block = cfg.get("models")
        if isinstance(models_block, dict):
            block = models_block.get(model_name)

    if not isinstance(block, dict):
        return {"prompt_per_token": 0.0, "completion_per_token": 0.0}

    unit = str(block.get("pricing_unit", "per_token"))

    prompt_token_cost = _first_not_none(
        block.get("prompt_token_cost"),
        block.get("input_token_cost"),
    )
    completion_token_cost = _first_not_none(
        block.get("response_token_cost"),
        block.get("completion_token_cost"),
        block.get("output_token_cost"),
    )

    if prompt_token_cost is not None or completion_token_cost is not None:
        return {
            "prompt_per_token": _convert_to_per_token(float(prompt_token_cost or 0.0), unit),
            "completion_per_token": _convert_to_per_token(float(completion_token_cost or 0.0), unit),
        }

    prompt_cost_per_1k = _first_not_none(
        block.get("prompt_cost_per_1k"),
        block.get("input_cost_per_1k"),
    )
    completion_cost_per_1k = _first_not_none(
        block.get("response_cost_per_1k"),
        block.get("completion_cost_per_1k"),
        block.get("output_cost_per_1k"),
    )

    if prompt_cost_per_1k is not None or completion_cost_per_1k is not None:
        return {
            "prompt_per_token": float(prompt_cost_per_1k or 0.0) / 1000.0,
            "completion_per_token": float(completion_cost_per_1k or 0.0) / 1000.0,
        }

    return {"prompt_per_token": 0.0, "completion_per_token": 0.0}


def estimate_cost_from_tokens(
    prompt_tokens: int,
    completion_tokens: int,
    pricing: Dict[str, float],
) -> float:
    return (
        prompt_tokens * float(pricing.get("prompt_per_token", 0.0)) +
        completion_tokens * float(pricing.get("completion_per_token", 0.0))
    )


def _load_model_block(config_path: str, model_name: str) -> Dict[str, Any]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        block = cfg.get(model_name, {})
        return block if isinstance(block, dict) else {}
    except Exception:
        return {}


def _build_runtime_lm(
    lm_config_path: str,
    lm_name: str,
    use_cache: bool,
    run_dir: str,
    override_model_path: str = "",
):
    lm_builder = getattr(language_models, "create_language_model", None)
    if not override_model_path:
        if callable(lm_builder):
            return lm_builder(lm_config_path, model_name=lm_name, cache=use_cache)
        return language_models.ChatGPT(lm_config_path, model_name=lm_name, cache=use_cache)

    block = _load_model_block(lm_config_path, lm_name)
    backend = str(block.get("backend", "")).lower()
    if backend not in {"local_hf", "hf_local", "qwen_local"}:
        if callable(lm_builder):
            return lm_builder(lm_config_path, model_name=lm_name, cache=use_cache)
        return language_models.ChatGPT(lm_config_path, model_name=lm_name, cache=use_cache)

    with open(lm_config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if lm_name in cfg and isinstance(cfg[lm_name], dict):
        cfg[lm_name]["model_path"] = override_model_path
        if not cfg[lm_name].get("tokenizer_path"):
            cfg[lm_name]["tokenizer_path"] = override_model_path
    runtime_cfg_path = os.path.join(run_dir, "online_runtime_lm_config.json")
    with open(runtime_cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    if callable(lm_builder):
        return lm_builder(runtime_cfg_path, model_name=lm_name, cache=use_cache)
    return language_models.ChatGPT(runtime_cfg_path, model_name=lm_name, cache=use_cache)


def run_graphwiz_eval(
    budget: float,
    lm_name: str,
    source: str = "test",
    subset: Optional[str] = None,
    max_samples: Optional[int] = None,
    data_ids: Optional[List[int]] = None,
    local_json_path: Optional[str] = None,
    data_root: str = "./data",
    prefer_local: bool = True,
    use_cache: bool = False,
    export_parquet: bool = False,
    alpha_tokens: float = 1e-6,
    online_dpo: bool = False,
    online_pref_backend: str = "sqlite",
    online_pref_db_path: str = "",
    online_batch_size: int = 128,
    online_update_interval_steps: int = 200,
    online_min_node_gap: float = 0.05,
    online_hot_swap: bool = True,
    online_poll_interval_sec: float = 5.0,
    online_dpo_output_root: str = "",
    online_dpo_beta: float = 0.1,
    online_dpo_lr: float = 5e-6,
    online_dpo_epochs: float = 1.0,
    online_dpo_max_steps: int = 30,
    online_max_checkpoints_keep: int = 3,
) -> str:
    """
    通用版 GraphWiz 统计脚本：
    - 支持任意 subset
    - 支持混合 task 自动路由
    - 统计 token / cost / accuracy
    - 与 graphwiz_got.py 的兼容接口保持一致
    """
    initial_budget = float(budget)

    data = utils.load_graphwiz_samples(
        source=source,
        subset=subset,
        max_samples=max_samples,
        local_json_path=local_json_path,
        data_root=data_root,
        prefer_local=prefer_local,
    )

    if data_ids:
        selected_data = [data[i] for i in data_ids]
    else:
        selected_data = data

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    ensure_dir(results_dir)

    source_tag = source if not subset else f"{source}_{subset}"
    cache_tag = "cache_on" if use_cache else "cache_off"
    run_dir = os.path.join(results_dir, f"{source_tag}_eval_{lm_name}_{cache_tag}_{timestamp}")
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "graphs"))

    logging.basicConfig(
        filename=os.path.join(run_dir, "run.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": source,
                "subset": subset,
                "max_samples": max_samples,
                "local_json_path": local_json_path,
                "data_root": data_root,
                "prefer_local": prefer_local,
                "num_selected_data": len(selected_data),
                "lm_name": lm_name,
                "initial_budget": initial_budget,
                "use_cache": use_cache,
                "export_parquet": export_parquet,
                "alpha_tokens": alpha_tokens,
                "online_dpo": bool(online_dpo),
                "online_pref_backend": online_pref_backend,
                "online_pref_db_path": online_pref_db_path,
                "online_batch_size": int(online_batch_size),
                "online_update_interval_steps": int(online_update_interval_steps),
                "online_min_node_gap": float(online_min_node_gap),
                "online_hot_swap": bool(online_hot_swap),
                "online_poll_interval_sec": float(online_poll_interval_sec),
                "online_dpo_output_root": online_dpo_output_root,
                "online_dpo_beta": float(online_dpo_beta),
                "online_dpo_lr": float(online_dpo_lr),
                "online_dpo_epochs": float(online_dpo_epochs),
                "online_dpo_max_steps": int(online_dpo_max_steps),
                "online_max_checkpoints_keep": int(online_max_checkpoints_keep),
                "mode": "graphwiz_eval_generic",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    lm_config_path = os.path.join(
        os.path.dirname(__file__),
        "../../graph_of_thoughts/language_models/config.json",
    )
    pricing = load_model_pricing(lm_config_path, lm_name)

    rows: List[Dict[str, Any]] = []
    trajectories: List[Dict[str, Any]] = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0
    correct_count = 0
    wrong_count = 0
    total_reward = 0.0

    task_stats: Dict[str, Dict[str, Any]] = defaultdict(init_task_stats)
    online_pairs_total = 0
    hot_swap_count = 0

    pref_store = None
    pair_builder = None
    worker = None
    model_registry = None
    active_model_override = ""
    last_seen_registry_version = 0
    last_hot_swap_step = 0
    lm_runtime = None
    lm_runtime_override = "__init__"

    if online_dpo:
        try:
            if __package__:
                from .model_registry import ModelRegistry
                from .multinode_pair_builder import MultiNodePairBuilder
                from .online_dpo_worker import AsyncDPOWorker
                from .online_pref_store import OnlinePreferenceStore
            else:
                from model_registry import ModelRegistry
                from multinode_pair_builder import MultiNodePairBuilder
                from online_dpo_worker import AsyncDPOWorker
                from online_pref_store import OnlinePreferenceStore
        except Exception as e:
            raise RuntimeError(
                "Failed to import online DPO modules. "
                "If you only want regular evaluation, set --online_dpo 0. "
                "If online DPO is required, fix training dependencies first."
            ) from e

        if str(online_pref_backend).lower() != "sqlite":
            raise ValueError("Only sqlite backend is currently supported for online preferences.")
        pref_db_path = online_pref_db_path or os.path.join(run_dir, "online_pref_store.sqlite3")
        pref_store = OnlinePreferenceStore(pref_db_path)
        pair_builder = MultiNodePairBuilder(min_node_gap=online_min_node_gap)
        model_registry = ModelRegistry(
            os.path.join(run_dir, "online_model_registry.json"),
            initial_checkpoint="",
        )
        dpo_output_root = online_dpo_output_root or os.path.join(run_dir, "online_dpo_ckpts")
        worker = AsyncDPOWorker(
            pref_store=pref_store,
            model_registry=model_registry,
            base_model_name_or_path=_load_model_block(lm_config_path, lm_name).get("model_path", lm_name),
            output_root=dpo_output_root,
            batch_size=online_batch_size,
            min_score_gap=online_min_node_gap,
            poll_interval_sec=online_poll_interval_sec,
            max_checkpoints_keep=online_max_checkpoints_keep,
            dpo_kwargs={
                "beta": online_dpo_beta,
                "learning_rate": online_dpo_lr,
                "num_train_epochs": online_dpo_epochs,
                "max_steps": online_dpo_max_steps,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "logging_steps": 5,
                "save_steps": max(10, int(online_dpo_max_steps) if int(online_dpo_max_steps) > 0 else 10),
                "max_train_samples": online_batch_size,
                "bf16": True,
            },
        )
        worker.start()

    try:
        for idx, sample in enumerate(selected_data):
            if budget <= 0.0:
                logging.error("Budget depleted before sample %s", sample["id"])
                break

            if online_dpo and online_hot_swap and model_registry is not None:
                if (idx - last_hot_swap_step) >= max(1, int(online_update_interval_steps)):
                    state = model_registry.get_state()
                    version = int(state.get("version", 0))
                    ckpt = str(state.get("active_checkpoint", "") or "")
                    if version > last_seen_registry_version and ckpt:
                        active_model_override = ckpt
                        last_seen_registry_version = version
                        last_hot_swap_step = idx
                        hot_swap_count += 1
                        logging.info("Online hot-swap activated: version=%s checkpoint=%s", version, ckpt)

            raw_task = str(sample.get("task", "")).strip()
            routed_task = infer_routed_task(
                sample,
                subset=subset,
                local_json_path=local_json_path,
            )
            sample["routed_task"] = routed_task

            logging.info(
                "Running idx=%s sample_id=%s raw_task=%s routed_task=%s budget_left=%s",
                idx,
                sample["id"],
                raw_task,
                routed_task,
                budget,
            )

            ensure_dir(os.path.join(run_dir, "graphs", routed_task))

            desired_override = active_model_override if online_dpo else ""
            if (lm_runtime is None) or (desired_override != lm_runtime_override):
                lm_runtime = _build_runtime_lm(
                    lm_config_path=lm_config_path,
                    lm_name=lm_name,
                    use_cache=use_cache,
                    run_dir=run_dir,
                    override_model_path=desired_override,
                )
                lm_runtime_override = desired_override
                logging.info(
                    "LM runtime (re)loaded. override_model_path=%s",
                    desired_override if desired_override else "<base-config>",
                )

            lm = lm_runtime
            before_history_len = len(getattr(lm, "query_history", []) or [])

            before = snapshot_lm_usage(lm)

            operations_graph, task_prompter, task_parser = get_task_runtime(routed_task)
            task_module = import_task_module(routed_task)
            search_score_fn = getattr(task_module, "search_score", utils.graphwiz_format_score)
            final_validator_fn = getattr(task_module, "final_validator", None)
            ground_truth_fn = getattr(task_module, "ground_truth", None)

            executor = controller.Controller(
                lm,
                operations_graph,
                task_prompter,
                task_parser,
                build_controller_state(sample, routed_task),
            )

            error_message = ""
            try:
                executor.run()
            except Exception as e:
                error_message = str(e)
                logging.exception(
                    "Exception while running sample_id=%s raw_task=%s routed_task=%s: %s",
                    sample["id"],
                    raw_task,
                    routed_task,
                    e,
                )

            output_json_path = os.path.join(
                run_dir,
                "graphs",
                routed_task,
                f"{sample['id']}.json",
            )

            try:
                executor.output_graph(output_json_path)
            except Exception as e:
                logging.exception("Failed to output graph for sample_id=%s: %s", sample["id"], e)

            after = snapshot_lm_usage(lm)
            prompt_delta = max(0, after["prompt_tokens"] - before["prompt_tokens"])
            completion_delta = max(0, after["completion_tokens"] - before["completion_tokens"])
            raw_cost_delta = max(0.0, after["cost"] - before["cost"])
            total_delta = prompt_delta + completion_delta

            if raw_cost_delta > 0.0:
                cost_delta = raw_cost_delta
                cost_source = "lm.cost"
            else:
                cost_delta = estimate_cost_from_tokens(
                    prompt_tokens=prompt_delta,
                    completion_tokens=completion_delta,
                    pricing=pricing,
                )
                cost_source = "token_fallback"

            final_answer = extract_final_answer_from_executor(executor)
            is_correct = evaluate_sample(sample, final_answer, routed_task=routed_task)

            trajectory = trajectory_exporter.build_trajectory_record(
                sample=sample,
                routed_task=routed_task,
                output_json_path=output_json_path,
                query_history=(getattr(lm, "query_history", []) or [])[before_history_len:],
                final_answer=final_answer,
                is_correct=is_correct,
                error_message=error_message,
                prompt_tokens=prompt_delta,
                completion_tokens=completion_delta,
                cost=cost_delta,
                search_score_fn=search_score_fn,
                final_validator_fn=final_validator_fn,
                ground_truth_fn=ground_truth_fn,
            )
            reward_info = reward_builder.compute_trajectory_reward(
                trajectory, alpha_tokens=alpha_tokens
            )
            trajectory["reward"] = reward_info["reward"]
            trajectory["reward_components"] = reward_info["components"]
            trajectory["step_rewards"] = reward_builder.compute_step_level_shaping(trajectory)
            trajectories.append(trajectory)
            total_reward += float(reward_info["reward"])

            if online_dpo and pref_store is not None and pair_builder is not None:
                node_pairs = pair_builder.build_pairs_from_trajectory(trajectory)
                inserted = pref_store.insert_many(node_pairs)
                online_pairs_total += inserted
                if inserted > 0:
                    logging.info(
                        "Online preference pairs inserted: sample_id=%s inserted=%s pending=%s",
                        sample["id"],
                        inserted,
                        pref_store.count_pending(min_score_gap=online_min_node_gap),
                    )

            if is_correct:
                correct_count += 1
            else:
                wrong_count += 1

            total_prompt_tokens += prompt_delta
            total_completion_tokens += completion_delta
            total_cost += cost_delta
            budget -= cost_delta

            task_stats[routed_task]["num_samples_run"] += 1
            task_stats[routed_task]["correct_count"] += int(is_correct)
            task_stats[routed_task]["wrong_count"] += int(not is_correct)
            task_stats[routed_task]["prompt_tokens"] += prompt_delta
            task_stats[routed_task]["completion_tokens"] += completion_delta
            task_stats[routed_task]["total_tokens"] += total_delta
            task_stats[routed_task]["total_cost"] += cost_delta

            rows.append(
                {
                    "index": idx,
                    "sample_id": sample["id"],
                    "raw_task": raw_task,
                    "routed_task": routed_task,
                    "subset": subset if subset is not None else "",
                    "is_correct": int(is_correct),
                    "is_wrong": int(not is_correct),
                    "prompt_tokens": prompt_delta,
                    "completion_tokens": completion_delta,
                    "total_tokens": total_delta,
                    "cost": round(cost_delta, 6),
                    "cost_source": cost_source,
                    "final_answer": final_answer,
                    "gold_answer": sample["answer"],
                    "output_json": output_json_path,
                    "error_message": error_message,
                    "reward": round(float(reward_info["reward"]), 6),
                }
            )

            logging.info(
                "Finished sample_id=%s correct=%s prompt_tokens=%s completion_tokens=%s cost=%s cost_source=%s budget_left=%s",
                sample["id"],
                is_correct,
                prompt_delta,
                completion_delta,
                cost_delta,
                cost_source,
                budget,
            )
    finally:
        if worker is not None:
            worker.stop()

    for task_name, stats in task_stats.items():
        n = stats["num_samples_run"]
        acc = (stats["correct_count"] / n) if n else 0.0
        stats["accuracy"] = acc
        stats["accuracy_percent"] = f"{acc * 100:.2f}%"
        stats["total_cost"] = round(stats["total_cost"], 6)

    csv_path = os.path.join(run_dir, "sample_stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "sample_id",
                "raw_task",
                "routed_task",
                "subset",
                "is_correct",
                "is_wrong",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cost",
                "cost_source",
                "final_answer",
                "gold_answer",
                "output_json",
                "error_message",
                "reward",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    task_summary_path = os.path.join(run_dir, "task_summary.json")
    with open(task_summary_path, "w", encoding="utf-8") as f:
        json.dump(task_stats, f, ensure_ascii=False, indent=2)

    task_csv_path = os.path.join(run_dir, "task_summary.csv")
    with open(task_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "num_samples_run",
                "correct_count",
                "wrong_count",
                "accuracy",
                "accuracy_percent",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "total_cost",
            ],
        )
        writer.writeheader()
        for task_name, stats in sorted(task_stats.items()):
            writer.writerow(
                {
                    "task": task_name,
                    "num_samples_run": stats["num_samples_run"],
                    "correct_count": stats["correct_count"],
                    "wrong_count": stats["wrong_count"],
                    "accuracy": stats["accuracy"],
                    "accuracy_percent": stats["accuracy_percent"],
                    "prompt_tokens": stats["prompt_tokens"],
                    "completion_tokens": stats["completion_tokens"],
                    "total_tokens": stats["total_tokens"],
                    "total_cost": stats["total_cost"],
                }
            )

    num_samples_run = len(rows)
    overall_accuracy = (correct_count / num_samples_run) if num_samples_run else 0.0

    summary = {
        "source": source,
        "subset": subset,
        "lm_name": lm_name,
        "use_cache": use_cache,
        "online_dpo": bool(online_dpo),
        "num_samples_selected": len(selected_data),
        "num_samples_run": num_samples_run,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "accuracy": overall_accuracy,
        "accuracy_percent": f"{overall_accuracy * 100:.2f}%",
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "initial_budget": round(initial_budget, 6),
        "spent_budget": round(total_cost, 6),
        "remaining_budget": round(budget, 6),
        "total_cost": round(total_cost, 6),
        "mean_reward": round((total_reward / num_samples_run) if num_samples_run else 0.0, 6),
        "online_pairs_total": int(online_pairs_total),
        "online_hot_swap_count": int(hot_swap_count),
        "csv_path": csv_path,
        "task_summary_path": task_summary_path,
        "task_csv_path": task_csv_path,
        "results_dir": run_dir,
    }
    if pref_store is not None:
        summary["online_pref_store"] = pref_store.get_stats()
    if model_registry is not None:
        summary["online_model_registry"] = model_registry.get_state()

    trajectories_jsonl_path = os.path.join(run_dir, "trajectories.jsonl")
    trajectory_exporter.export_trajectories_jsonl(trajectories_jsonl_path, trajectories)
    summary["trajectories_jsonl"] = trajectories_jsonl_path

    if export_parquet:
        trajectories_parquet_path = os.path.join(run_dir, "trajectories.parquet")
        parquet_ok = trajectory_exporter.export_trajectories_parquet(
            trajectories_parquet_path, trajectories
        )
        summary["trajectories_parquet"] = trajectories_parquet_path if parquet_ok else ""
        summary["trajectories_parquet_exported"] = bool(parquet_ok)

    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return run_dir


if __name__ == "__main__":
    parser_ = argparse.ArgumentParser()

    parser_.add_argument("--source", type=str, default="test", choices=["train", "test", "rft"])
    parser_.add_argument("--subset", type=str, default="")
    parser_.add_argument("--local_json_path", type=str, default="")
    parser_.add_argument("--data_root", type=str, default="./data")
    parser_.add_argument("--prefer_local", type=int, default=1)
    parser_.add_argument("--data_ids", type=str, default="")
    parser_.add_argument("--max_samples", type=int, default=0)
    parser_.add_argument("--budget", type=float, default=100.0)
    parser_.add_argument("--lm_name", type=str, default="chatgpt")
    parser_.add_argument("--use_cache", type=int, default=0)
    parser_.add_argument("--export_parquet", type=int, default=0)
    parser_.add_argument("--alpha_tokens", type=float, default=1e-6)
    parser_.add_argument("--online_dpo", type=int, default=0)
    parser_.add_argument("--online_pref_backend", type=str, default="sqlite")
    parser_.add_argument("--online_pref_db_path", type=str, default="")
    parser_.add_argument("--online_batch_size", type=int, default=128)
    parser_.add_argument("--online_update_interval_steps", type=int, default=200)
    parser_.add_argument("--online_min_node_gap", type=float, default=0.05)
    parser_.add_argument("--online_hot_swap", type=int, default=1)
    parser_.add_argument("--online_poll_interval_sec", type=float, default=5.0)
    parser_.add_argument("--online_dpo_output_root", type=str, default="")
    parser_.add_argument("--online_dpo_beta", type=float, default=0.1)
    parser_.add_argument("--online_dpo_lr", type=float, default=5e-6)
    parser_.add_argument("--online_dpo_epochs", type=float, default=1.0)
    parser_.add_argument("--online_dpo_max_steps", type=int, default=30)
    parser_.add_argument("--online_max_checkpoints_keep", type=int, default=3)

    args = parser_.parse_args()

    data_ids = parse_data_ids(args.data_ids)
    max_samples = args.max_samples if args.max_samples > 0 else None
    subset = args.subset if args.subset else None

    run_graphwiz_eval(
        budget=args.budget,
        lm_name=args.lm_name,
        source=args.source,
        subset=subset,
        max_samples=max_samples,
        data_ids=data_ids,
        local_json_path=args.local_json_path if args.local_json_path else None,
        data_root=args.data_root,
        prefer_local=bool(args.prefer_local),
        use_cache=bool(args.use_cache),
        export_parquet=bool(args.export_parquet),
        alpha_tokens=args.alpha_tokens,
        online_dpo=bool(args.online_dpo),
        online_pref_backend=args.online_pref_backend,
        online_pref_db_path=args.online_pref_db_path,
        online_batch_size=args.online_batch_size,
        online_update_interval_steps=args.online_update_interval_steps,
        online_min_node_gap=args.online_min_node_gap,
        online_hot_swap=bool(args.online_hot_swap),
        online_poll_interval_sec=args.online_poll_interval_sec,
        online_dpo_output_root=args.online_dpo_output_root,
        online_dpo_beta=args.online_dpo_beta,
        online_dpo_lr=args.online_dpo_lr,
        online_dpo_epochs=args.online_dpo_epochs,
        online_dpo_max_steps=args.online_dpo_max_steps,
        online_max_checkpoints_keep=args.online_max_checkpoints_keep,
    )