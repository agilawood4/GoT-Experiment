import datetime
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

try:
    from .model_registry import ModelRegistry
    from .online_pref_store import OnlinePreferenceStore
except Exception:
    from model_registry import ModelRegistry
    from online_pref_store import OnlinePreferenceStore


class AsyncDPOWorker:
    def __init__(
        self,
        pref_store: OnlinePreferenceStore,
        model_registry: ModelRegistry,
        base_model_name_or_path: str,
        output_root: str,
        batch_size: int = 128,
        min_score_gap: float = 0.05,
        poll_interval_sec: float = 5.0,
        max_checkpoints_keep: int = 3,
        dpo_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.pref_store = pref_store
        self.model_registry = model_registry
        self.base_model_name_or_path = str(base_model_name_or_path)
        self.output_root = os.path.abspath(output_root)
        self.batch_size = int(batch_size)
        self.min_score_gap = float(min_score_gap)
        self.poll_interval_sec = float(poll_interval_sec)
        self.max_checkpoints_keep = int(max_checkpoints_keep)
        self.dpo_kwargs = dpo_kwargs or {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._checkpoints: List[str] = []
        os.makedirs(self.output_root, exist_ok=True)

    def _to_pref_records(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "instruction": r.get("prompt", ""),
                    "input": "",
                    "chosen": r.get("chosen", ""),
                    "rejected": r.get("rejected", ""),
                    "metadata": {
                        "node_group_key": r.get("node_group_key", ""),
                        "score_gap": float(r.get("score_gap", 0.0)),
                        "task": r.get("task", ""),
                        "part": r.get("part", ""),
                        "phase": r.get("phase", ""),
                        "sample_id": r.get("sample_id", ""),
                    },
                }
            )
        return out

    def _cleanup_old(self) -> None:
        if self.max_checkpoints_keep <= 0:
            return
        while len(self._checkpoints) > self.max_checkpoints_keep:
            old = self._checkpoints.pop(0)
            try:
                if os.path.isdir(old):
                    import shutil

                    shutil.rmtree(old, ignore_errors=True)
            except Exception:
                pass

    def _run_once(self) -> bool:
        try:
            if __package__:
                from .train_dpo_trl import run_dpo_once
            else:
                from train_dpo_trl import run_dpo_once
        except Exception as e:
            raise RuntimeError(
                "Online DPO worker failed to import train_dpo_trl dependencies. "
                "Please align transformers/peft/trl versions."
            ) from e

        rows = self.pref_store.fetch_pending(
            limit=self.batch_size,
            min_score_gap=self.min_score_gap,
        )
        if len(rows) < self.batch_size:
            return False
        pref_records = self._to_pref_records(rows)
        base_model = self.model_registry.get_active_checkpoint() or self.base_model_name_or_path
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(self.output_root, f"online_dpo_{ts}_{len(pref_records)}")
        os.makedirs(out_dir, exist_ok=True)

        checkpoint_dir = run_dpo_once(
            model_name_or_path=base_model,
            pref_records=pref_records,
            output_dir=out_dir,
            **self.dpo_kwargs,
        )

        ids = [int(x["id"]) for x in rows]
        batch_tag = os.path.basename(checkpoint_dir)
        self.pref_store.mark_consumed(ids, batch_tag=batch_tag)
        self.model_registry.set_active_checkpoint(
            checkpoint_dir,
            meta={
                "batch_size": len(pref_records),
                "batch_tag": batch_tag,
                "source_model": base_model,
            },
        )
        self._checkpoints.append(checkpoint_dir)
        self._cleanup_old()
        logging.info("[online_dpo_worker] updated checkpoint=%s", checkpoint_dir)
        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                changed = self._run_once()
                if not changed:
                    time.sleep(self.poll_interval_sec)
            except Exception as e:
                logging.exception("[online_dpo_worker] loop error: %s", e)
                time.sleep(self.poll_interval_sec)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, join_timeout_sec: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=join_timeout_sec)
