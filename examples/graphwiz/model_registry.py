import json
import os
import time
from typing import Any, Dict, Optional


class ModelRegistry:
    """
    File-based active model registry.
    """

    def __init__(self, path: str, initial_checkpoint: str = "") -> None:
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            self._write_state(
                {
                    "active_checkpoint": str(initial_checkpoint or ""),
                    "version": 0,
                    "updated_at": time.time(),
                    "meta": {},
                }
            )

    def _read_state(self) -> Dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {
            "active_checkpoint": "",
            "version": 0,
            "updated_at": 0.0,
            "meta": {},
        }

    def _write_state(self, state: Dict[str, Any]) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def get_state(self) -> Dict[str, Any]:
        return self._read_state()

    def get_active_checkpoint(self) -> str:
        return str(self._read_state().get("active_checkpoint", "") or "")

    def set_active_checkpoint(self, checkpoint_path: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        prev = self._read_state()
        state = {
            "active_checkpoint": str(checkpoint_path or ""),
            "version": int(prev.get("version", 0)) + 1,
            "updated_at": time.time(),
            "meta": meta or {},
        }
        self._write_state(state)
        return state
