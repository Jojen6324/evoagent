"""Incremental JSONL logging with concise terminal progress."""
from datetime import datetime, timezone
import json
import os
import time
from typing import Dict


class EvaluationProgressReporter:
    def __init__(self, jsonl_path: str, echo: bool = True):
        self.jsonl_path = os.path.abspath(jsonl_path)
        self.echo = bool(echo)
        self._started: Dict[str, float] = {}
        os.makedirs(os.path.dirname(self.jsonl_path), exist_ok=True)
        with open(self.jsonl_path, "w", encoding="utf-8", newline="\n"):
            pass

    def __call__(self, event: dict) -> None:
        item = {
            "timestamp": datetime.now(timezone.utc).isoformat(), **dict(event),
        }
        with open(self.jsonl_path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        line = self._console_line(item)
        if self.echo and line:
            print(line, flush=True)

    @staticmethod
    def _duration(value: float) -> str:
        seconds = max(0, int(round(value)))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return "%dh%02dm%02ds" % (hours, minutes, seconds)
        if minutes:
            return "%dm%02ds" % (minutes, seconds)
        return "%ds" % seconds

    def _console_line(self, item: dict) -> str:
        event = item.get("event")
        name = str(item.get("name", "evaluation"))
        if event == "run_started":
            self._started[name] = time.monotonic()
            return "[START] %s | %d cases" % (name, item.get("total_cases", 0))
        if event == "case_completed":
            result = item.get("result") or {}
            index = int(item.get("index", 0) or 0)
            total = int(item.get("total_cases", 0) or 0)
            elapsed = time.monotonic() - self._started.get(name, time.monotonic())
            remaining = max(0, total - index)
            eta = elapsed / index * remaining if index else 0
            status = "PASS" if result.get("execution_success") else "FAIL"
            error = str(result.get("error") or "")
            return (
                "[%3d/%d %5.1f%%] %-6s %-18s | TP/FP/FN %s/%s/%s | "
                "LLM %s | Token %s | elapsed %s | ETA %s%s"
            ) % (
                index, total, (100 * index / total) if total else 0,
                status, str(item.get("case_id", ""))[:18],
                result.get("tp", 0), result.get("fp", 0), result.get("fn", 0),
                result.get("llm_calls", 0), result.get("total_tokens", 0),
                self._duration(elapsed), self._duration(eta),
                (" | " + error[:120]) if error else "",
            )
        if event == "run_completed":
            metrics = item.get("metrics") or {}
            self._started.pop(name, None)
            return "[DONE] %s | F1 %s | failure %s | duration %s" % (
                name, metrics.get("f1"), metrics.get("failure_rate", 0),
                self._duration(float(item.get("duration_seconds", 0) or 0)),
            )
        return ""
