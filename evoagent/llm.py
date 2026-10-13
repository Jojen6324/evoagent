"""Small JSON LLM client with Chat Completions/Responses compatibility."""
import os
import json
import subprocess
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .telemetry import ExecutionLedger


class _EndpointError(RuntimeError):
    """An HTTP/transport error that may justify trying another API style."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class JsonChatClient:
    def __init__(
        self, base_url: str, api_key: str, model: str,
        provider: str = "openai-compatible", timeout: int = 60,
        extra_headers: Optional[Dict[str, str]] = None,
        api_style: Optional[str] = None,
        chat_response_format: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        configured_style = api_style or os.getenv("EVOAGENT_LLM_API_STYLE", "auto")
        aliases = {
            "chat-completions": "chat", "chat_completion": "chat",
            "response": "responses", "response-api": "responses",
        }
        self.api_style = aliases.get(
            str(configured_style).strip().lower(),
        ) or str(configured_style).strip().lower()
        if self.api_style not in {"auto", "chat", "responses"}:
            raise ValueError(
                "EVOAGENT_LLM_API_STYLE must be auto, chat or responses"
            )
        configured_format = chat_response_format or os.getenv(
            "EVOAGENT_LLM_CHAT_RESPONSE_FORMAT",
            # Some Chat-only proxies reject response_format.  An explicit
            # chat client therefore retries without it; the auto mode keeps
            # its existing Chat -> Responses fallback semantics by default.
            "auto" if self.api_style == "chat" else "json_object",
        )
        format_aliases = {
            "json": "json_object", "json-object": "json_object",
            "off": "none", "omit": "none", "disabled": "none",
        }
        self.chat_response_format = format_aliases.get(
            str(configured_format).strip().lower(),
        ) or str(configured_format).strip().lower()
        if self.chat_response_format not in {"auto", "json_object", "none"}:
            raise ValueError(
                "EVOAGENT_LLM_CHAT_RESPONSE_FORMAT must be auto, "
                "json_object or none"
            )

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            # A few OpenAI-compatible gateways reject urllib's default
            # Python-urllib user agent/connection negotiation.  Use ordinary
            # client headers while keeping the stdlib transport dependency-free.
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "User-Agent": "EvoAgent/1.0",
        }
        headers.update(self.extra_headers)
        return headers

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(), method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("LLM response root is not an object")
            return body
        except urllib.error.HTTPError as exc:
            detail = exc.read(2000).decode("utf-8", errors="replace")
            raise _EndpointError(
                "%s API returned HTTP %d: %s"
                % (self.provider, exc.code, detail), exc.code,
            ) from exc
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            raise _EndpointError(
                "%s request failed: %s" % (self.provider, exc),
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise _EndpointError(
                "%s returned invalid JSON: %s" % (self.provider, exc),
            ) from exc

    @staticmethod
    def _parse_json_text(content: Any) -> Dict[str, Any]:
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", ""))
                if isinstance(item, dict) else str(item)
                for item in content
            )
        if isinstance(content, dict):
            result = content
        else:
            text = str(content or "").strip()
            if text.startswith("```") and text.endswith("```"):
                text = text[3:-3].strip()
                if text.lower().startswith("json"):
                    text = text[4:].lstrip()
            result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError("model JSON root is not an object")
        return result

    def _complete_chat_request(
        self, system: str, user: str, max_tokens: Optional[int],
        include_response_format: bool,
    ):
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if include_response_format:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        body = self._post_json("/chat/completions", payload)
        message = body["choices"][0]["message"]
        content = message.get("content", "") if isinstance(message, dict) else message
        return self._parse_json_text(content), body.get("usage") or {}

    def _complete_chat(
        self, system: str, user: str, max_tokens: Optional[int],
    ):
        mode = self.chat_response_format
        if mode == "none":
            return self._complete_chat_request(
                system, user, max_tokens, include_response_format=False,
            )
        if mode == "json_object":
            return self._complete_chat_request(
                system, user, max_tokens, include_response_format=True,
            )
        try:
            return self._complete_chat_request(
                system, user, max_tokens, include_response_format=True,
            )
        except _EndpointError as format_error:
            if not self._can_fallback(format_error.status_code):
                raise
            # A few OpenAI-compatible gateways reject response_format while
            # still accepting ordinary Chat Completions.  The prompts already
            # require JSON, so retrying without the optional hint preserves
            # compatibility and still uses the same parser and gate.
            return self._complete_chat_request(
                system, user, max_tokens, include_response_format=False,
            )

    @staticmethod
    def _responses_text(body: Dict[str, Any]) -> str:
        direct = body.get("output_text")
        if direct:
            return str(direct)
        for item in body.get("output") or []:
            if not isinstance(item, dict):
                continue
            for block in item.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in {"output_text", "text"} and block.get("text"):
                    return str(block["text"])
        raise ValueError("Responses API returned no output text")

    def _complete_responses(
        self, system: str, user: str, max_tokens: Optional[int],
    ):
        payload: Dict[str, Any] = {
            "model": self.model,
            # Some OpenAI-compatible Responses proxies accept the portable
            # string form but reject the richer message-array/text-format
            # shape. Keep the two logical messages separated and let the
            # existing system instruction enforce JSON-only output.
            "input": (
                "[SYSTEM INSTRUCTIONS]\n" + system
                + "\n\n[USER INPUT]\n" + user
            ),
        }
        if max_tokens:
            payload["max_output_tokens"] = int(max_tokens)
        body = self._post_json("/responses", payload)
        return self._parse_json_text(self._responses_text(body)), body.get("usage") or {}

    @staticmethod
    def _can_fallback(status_code: Optional[int]) -> bool:
        # These codes commonly mean that the provider exposes another API
        # shape. Auth failures are excluded so a bad credential is not hidden.
        return status_code in {400, 404, 405, 415, 422, 500, 501, 502, 503, 504}

    def complete_json(
        self, role: str, system: str, user: str,
        ledger: Optional[ExecutionLedger] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        started = time.monotonic()
        try:
            if self.api_style == "responses":
                result, usage = self._complete_responses(system, user, max_tokens)
            elif self.api_style == "chat":
                result, usage = self._complete_chat(system, user, max_tokens)
            else:
                try:
                    result, usage = self._complete_chat(system, user, max_tokens)
                except _EndpointError as chat_error:
                    if not self._can_fallback(chat_error.status_code):
                        raise
                    try:
                        result, usage = self._complete_responses(
                            system, user, max_tokens,
                        )
                    except Exception as response_error:
                        raise RuntimeError(
                            "%s; Responses fallback failed: %s"
                            % (chat_error, response_error)
                        ) from response_error
            if ledger:
                ledger.record_model(
                    role, self.provider, self.model, usage,
                    int((time.monotonic() - started) * 1000), True,
                )
            return result
        except Exception as exc:
            message = str(exc)
            if not isinstance(exc, (_EndpointError, RuntimeError)):
                message = "%s JSON request failed: %s" % (self.provider, exc)
        if ledger:
            ledger.record_model(
                role, self.provider, self.model, {},
                int((time.monotonic() - started) * 1000), False, message,
            )
        raise RuntimeError(message)


class CodexCliClient:
    """Use the locally authenticated Codex CLI as a JSON decision backend.

    Every call is ephemeral and runs in an empty, read-only workspace.  This
    prevents the evaluator implementation and earlier cases from leaking into
    the model context while preserving the ``JsonChatClient`` interface used
    by the agent runtime.
    """

    def __init__(
        self, model: str = "", executable: str = "codex", timeout: int = 120,
        reasoning_effort: str = "", extra_args: Optional[List[str]] = None,
        progress: bool = True,
    ):
        self.model = model or "configured-default"
        self.executable = executable
        self.timeout = int(timeout)
        self.reasoning_effort = reasoning_effort.strip().lower()
        self.extra_args = list(extra_args or [])
        self.progress = bool(progress)
        self.provider = "codex-cli"
        self._progress_lock = threading.Lock()
        self._call_count = 0

    def _progress(self, call_id: int, role: str, status: str, **fields: Any) -> None:
        if not self.progress:
            return
        details = " ".join("%s=%s" % item for item in fields.items())
        line = "[codex-cli] call=%d role=%s status=%s" % (call_id, role, status)
        if details:
            line += " " + details
        with self._progress_lock:
            print(line, file=os.sys.stderr, flush=True)

    def _command(self, output_path: str, workdir: str) -> List[str]:
        command = [
            self.executable, "exec", "--ephemeral", "--ignore-rules",
            "--skip-git-repo-check", "--sandbox", "read-only",
            "--color", "never", "--cd", workdir,
            "--output-last-message", output_path,
        ]
        if self.model != "configured-default":
            command.extend(["--model", self.model])
        if self.reasoning_effort:
            command.extend([
                "--config", "model_reasoning_effort=%s" % self.reasoning_effort,
            ])
        command.extend(self.extra_args)
        command.append("-")
        return command

    def complete_json(
        self, role: str, system: str, user: str,
        ledger: Optional[ExecutionLedger] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        del max_tokens  # The outer runtime enforces the shared token budget.
        started = time.monotonic()
        with self._progress_lock:
            self._call_count += 1
            call_id = self._call_count
        self._progress(call_id, role, "started", model=self.model)
        prompt = (
            "You are the %s component inside a deterministic evaluation harness.\n"
            "Follow the system instructions below. Return exactly one JSON object "
            "and no Markdown or commentary.\n\n[SYSTEM INSTRUCTIONS]\n%s"
            "\n\n[USER INPUT]\n%s"
        ) % (role, system, user)
        try:
            with tempfile.TemporaryDirectory(prefix="evoagent-codex-") as workdir:
                output_path = os.path.join(workdir, "last-message.json")
                completed = subprocess.run(
                    self._command(output_path, workdir), input=prompt,
                    text=True, encoding="utf-8", errors="replace",
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    timeout=self.timeout, check=False,
                )
                if completed.returncode != 0:
                    detail = (completed.stderr or completed.stdout or "").strip()
                    raise RuntimeError(
                        "codex exec exited with %d: %s"
                        % (completed.returncode, detail[-2000:])
                    )
                if not os.path.exists(output_path):
                    raise RuntimeError("codex exec did not write its final response")
                with open(output_path, "r", encoding="utf-8") as handle:
                    result = JsonChatClient._parse_json_text(handle.read())
            if ledger:
                ledger.record_model(
                    role, self.provider, self.model, {},
                    int((time.monotonic() - started) * 1000), True,
                )
            self._progress(
                call_id, role, "completed",
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
            return result
        except Exception as exc:
            message = "codex-cli JSON request failed: %s" % exc
            if ledger:
                ledger.record_model(
                    role, self.provider, self.model, {},
                    int((time.monotonic() - started) * 1000), False, message,
                )
            self._progress(
                call_id, role, "failed",
                elapsed_ms=int((time.monotonic() - started) * 1000),
                error=str(exc).replace("\n", " ")[:240],
            )
            raise RuntimeError(message) from exc
