import io
import json
import os
import unittest
from subprocess import CompletedProcess
from unittest.mock import patch
from urllib.error import HTTPError

from evoagent.llm import CodexCliClient, JsonChatClient
from evoagent.telemetry import ExecutionLedger


class _Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class ResponsesAdapterTests(unittest.TestCase):
    def test_responses_style_converts_output_and_usage(self):
        response = _Response({
            "output": [{
                "content": [{"type": "output_text", "text": '{"action":"final"}'}],
            }],
            "usage": {"input_tokens": 12, "output_tokens": 4},
        })
        ledger = ExecutionLedger("test")
        with patch("evoagent.llm.urllib.request.urlopen", return_value=response) as call:
            result = JsonChatClient(
                "https://example.test/v1", "secret", "grok-4.6",
                api_style="responses",
            ).complete_json("lead", "system", "user", ledger=ledger, max_tokens=16)

        self.assertEqual({"action": "final"}, result)
        request = call.call_args.args[0]
        self.assertTrue(request.full_url.endswith("/responses"))
        payload = json.loads(request.data.decode("utf-8"))
        self.assertIsInstance(payload["input"], str)
        self.assertNotIn("text", payload)
        summary = ledger.summary()
        self.assertEqual(1, summary["llm_calls"])
        self.assertEqual(16, summary["total_tokens"])

    def test_auto_falls_back_to_responses_after_endpoint_error(self):
        chat_error = HTTPError(
            "https://example.test/v1/chat/completions", 502,
            "Bad Gateway", {}, io.BytesIO(b"upstream access forbidden"),
        )
        response = _Response({
            "output_text": '{"action":"final"}',
            "usage": {"input_tokens": 3, "output_tokens": 2},
        })
        with patch(
            "evoagent.llm.urllib.request.urlopen",
            side_effect=[chat_error, response],
        ) as call:
            result = JsonChatClient(
                "https://example.test/v1", "secret", "grok-4.6",
                api_style="auto",
            ).complete_json("lead", "system", "user")

        self.assertEqual({"action": "final"}, result)
        self.assertTrue(call.call_args_list[0].args[0].full_url.endswith("/chat/completions"))
        self.assertTrue(call.call_args_list[1].args[0].full_url.endswith("/responses"))

    def test_chat_auto_retries_without_response_format(self):
        chat_error = HTTPError(
            "https://example.test/v1/chat/completions", 502,
            "Bad Gateway", {}, io.BytesIO(b"response_format unsupported"),
        )
        response = _Response({
            "choices": [{"message": {"content": '{"action":"final"}'}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        })
        with patch(
            "evoagent.llm.urllib.request.urlopen",
            side_effect=[chat_error, response],
        ) as call:
            result = JsonChatClient(
                "https://example.test/v1", "secret", "gpt-5.6-sol",
                api_style="chat", chat_response_format="auto",
            ).complete_json("lead", "system", "user")

        self.assertEqual({"action": "final"}, result)
        first_payload = json.loads(call.call_args_list[0].args[0].data.decode("utf-8"))
        second_payload = json.loads(call.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertIn("response_format", first_payload)
        self.assertNotIn("response_format", second_payload)


class CodexCliAdapterTests(unittest.TestCase):
    def test_codex_cli_uses_ephemeral_read_only_isolated_run(self):
        def fake_run(command, **kwargs):
            output_path = command[command.index("--output-last-message") + 1]
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write('{"action":"final","findings":[]}')
            self.assertIn("--ephemeral", command)
            self.assertIn("read-only", command)
            self.assertIn("--skip-git-repo-check", command)
            self.assertIn("Return exactly one JSON object", kwargs["input"])
            self.assertTrue(os.path.isdir(command[command.index("--cd") + 1]))
            return CompletedProcess(command, 0, "", "")

        ledger = ExecutionLedger("codex-test")
        with patch("evoagent.llm.subprocess.run", side_effect=fake_run):
            result = CodexCliClient(
                model="test-model", timeout=10, reasoning_effort="low",
                progress=False,
            ).complete_json("security", "system", "user", ledger=ledger)

        self.assertEqual({"action": "final", "findings": []}, result)
        self.assertEqual(1, ledger.summary()["llm_calls"])

    def test_codex_cli_records_failed_invocation(self):
        failed = CompletedProcess(["codex"], 2, "", "bad config")
        ledger = ExecutionLedger("codex-test")
        with patch("evoagent.llm.subprocess.run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "bad config"):
                CodexCliClient(progress=False).complete_json(
                    "lead", "system", "user", ledger=ledger,
                )
        self.assertEqual(1, ledger.summary()["failed_model_calls"])


if __name__ == "__main__":
    unittest.main()
