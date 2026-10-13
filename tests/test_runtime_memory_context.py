import os
import tempfile
import unittest
import json

from evoagent.agentic_core import BoundedRole
from evoagent.context_manager import ContextManager
from evoagent.memory import MemoryManager
from evoagent.runtime import (
    AgentRuntime, AgentTool, RuntimeNode, ToolProtocolError, ToolRegistry,
)
from evoagent.store import TaskStore, utc_now
from evoagent.telemetry import ExecutionLedger


class RuntimeMemoryTests(unittest.TestCase):
    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        self.store = TaskStore(self.path)

    def tearDown(self):
        os.unlink(self.path)

    def test_runtime_restores_completed_node_checkpoints(self):
        self.store.create("runtime-task", "org/repo", 1, {})
        runtime = AgentRuntime(max_steps=4, timeout_seconds=5)
        calls = []
        nodes = [
            RuntimeNode("plan", lambda _state: calls.append("plan") or {"value": 2}),
            RuntimeNode(
                "execute",
                lambda state: calls.append("execute") or {"result": state["value"] * 3},
            ),
        ]

        first = runtime.execute({}, nodes, "runtime-task", self.store)
        second = runtime.execute({}, nodes, "runtime-task", self.store)

        self.assertEqual(6, first["result"])
        self.assertEqual(6, second["result"])
        self.assertEqual(["plan", "execute"], calls)

    def test_tool_registry_validates_arguments_before_invocation(self):
        calls = []
        registry = ToolRegistry([AgentTool(
            "lookup", "Lookup one key.",
            {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"], "additionalProperties": False,
            },
            lambda key: calls.append(key) or "value:%s" % key,
        )])

        with self.assertRaisesRegex(ToolProtocolError, "unknown tool arguments"):
            registry.invoke("lookup", {"key": 7, "unexpected": True})

        self.assertEqual([], calls)
        self.assertEqual("value:x", registry.invoke("lookup", {"key": "x"}))
        self.assertEqual(["x"], calls)

    def test_memory_recall_is_repository_and_tenant_isolated(self):
        memory = MemoryManager(self.store, recall_limit=5)
        memory.remember(
            "tenant-a", "org/repo", "semantic", "review_feedback",
            "SEC-EVAL was a confirmed missed issue in authentication code",
            importance=0.9,
        )
        memory.remember(
            "tenant-b", "org/repo", "semantic", "review_feedback",
            "REL-DEBUG-PRINT was accepted", importance=0.9,
        )

        recalled = memory.recall(
            "tenant-a", "org/repo", "authentication SEC-EVAL security"
        )

        self.assertEqual(1, len(recalled))
        self.assertIn("SEC-EVAL", recalled[0]["content"])
        self.assertEqual([], memory.recall("tenant-a", "org/other", "SEC-EVAL"))

    def test_memory_recall_purges_expired_records(self):
        self.store.save_agent_memory({
            "id": "expired-memory", "tenant_id": "tenant-a",
            "repository": "org/repo", "task_id": "old-task", "agent": "agent",
            "scope": "working", "kind": "observation", "content": "expired content",
            "keywords": ["expired"], "metadata": {}, "importance": 0.5,
            "created_at": utc_now(), "expires_at": "2000-01-01T00:00:00+00:00",
        })

        MemoryManager(self.store).recall("tenant-a", "org/repo", "expired")

        self.assertEqual([], self.store.list_agent_memories(
            "tenant-a", "org/repo", ("working",), 10
        ))

    def test_tool_observation_is_written_and_visible_to_next_agent_step(self):
        memory = MemoryManager(self.store, recall_limit=8)
        seen = []

        class ToolThenFinalClient:
            provider = "fake"
            model = "fake"

            def complete_json(_self, _role, _prompt, user, ledger=None, max_tokens=None):
                managed = json.loads(user)
                task = json.loads(managed["task"])
                seen.append(task.get("working_memory"))
                if ledger:
                    ledger.record_model("security", "fake", "fake", {
                        "prompt_tokens": 10, "completion_tokens": 3,
                    }, 1)
                if len(seen) == 1:
                    return {"action": "tool", "tool": "lookup", "arguments": {"key": "auth"}}
                return {"action": "final", "findings": []}

        def supplier():
            values = memory.recall_working("tenant-a", "org/repo", "task")
            return ContextManager.format_memories(values) if values else None

        def sink(agent, observation):
            memory.remember_observation(
                "tenant-a", "org/repo", "task", agent, observation,
            )

        role = BoundedRole(
            "security", "Use facts.", ToolThenFinalClient(), 1000, 10,
            context_manager=ContextManager(), working_memory_supplier=supplier,
            observation_sink=sink,
        )
        registry = ToolRegistry([AgentTool(
            "lookup", "Lookup a fact.", {
                "type": "object", "properties": {"key": {"type": "string"}},
                "required": ["key"], "additionalProperties": False,
            }, lambda key: {"evidence_id": "lookup:%s" % key, "output": {"value": "allowed"}},
        )])

        role.run(json.dumps({"phase": "worker"}), registry, ExecutionLedger("agentic"))

        self.assertIsNone(seen[0])
        self.assertEqual("tool_observation", seen[1]["items"][0]["kind"])
        self.assertIn("lookup:auth", seen[1]["items"][0]["content"])

    def test_role_recovers_missing_action_for_expected_phase_payload(self):
        class MissingActionClient:
            provider = "fake"
            model = "fake"

            def complete_json(_self, role, _prompt, _user, ledger=None, max_tokens=None):
                if ledger:
                    ledger.record_model(role, "fake", "fake", {
                        "prompt_tokens": 5, "completion_tokens": 3,
                    }, 1)
                return {"findings": []}

        ledger = ExecutionLedger("agentic")
        result = BoundedRole(
            "security", "Return JSON.", MissingActionClient(), 1000, 10,
            max_steps=1, expected_final_fields=("findings",),
        ).run("{}", ToolRegistry(), ledger)

        self.assertEqual("final", result["action"])
        self.assertEqual(
            1,
            len([
                item for item in ledger.summary()["agent_traces"]["security"]
                if item["event"] == "protocol_recovered"
            ]),
        )

    def test_role_gives_invalid_action_one_correction_attempt(self):
        class CorrectingClient:
            provider = "fake"
            model = "fake"

            def __init__(self):
                self.calls = 0

            def complete_json(_self, role, _prompt, user, ledger=None, max_tokens=None):
                _self.calls += 1
                if ledger:
                    ledger.record_model(role, "fake", "fake", {
                        "prompt_tokens": 5, "completion_tokens": 3,
                    }, 1)
                if _self.calls == 1:
                    return {"status": "done"}
                managed = json.loads(user)
                self.assertIn("protocol_feedback", managed)
                return {"action": "final", "findings": []}

        client = CorrectingClient()
        result = BoundedRole(
            "security", "Return JSON.", client, 1000, 10,
            max_steps=1, expected_final_fields=("findings",),
        ).run("{}", ToolRegistry(), ExecutionLedger("agentic"))

        self.assertEqual("final", result["action"])
        self.assertEqual(2, client.calls)

    def test_task_consolidation_releases_working_memory_but_keeps_episode(self):
        memory = MemoryManager(self.store)
        memory.remember_observation("tenant-a", "org/repo", "task", "security", {
            "step": 1, "tool": "symbol", "ok": True,
            "result": {"evidence_id": "symbol:auth", "output": {"callers": ["api"]}},
        })

        memory.consolidate_task("tenant-a", "org/repo", "task", {"accepted_findings": []})

        self.assertEqual([], memory.recall_working("tenant-a", "org/repo", "task"))
        archived = memory.recall("tenant-a", "org/repo", "completed task")
        self.assertEqual("task_summary", archived[0]["kind"])

    def test_working_memory_is_isolated_by_agent_role(self):
        memory = MemoryManager(self.store)
        memory.remember_observation("tenant-a", "org/repo", "task", "security", {
            "step": 1, "tool": "lookup", "ok": True,
            "result": {"evidence_id": "security:auth", "output": "security fact"},
        })
        memory.remember_observation("tenant-a", "org/repo", "task", "correctness-reliability", {
            "step": 1, "tool": "lookup", "ok": True,
            "result": {"evidence_id": "correctness:auth", "output": "correctness fact"},
        })

        security = memory.recall_working(
            "tenant-a", "org/repo", "task", agent="security",
        )
        critic = memory.recall_working(
            "tenant-a", "org/repo", "task", agent="critic",
        )

        self.assertEqual(1, len(security))
        self.assertIn("security:auth", security[0]["content"])
        self.assertEqual([], critic)


if __name__ == "__main__":
    unittest.main()
