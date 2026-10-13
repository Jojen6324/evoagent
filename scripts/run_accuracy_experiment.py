"""Run experiment one: controlled Harness validation and optional real-PR accuracy."""
import argparse
import json
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evoagent.evaluation_experiments import AccuracyExperimentSuite  # noqa: E402
from evoagent.evaluation_harness import load_jsonl  # noqa: E402
from evoagent.evaluation_progress import EvaluationProgressReporter  # noqa: E402
from evoagent.evaluation_v2 import ProductArmReviewer  # noqa: E402
from evoagent.llm import CodexCliClient, JsonChatClient  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate code-review accuracy on the 100-case controlled set and real PRs."
    )
    parser.add_argument("--real-dataset", default="")
    parser.add_argument("--minimum-real-cases", type=int, default=300)
    parser.add_argument(
        "--dataset-split", choices=("all", "validation", "holdout"),
        default="all", help="Optionally evaluate only one named dataset split.",
    )
    parser.add_argument(
        "--case-limit", type=int, default=0,
        help="Limit real cases for a smoke run; zero evaluates every selected case.",
    )
    parser.add_argument("--base-url", default=os.getenv("EVOAGENT_LLM_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("EVOAGENT_LLM_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("EVOAGENT_LLM_MODEL", ""))
    parser.add_argument("--provider", default=os.getenv("EVOAGENT_LLM_PROVIDER", "custom"))
    parser.add_argument(
        "--llm-backend", choices=("http", "codex-cli"), default="http",
        help="Use an HTTP-compatible model API or the locally authenticated Codex CLI.",
    )
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument("--codex-reasoning-effort", default="")
    parser.add_argument(
        "--no-model-progress", action="store_true",
        help="Suppress per-role Codex CLI start/completion lines.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--token-budget", type=int, default=12000)
    parser.add_argument("--time-budget", type=int, default=240)
    parser.add_argument(
        "--worker-execution", choices=("serial", "parallel"), default="serial",
        help="Run specialist workers sequentially or concurrently (default: serial).",
    )
    parser.add_argument("--allow-non-production-data", action="store_true")
    parser.add_argument(
        "--output", default=os.path.join(
            ROOT, "output", "accuracy-experiment", "evaluation.json",
        ),
    )
    parser.add_argument("--progress-jsonl", default="")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    output = os.path.abspath(args.output)
    stem, _extension = os.path.splitext(output)
    progress = None if args.no_progress else EvaluationProgressReporter(
        args.progress_jsonl or stem + ".progress.jsonl",
    )

    real_cases = real_reviewer = None
    if args.real_dataset:
        if args.llm_backend == "http" and (
            not args.base_url or not args.api_key or not args.model
        ):
            parser.error(
                "--base-url, --api-key and --model are required with --real-dataset"
            )
        if args.llm_backend == "codex-cli":
            client = CodexCliClient(
                model=args.model, executable=args.codex_executable,
                timeout=args.timeout,
                reasoning_effort=args.codex_reasoning_effort,
                progress=not args.no_model_progress,
            )
        else:
            client = JsonChatClient(
                args.base_url, args.api_key, args.model,
                provider=args.provider, timeout=args.timeout,
            )
        real_reviewer = ProductArmReviewer(
            "full-agentic", client, args.token_budget, args.time_budget,
            worker_execution=args.worker_execution,
        )
        real_cases = load_jsonl(args.real_dataset)
        if args.dataset_split != "all":
            real_cases = [
                case for case in real_cases
                if str(case.get("split", "")).lower() == args.dataset_split
            ]
        if args.case_limit > 0:
            real_cases = real_cases[:args.case_limit]

    report = AccuracyExperimentSuite(args.minimum_real_cases).run(
        real_cases=real_cases, real_reviewer=real_reviewer,
        require_production_ready=not args.allow_non_production_data,
        progress_callback=progress,
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(output)
    if progress is not None:
        print(progress.jsonl_path)


if __name__ == "__main__":
    main()
