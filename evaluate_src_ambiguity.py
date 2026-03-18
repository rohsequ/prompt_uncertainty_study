import argparse
import ast
import copy
import csv
import datetime as dt
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import agentdojo.attacks  # noqa: F401  # Ensure attacks are registered
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import is_string_list, tool_result_to_str
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.task_suite.task_suite import TaskSuite
from agentdojo.types import ChatMessage, text_content_block_from_string
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from src.agents import SimpleAgent

logging.basicConfig(level=logging.INFO)

BENCHMARK_VERSION = "v1.2.2"


def _serialize_langchain_message(message: BaseMessage) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump()
    if hasattr(message, "dict"):
        return message.dict()
    return {"type": getattr(message, "type", "unknown"), "content": str(message.content)}


def _flatten_ground_truth_calls(function_calls: Sequence[FunctionCall]) -> list[FunctionCall]:
    flattened: list[FunctionCall] = []

    def visit(call: FunctionCall) -> None:
        flattened.append(call)
        for arg in call.args.values():
            if isinstance(arg, FunctionCall):
                visit(arg)

    for function_call in function_calls:
        visit(function_call)
    return flattened


def _task_required_tools(task: BaseUserTask | BaseInjectionTask, environment: Env) -> set[str]:
    return {
        call.function
        for call in _flatten_ground_truth_calls(task.ground_truth(environment.model_copy(deep=True)))
    }


def _task_model_output(messages: Sequence[ChatMessage]) -> list[dict[str, str]] | list[Any]:
    if not messages:
        return []
    last_message = messages[-1]
    if last_message["role"] != "assistant":
        return []
    return last_message.get("content") or []


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


@dataclass
class AgentRunArtifacts:
    final_output: str
    messages: list[ChatMessage]
    tool_trace: list[FunctionCall]
    langchain_messages: list[dict[str, Any]]
    iterations: int


class SrcSimpleAgentPipeline(BasePipelineElement):
    """
    AgentDojo-compatible pipeline that delegates planning/tool choice to `src`'s
    `SimpleAgent`, while executing tools against a real AgentDojo environment.

    This leaves the existing simulated tool-response implementation untouched and
    only introduces a new execution backend for evaluation.
    """

    def __init__(self, agent_type: str, config_path: str, max_iters: int = 15) -> None:
        self.agent_type = agent_type
        self.config_path = config_path
        self.max_iters = max_iters
        self.simple_agent = SimpleAgent(agent_type, config_path=config_path)
        model_name = self.simple_agent.config.get("simple_agents", "model_name", "unknown")
        self.name = f"src-simple-agent::{agent_type}::{model_name}"
        self.available_tools = {
            spec["name"] for spec in self.simple_agent.registry.get_tool_specs(agent_type)
        }

    def _normalize_tool_calls(self, tool_calls: Any) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index, tool_call in enumerate(tool_calls or []):
            if isinstance(tool_call, dict):
                name = tool_call.get("name")
                args = tool_call.get("args", {})
                tool_id = tool_call.get("id") or f"tool_call_{index}"
            else:
                name = getattr(tool_call, "name", None)
                args = getattr(tool_call, "args", {})
                tool_id = getattr(tool_call, "id", None) or f"tool_call_{index}"

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            if not isinstance(args, dict):
                args = {}

            normalized.append({"name": name, "args": args, "id": tool_id})
        return normalized

    def _normalize_runtime_args(self, args: dict[str, Any]) -> dict[str, Any]:
        normalized_args = dict(args)
        for key, value in normalized_args.items():
            if isinstance(value, str) and is_string_list(value):
                normalized_args[key] = ast.literal_eval(value)
        return normalized_args

    def _assistant_message(
        self,
        content: str | None,
        tool_calls: list[FunctionCall],
    ) -> ChatMessage:
        if tool_calls and not content:
            content_blocks = None
        else:
            content_blocks = [text_content_block_from_string(content or "")]
        return {
            "role": "assistant",
            "content": content_blocks,
            "tool_calls": tool_calls,
        }

    def _tool_message(
        self,
        tool_call: FunctionCall,
        content: str,
        error: str | None,
    ) -> ChatMessage:
        return {
            "role": "tool",
            "content": [text_content_block_from_string(content)],
            "tool_call": tool_call,
            "tool_call_id": tool_call.id,
            "error": error,
        }

    def run_with_artifacts(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
    ) -> AgentRunArtifacts:
        system_prompt = self.simple_agent.load_system_prompt()
        langchain_messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]
        agentdojo_messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": [text_content_block_from_string(system_prompt)],
            },
            {
                "role": "user",
                "content": [text_content_block_from_string(query)],
            },
        ]
        tool_trace: list[FunctionCall] = []

        for iteration in range(1, self.max_iters + 1):
            ai_response = self.simple_agent.agent_model.invoke(langchain_messages)
            langchain_messages.append(ai_response)

            normalized_tool_calls = self._normalize_tool_calls(getattr(ai_response, "tool_calls", []))
            function_calls = [
                FunctionCall(
                    function=tool_call["name"],
                    args=self._normalize_runtime_args(tool_call["args"]),
                    id=tool_call["id"],
                )
                for tool_call in normalized_tool_calls
                if tool_call["name"] is not None
            ]
            tool_trace.extend(function_calls)

            agentdojo_messages.append(
                self._assistant_message(str(ai_response.content or ""), function_calls)
            )

            if not function_calls:
                final_output = str(ai_response.content or "")
                return AgentRunArtifacts(
                    final_output=final_output,
                    messages=agentdojo_messages,
                    tool_trace=tool_trace,
                    langchain_messages=[
                        _serialize_langchain_message(message) for message in langchain_messages
                    ],
                    iterations=iteration,
                )

            for function_call in function_calls:
                tool_result_text = ""
                tool_error: str | None = None

                if function_call.function not in runtime.functions:
                    tool_error = (
                        f"ToolNotFoundError: The requested function `{function_call.function}` is not available."
                    )
                    langchain_tool_content = tool_error
                else:
                    tool_result, runtime_error = runtime.run_function(
                        env,
                        function_call.function,
                        function_call.args,
                    )
                    tool_error = runtime_error
                    tool_result_text = tool_result_to_str(tool_result)
                    langchain_tool_content = tool_error or tool_result_text

                langchain_messages.append(
                    ToolMessage(
                        content=langchain_tool_content,
                        tool_call_id=function_call.id or function_call.function,
                        name=function_call.function,
                    )
                )
                agentdojo_messages.append(
                    self._tool_message(function_call, tool_result_text, tool_error)
                )

        max_iter_message = "Maximum tool iterations reached before the agent produced a final answer."
        agentdojo_messages.append(self._assistant_message(max_iter_message, []))
        langchain_messages.append(AIMessage(content=max_iter_message, tool_calls=[]))
        return AgentRunArtifacts(
            final_output=max_iter_message,
            messages=agentdojo_messages,
            tool_trace=tool_trace,
            langchain_messages=[
                _serialize_langchain_message(message) for message in langchain_messages
            ],
            iterations=self.max_iters,
        )

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict | None = None,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if messages:
            raise ValueError(
                "SrcSimpleAgentPipeline only supports fresh task execution and does not accept pre-populated messages."
            )
        if extra_args is None:
            extra_args = {}

        run = self.run_with_artifacts(query, runtime, env)
        extra_args = {
            **extra_args,
            "final_output": run.final_output,
            "langchain_messages": run.langchain_messages,
            "tool_trace": [call.model_dump() for call in run.tool_trace],
            "iterations": run.iterations,
        }
        return query, runtime, env, run.messages, extra_args


@dataclass
class EvaluationRecord:
    mode: str
    task_id: str
    ambiguity_level: str
    run_index: int
    prompt: str
    utility: bool
    security: bool
    injection_task_id: str
    attack_name: str
    injections: dict[str, str]
    iterations: int
    final_output: str
    tool_trace: list[dict[str, Any]]
    langchain_messages: list[dict[str, Any]]
    agentdojo_messages: list[ChatMessage]
    error: str | None = None
    skip_reason: str | None = None


def _resolve_prompt_files(prompts_dir: Path, agent: str) -> list[Path]:
    agent_prompts_dir = prompts_dir / agent
    if agent_prompts_dir.exists():
        files = sorted(agent_prompts_dir.glob(f"{agent}_*.json"))
    else:
        files = sorted(prompts_dir.glob(f"{agent}_*.json"))
    return files


def _extract_task_number(value: str) -> int | None:
    match = re.search(r"user_task_(\d+)", value)
    if not match:
        return None
    return int(match.group(1))


def _filter_prompt_files_by_user_tasks(
    prompt_files: Sequence[Path],
    user_tasks: Sequence[int] | None,
) -> list[Path]:
    if not user_tasks:
        return list(prompt_files)

    requested = set(user_tasks)
    filtered: list[Path] = []

    for prompt_file in prompt_files:
        task_number = _extract_task_number(prompt_file.stem)
        if task_number is None:
            try:
                prompt_data = _load_variations(prompt_file)
            except Exception:
                continue
            task_number = _extract_task_number(str(prompt_data.get("task_id", "")))

        if task_number in requested:
            filtered.append(prompt_file)

    return filtered


def _supported_task(task: BaseUserTask | BaseInjectionTask, suite: TaskSuite, pipeline: SrcSimpleAgentPipeline) -> tuple[bool, str | None]:
    probe_environment = suite.load_and_inject_default_environment({})
    required_tools = _task_required_tools(task, probe_environment)
    missing_tools = sorted(required_tools - pipeline.available_tools)
    if missing_tools:
        return False, f"Unsupported by src agent toolset: {', '.join(missing_tools)}"
    return True, None


def _evaluate_run(
    suite: TaskSuite,
    pipeline: SrcSimpleAgentPipeline,
    user_task: BaseUserTask,
    prompt: str,
    injections: dict[str, str],
    injection_task: BaseInjectionTask | None,
) -> EvaluationRecord:
    environment = suite.load_and_inject_default_environment(injections)
    task_environment = user_task.init_environment(environment)
    pre_environment = task_environment.model_copy(deep=True)
    runtime = FunctionsRuntime(suite.tools)

    run = pipeline.run_with_artifacts(prompt, runtime, task_environment)
    model_output = _task_model_output(run.messages)

    utility = suite._check_task_result(
        user_task,
        model_output,
        pre_environment,
        task_environment,
        run.tool_trace,
    )
    security = True
    injection_task_id = "none"

    if injection_task is not None:
        injection_task_id = injection_task.ID
        security = suite._check_task_result(
            injection_task,
            model_output,
            pre_environment,
            task_environment,
            run.tool_trace,
        )

    return EvaluationRecord(
        mode="injection" if injection_task is not None else "utility",
        task_id=user_task.ID,
        ambiguity_level="",
        run_index=0,
        prompt=prompt,
        utility=utility,
        security=security,
        injection_task_id=injection_task_id,
        attack_name="none",
        injections=injections,
        iterations=run.iterations,
        final_output=run.final_output,
        tool_trace=[call.model_dump() for call in run.tool_trace],
        langchain_messages=run.langchain_messages,
        agentdojo_messages=list(run.messages),
    )


def _write_trace(trace_dir: Path, record: EvaluationRecord) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"{record.task_id}_{record.ambiguity_level}_run{record.run_index}_{record.injection_task_id}.json"
    )
    path = trace_dir / filename
    with path.open("w") as handle:
        json.dump(_json_safe(asdict(record)), handle, indent=2)


def _write_csv(csv_path: Path, records: Sequence[EvaluationRecord]) -> None:
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Mode",
                "Task_ID",
                "Ambiguity_Level",
                "Run_Index",
                "Injection_Task_ID",
                "Attack_Name",
                "Utility",
                "Security",
                "Iterations",
                "Error",
                "Skip_Reason",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.mode,
                    record.task_id,
                    record.ambiguity_level,
                    record.run_index,
                    record.injection_task_id,
                    record.attack_name,
                    int(record.utility),
                    int(record.security),
                    record.iterations,
                    record.error or "",
                    record.skip_reason or "",
                ]
            )


def _load_variations(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _build_task_copy(task: BaseUserTask, task_id: str, prompt: str) -> BaseUserTask:
    task_copy = copy.deepcopy(task)
    task_copy.ID = task_id
    task_copy.PROMPT = prompt
    return task_copy


def _resolve_config_path(agent: str, config_path: str | None) -> str:
    if config_path is not None:
        return config_path
    candidate = Path("src") / "configs" / f"{agent}_config.ini"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(
        f"Could not infer config for agent '{agent}'. Pass --config explicitly."
    )


def _evaluate_prompt_variations(
    suite: TaskSuite,
    pipeline: SrcSimpleAgentPipeline,
    prompt_files: Sequence[Path],
    output_dir: Path,
    n_runs: int,
    mode: str,
    attack_name: str,
    n_injection_tasks: int,
) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    attack = load_attack(attack_name, suite, pipeline) if mode in {"injection", "both"} else None
    injection_items = list(suite.injection_tasks.items())
    if n_injection_tasks > 0:
        injection_items = injection_items[:n_injection_tasks]

    for prompt_file in prompt_files:
        prompt_data = _load_variations(prompt_file)
        task_id = prompt_data["task_id"]
        if task_id not in suite.user_tasks:
            logging.warning("Skipping prompt file %s because task %s is not in suite %s", prompt_file, task_id, suite.name)
            continue

        base_task = suite.user_tasks[task_id]
        supported, skip_reason = _supported_task(base_task, suite, pipeline)
        if not supported:
            logging.warning("Skipping task %s: %s", task_id, skip_reason)
            for level in sorted(prompt_data.get("variations", {})):
                for run_index in range(n_runs):
                    if mode in {"utility", "both"}:
                        records.append(
                            EvaluationRecord(
                                mode="utility",
                                task_id=task_id,
                                ambiguity_level=level,
                                run_index=run_index,
                                prompt=prompt_data["variations"][level],
                                utility=False,
                                security=True,
                                injection_task_id="none",
                                attack_name="none",
                                injections={},
                                iterations=0,
                                final_output="",
                                tool_trace=[],
                                langchain_messages=[],
                                agentdojo_messages=[],
                                skip_reason=skip_reason,
                            )
                        )
                    if mode in {"injection", "both"}:
                        for injection_task_id, _ in injection_items:
                            records.append(
                                EvaluationRecord(
                                    mode="injection",
                                    task_id=task_id,
                                    ambiguity_level=level,
                                    run_index=run_index,
                                    prompt=prompt_data["variations"][level],
                                    utility=False,
                                    security=False,
                                    injection_task_id=injection_task_id,
                                    attack_name=attack_name,
                                    injections={},
                                    iterations=0,
                                    final_output="",
                                    tool_trace=[],
                                    langchain_messages=[],
                                    agentdojo_messages=[],
                                    skip_reason=skip_reason,
                                )
                            )
            continue

        variations = prompt_data.get("variations", {})
        for level, new_prompt in sorted(variations.items()):
            logging.info("Evaluating %s at %s", task_id, level)
            for run_index in range(n_runs):
                utility_task = _build_task_copy(
                    base_task,
                    task_id=f"{task_id}_{level}_run{run_index}",
                    prompt=new_prompt,
                )

                if mode in {"utility", "both"}:
                    try:
                        utility_record = _evaluate_run(
                            suite=suite,
                            pipeline=pipeline,
                            user_task=utility_task,
                            prompt=new_prompt,
                            injections={},
                            injection_task=None,
                        )
                        utility_record.ambiguity_level = level
                        utility_record.run_index = run_index
                    except Exception as exc:
                        utility_record = EvaluationRecord(
                            mode="utility",
                            task_id=utility_task.ID,
                            ambiguity_level=level,
                            run_index=run_index,
                            prompt=new_prompt,
                            utility=False,
                            security=True,
                            injection_task_id="none",
                            attack_name="none",
                            injections={},
                            iterations=0,
                            final_output="",
                            tool_trace=[],
                            langchain_messages=[],
                            agentdojo_messages=[],
                            error=str(exc),
                        )
                    records.append(utility_record)
                    _write_trace(output_dir / "traces" / "utility", utility_record)

                if mode not in {"injection", "both"}:
                    continue

                assert attack is not None
                for injection_task_id, injection_task in injection_items:
                    supported_injection, injection_skip = _supported_task(
                        injection_task, suite, pipeline
                    )
                    if not supported_injection:
                        records.append(
                            EvaluationRecord(
                                mode="injection",
                                task_id=utility_task.ID,
                                ambiguity_level=level,
                                run_index=run_index,
                                prompt=new_prompt,
                                utility=False,
                                security=False,
                                injection_task_id=injection_task_id,
                                attack_name=attack_name,
                                injections={},
                                iterations=0,
                                final_output="",
                                tool_trace=[],
                                langchain_messages=[],
                                agentdojo_messages=[],
                                skip_reason=injection_skip,
                            )
                        )
                        continue

                    injections = attack.attack(utility_task, injection_task)
                    try:
                        injection_record = _evaluate_run(
                            suite=suite,
                            pipeline=pipeline,
                            user_task=utility_task,
                            prompt=new_prompt,
                            injections=injections,
                            injection_task=injection_task,
                        )
                        injection_record.ambiguity_level = level
                        injection_record.run_index = run_index
                        injection_record.attack_name = attack_name
                    except Exception as exc:
                        injection_record = EvaluationRecord(
                            mode="injection",
                            task_id=utility_task.ID,
                            ambiguity_level=level,
                            run_index=run_index,
                            prompt=new_prompt,
                            utility=False,
                            security=False,
                            injection_task_id=injection_task_id,
                            attack_name=attack_name,
                            injections=injections,
                            iterations=0,
                            final_output="",
                            tool_trace=[],
                            langchain_messages=[],
                            agentdojo_messages=[],
                            error=str(exc),
                        )
                    records.append(injection_record)
                    _write_trace(output_dir / "traces" / "injection", injection_record)

    return records


def _write_summary(output_dir: Path, records: Sequence[EvaluationRecord]) -> None:
    summary_path = output_dir / "aggregate_results.csv"
    grouped: dict[tuple[str, str], dict[str, int]] = {}

    for record in records:
        key = (record.mode, record.ambiguity_level)
        if key not in grouped:
            grouped[key] = {
                "runs": 0,
                "utility_success": 0,
                "security_success": 0,
            }
        grouped[key]["runs"] += 1
        grouped[key]["utility_success"] += int(record.utility)
        grouped[key]["security_success"] += int(record.security)

    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["Mode", "Ambiguity_Level", "Runs", "Utility_Successes", "Security_Successes"]
        )
        for (mode, level), metrics in sorted(grouped.items()):
            writer.writerow(
                [
                    mode,
                    level,
                    metrics["runs"],
                    metrics["utility_success"],
                    metrics["security_success"],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ambiguity prompt variations by running the `src` SimpleAgent "
            "against real AgentDojo environments, then score with AgentDojo task evaluation."
        )
    )
    parser.add_argument("--agent", type=str, default="workspace", help="Agent suite to evaluate.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file for the src SimpleAgent. Defaults to src/configs/<agent>_config.ini.",
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=Path("ambiguity_prompts"),
        help="Directory containing prompt-variation JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs_src_eval"),
        help="Output directory for CSV summaries and per-run traces.",
    )
    parser.add_argument(
        "--mode",
        choices=["utility", "injection", "both"],
        default="both",
        help="Whether to run utility-only evaluation, injection evaluation, or both.",
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="direct",
        help="AgentDojo attack name to use when --mode includes injections.",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=0,
        help="Number of prompt files to evaluate. Use 0 for all available tasks.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of repeated runs per ambiguity level.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=50,
        help="Maximum tool-call iterations allowed for the src SimpleAgent runner.",
    )
    parser.add_argument(
        "--n-injection-tasks",
        type=int,
        default=3,
        help="Number of injection tasks to evaluate when --mode includes injections. Use 0 for all.",
    )
    parser.add_argument(
        "--user-tasks",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional list of user task numbers to evaluate, for example "
            "--user-tasks 0 3 12 will only test user_task_0, user_task_3, and user_task_12."
        ),
    )
    args = parser.parse_args()

    config_path = _resolve_config_path(args.agent, args.config)
    suite = get_suite(BENCHMARK_VERSION, args.agent)
    pipeline = SrcSimpleAgentPipeline(args.agent, config_path=config_path, max_iters=args.max_iters)

    prompt_files = _resolve_prompt_files(args.prompts_dir, args.agent)
    if not prompt_files:
        raise FileNotFoundError(
            f"No prompt variation JSON files found for agent '{args.agent}' under {args.prompts_dir}."
        )
    prompt_files = _filter_prompt_files_by_user_tasks(prompt_files, args.user_tasks)
    if args.user_tasks and not prompt_files:
        requested_tasks = ", ".join(str(task) for task in args.user_tasks)
        raise FileNotFoundError(
            f"No prompt variation JSON files found for agent '{args.agent}' matching user tasks [{requested_tasks}] under {args.prompts_dir}."
        )
    if args.n_tasks > 0:
        prompt_files = prompt_files[: args.n_tasks]

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"results_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Using config: %s", config_path)
    logging.info("Pipeline name: %s", pipeline.name)
    logging.info("Found %d prompt variation files", len(prompt_files))
    if args.user_tasks:
        logging.info("Filtering to user task numbers: %s", ", ".join(map(str, args.user_tasks)))
    logging.info(
        "Suite tool coverage: %d src tools exposed, %d AgentDojo suite tools available",
        len(pipeline.available_tools),
        len(suite.tools),
    )

    records = _evaluate_prompt_variations(
        suite=suite,
        pipeline=pipeline,
        prompt_files=prompt_files,
        output_dir=output_dir,
        n_runs=args.n_runs,
        mode=args.mode,
        attack_name=args.attack,
        n_injection_tasks=args.n_injection_tasks,
    )

    _write_csv(output_dir / "results.csv", records)
    _write_summary(output_dir, records)

    metadata = {
        "agent": args.agent,
        "config_path": config_path,
        "mode": args.mode,
        "attack": args.attack,
        "n_tasks": args.n_tasks,
        "n_runs": args.n_runs,
        "n_injection_tasks": args.n_injection_tasks,
        "pipeline_name": pipeline.name,
        "benchmark_version": BENCHMARK_VERSION,
        "timestamp": timestamp,
    }
    with (output_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    logging.info("Saved results to %s", output_dir)


if __name__ == "__main__":
    main()
