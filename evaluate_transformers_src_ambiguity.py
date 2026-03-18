import argparse
import ast
import copy
import csv
import datetime as dt
import json
import logging
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
AGENTDOJO_SRC_ROOT = PROJECT_ROOT / "agentdojo" / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(AGENTDOJO_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENTDOJO_SRC_ROOT))

import agentdojo.attacks  # noqa: F401  # Ensure attacks are registered
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.tool_execution import is_string_list, tool_result_to_str
from agentdojo.ast_utils import (
    ASTParsingError,
    parse_tool_calls_from_python_function,
)
from agentdojo.attacks.attack_registry import load_attack
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.task_suite.task_suite import TaskSuite
from agentdojo.types import ChatMessage, text_content_block_from_string
from src.agents import get_agent_registry
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)

BENCHMARK_VERSION = "v1.2.2"


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


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype | str:
    normalized = dtype_name.lower()
    if normalized == "auto":
        return "auto"
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in dtype_map:
        raise ValueError(
            f"Unsupported torch dtype '{dtype_name}'. Choose from auto, float16, bfloat16, float32."
        )
    return dtype_map[normalized]


def _extract_task_number(value: str) -> int | None:
    match = re.search(r"user_task_(\d+)", value)
    if not match:
        return None
    return int(match.group(1))


def _resolve_prompt_files(prompts_dir: Path, agent: str) -> list[Path]:
    agent_prompts_dir = prompts_dir / agent
    if agent_prompts_dir.exists():
        files = sorted(agent_prompts_dir.glob(f"{agent}_*.json"))
    else:
        files = sorted(prompts_dir.glob(f"{agent}_*.json"))
    return files


def _load_variations(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


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


def _build_task_copy(task: BaseUserTask, task_id: str, prompt: str) -> BaseUserTask:
    task_copy = copy.deepcopy(task)
    task_copy.ID = task_id
    task_copy.PROMPT = prompt
    return task_copy


def _base_user_task_id(task_id: str) -> str:
    task_number = _extract_task_number(task_id)
    if task_number is not None:
        return f"user_task_{task_number}"
    return task_id


@dataclass
class GenerationStepArtifact:
    iteration: int
    messages: list[dict[str, Any]]
    prompt_text: str
    raw_completion: str
    parsed_content: str
    parsed_tool_calls: list[dict[str, Any]]
    parser_used: str
    input_token_count: int
    generated_token_count: int
    token_scores: list[dict[str, Any]]
    hidden_state_summary: dict[str, Any] | None
    attention_summary: dict[str, Any] | None


@dataclass
class AgentRunArtifacts:
    final_output: str
    messages: list[ChatMessage]
    tool_trace: list[FunctionCall]
    generation_steps: list[GenerationStepArtifact]
    iterations: int


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
    generation_steps: list[dict[str, Any]]
    agentdojo_messages: list[ChatMessage]
    error: str | None = None
    skip_reason: str | None = None


class TransformersSimpleAgentPipeline(BasePipelineElement):
    """
    AgentDojo-compatible pipeline that reuses the `src` agent definitions, but runs
    the planner with a direct `transformers` model instead of LangChain.

    The tool-calling protocol is explicit and parseable, which keeps the execution
    loop under our control and makes it easier to add hidden-state/attention hooks later.
    """

    def __init__(
        self,
        agent_type: str,
        model_path: str,
        max_iters: int = 15,
        max_new_tokens: int = 512,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        enable_thinking: bool | None = None,
        capture_scores: bool = False,
        capture_hidden_states: bool = False,
        capture_attentions: bool = False,
        score_top_k: int = 5,
    ) -> None:
        self.agent_type = agent_type
        self.model_path = model_path
        self.max_iters = max_iters
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.capture_scores = capture_scores
        self.capture_hidden_states = capture_hidden_states
        self.capture_attentions = capture_attentions
        self.score_top_k = score_top_k

        self.registry = get_agent_registry()
        self.system_prompt = self.registry.get_system_prompt(agent_type)
        self.tool_specs = self.registry.get_tool_specs(agent_type)
        self.tool_schemas = [
            {
                "name": tool_spec["name"],
                "description": tool_spec["description"],
                "parameters": tool_spec["parameters"],
                **({"returns": tool_spec["returns"]} if "returns" in tool_spec else {}),
            }
            for tool_spec in self.tool_specs
        ]
        self.available_tools = {spec["name"] for spec in self.tool_specs}
        self.name = f"transformers-simple-agent::{agent_type}::{Path(model_path).name}"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        if not hasattr(self.tokenizer, "apply_chat_template") or self.tokenizer.chat_template is None:
            raise ValueError(
                "This tokenizer does not expose a chat template. Built-in transformers tool calling "
                "requires `tokenizer.apply_chat_template(...)` support."
            )
        added_pad_token = False
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
                added_pad_token = True

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=_resolve_torch_dtype(torch_dtype),
            trust_remote_code=trust_remote_code,
        )
        if added_pad_token:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        self.input_device = next(self.model.parameters()).device
        self._validate_tool_chat_template_support()

    def _normalize_runtime_args(self, args: dict[str, Any]) -> dict[str, Any]:
        normalized_args = dict(args)
        for key, value in normalized_args.items():
            if isinstance(value, str) and is_string_list(value):
                normalized_args[key] = ast.literal_eval(value)
        return normalized_args

    def _assistant_message(
        self,
        content: str,
        tool_calls: list[FunctionCall],
    ) -> ChatMessage:
        content_blocks = [text_content_block_from_string(content)] if content else None
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

    def _validate_tool_chat_template_support(self) -> None:
        probe_messages = [{"role": "user", "content": "ping"}]
        try:
            rendered_without_tools = self.tokenizer.apply_chat_template(
                probe_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            rendered_with_tools = self.tokenizer.apply_chat_template(
                probe_messages,
                tools=self.tool_schemas[:1],
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception as exc:
            raise ValueError(
                "Failed to apply the tokenizer chat template with tools. This model/tokenizer pair "
                "does not appear to support built-in transformers tool calling."
            ) from exc

        if rendered_with_tools == rendered_without_tools:
            raise ValueError(
                "The tokenizer chat template did not change when tool schemas were provided. "
                "This usually means the model's template does not support built-in tool calling."
            )

    def _extract_answer_text(self, content: str) -> str:
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()

        cleaned = re.sub(r"<function-thoughts>.*?</function-thoughts>", "", content, flags=re.DOTALL)
        cleaned = re.sub(r"<function-call>.*?</function-call>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", cleaned, flags=re.DOTALL)
        cleaned = re.sub(r"</?answer>", "", cleaned)
        return cleaned.strip()

    def _coerce_tool_argument(self, raw_value: str) -> Any:
        value = raw_value.strip()
        if not value:
            return ""
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _tool_calls_from_xml_content(self, content: str) -> list[FunctionCall]:
        tool_calls: list[FunctionCall] = []
        tool_call_blocks = re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
        if not tool_call_blocks:
            return []

        for index, block in enumerate(tool_call_blocks):
            function_match = re.search(r"<function\s*=\s*([^>]+)>(.*?)</function>", block, re.DOTALL)
            if not function_match:
                continue

            function_name = function_match.group(1).strip()
            function_body = function_match.group(2)
            arguments: dict[str, Any] = {}

            for parameter_name, raw_value in re.findall(
                r"<parameter\s*=\s*([^>]+)>\s*(.*?)\s*</parameter>",
                function_body,
                re.DOTALL,
            ):
                arguments[parameter_name.strip()] = self._coerce_tool_argument(raw_value)

            tool_calls.append(
                FunctionCall(
                    function=function_name,
                    args=self._normalize_runtime_args(arguments),
                    id=f"tool_call_{index}",
                )
            )

        return tool_calls

    def _make_chat_template_tool_call(self, tool_call: FunctionCall) -> dict[str, Any]:
        return {
            "type": "function",
            "id": tool_call.id or tool_call.function,
            "function": {
                "name": tool_call.function,
                "arguments": tool_call.args,
            },
        }

    def _normalize_native_tool_call(
        self,
        payload: dict[str, Any],
        index: int,
    ) -> FunctionCall | None:
        if "function" in payload and isinstance(payload["function"], dict):
            function_payload = payload["function"]
            name = function_payload.get("name")
            args = function_payload.get("arguments", {})
            tool_id = payload.get("id") or f"tool_call_{index}"
        else:
            name = payload.get("name")
            args = payload.get("arguments", payload.get("args", payload.get("parameters", {})))
            tool_id = payload.get("id") or f"tool_call_{index}"

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        if name is None or not isinstance(args, dict):
            return None

        return FunctionCall(
            function=name,
            args=self._normalize_runtime_args(args),
            id=tool_id,
        )

    def _tool_calls_from_json_payload(self, payload: Any) -> list[FunctionCall]:
        if isinstance(payload, dict):
            if "tool_calls" in payload and isinstance(payload["tool_calls"], list):
                calls = [
                    self._normalize_native_tool_call(item, index)
                    for index, item in enumerate(payload["tool_calls"])
                ]
                return [call for call in calls if call is not None]

            call = self._normalize_native_tool_call(payload, 0)
            return [call] if call is not None else []

        if isinstance(payload, list):
            calls = [
                self._normalize_native_tool_call(item, index)
                for index, item in enumerate(payload)
                if isinstance(item, dict)
            ]
            return [call for call in calls if call is not None]

        return []

    def _extract_json_candidates(self, content: str) -> list[str]:
        candidates: list[str] = []
        fenced_matches = re.findall(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        candidates.extend(match.strip() for match in fenced_matches if match.strip())

        tag_matches = re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
        candidates.extend(match.strip() for match in tag_matches if match.strip())

        object_matches = re.findall(r"(\{[\s\S]*\})", content)
        array_matches = re.findall(r"(\[[\s\S]*\])", content)
        candidates.extend(object_matches)
        candidates.extend(array_matches)
        return candidates

    def _parse_assistant_output(self, content: str) -> tuple[str, list[FunctionCall], str]:
        if hasattr(self.tokenizer, "parse_response") and getattr(self.tokenizer, "response_schema", None) is not None:
            try:
                parsed_response = self.tokenizer.parse_response(content)
                native_calls = self._tool_calls_from_json_payload(parsed_response)
                if native_calls:
                    final_text = ""
                    if isinstance(parsed_response, dict):
                        final_text = str(parsed_response.get("content", "") or "")
                    return final_text, native_calls, "tokenizer.parse_response"
                if isinstance(parsed_response, dict) and parsed_response.get("content"):
                    return str(parsed_response["content"]).strip(), [], "tokenizer.parse_response"
            except Exception:
                pass

        for candidate in self._extract_json_candidates(content):
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            json_calls = self._tool_calls_from_json_payload(payload)
            if json_calls:
                return "", json_calls, "json_fallback"

        xml_calls = self._tool_calls_from_xml_content(content)
        if xml_calls:
            return self._extract_answer_text(content), xml_calls, "xml_tool_call_fallback"

        function_tag_match = re.search(
            r"<function\s*=\s*([^>]+)>(.*?)</function>",
            content,
            re.DOTALL,
        )
        if function_tag_match:
            function_name = function_tag_match.group(1).strip()
            raw_args = function_tag_match.group(2).strip()
            try:
                arguments = json.loads(raw_args) if raw_args else {}
                if isinstance(arguments, dict):
                    return (
                        "",
                        [
                            FunctionCall(
                                function=function_name,
                                args=self._normalize_runtime_args(arguments),
                                id="tool_call_0",
                            )
                        ],
                        "function_tag_fallback",
                    )
            except json.JSONDecodeError:
                pass

        tool_call_match = re.search(r"<function-call>(.*?)</function-call>", content, re.DOTALL)
        tool_call_content = tool_call_match.group(1).strip() if tool_call_match else "[]"

        try:
            tool_calls = parse_tool_calls_from_python_function(tool_call_content)
        except ASTParsingError:
            tool_calls = []

        normalized_calls = [
            FunctionCall(
                function=call.function,
                args=self._normalize_runtime_args(call.args),
                id=call.id,
            )
            for call in tool_calls
        ]
        final_text = self._extract_answer_text(content)
        return final_text, normalized_calls, "python_tag_fallback"

    def _tool_result_as_chat_message(
        self,
        tool_call: FunctionCall,
        tool_result_text: str,
        tool_error: str | None,
    ) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call.id or tool_call.function,
            "name": tool_call.function,
            "content": tool_error or tool_result_text or "Success",
        }

    def _extract_token_scores(
        self,
        generated_ids: torch.Tensor,
        scores: Sequence[torch.Tensor] | None,
    ) -> list[dict[str, Any]]:
        if not scores:
            return []

        token_scores: list[dict[str, Any]] = []
        for step_index, score_tensor in enumerate(scores):
            if step_index >= generated_ids.shape[0]:
                break

            log_probs = torch.log_softmax(score_tensor[0], dim=-1)
            token_id = int(generated_ids[step_index].item())
            token_logprob = float(log_probs[token_id].item())
            token_probability = float(math.exp(token_logprob))

            top_k = min(self.score_top_k, log_probs.shape[-1])
            top_log_probs, top_ids = torch.topk(log_probs, k=top_k)
            top_tokens = []
            for candidate_id, candidate_logprob in zip(top_ids.tolist(), top_log_probs.tolist()):
                top_tokens.append(
                    {
                        "token_id": int(candidate_id),
                        "token_text": self.tokenizer.decode([candidate_id]),
                        "logprob": float(candidate_logprob),
                        "probability": float(math.exp(candidate_logprob)),
                    }
                )

            token_scores.append(
                {
                    "position": step_index,
                    "token_id": token_id,
                    "token_text": self.tokenizer.decode([token_id]),
                    "logprob": token_logprob,
                    "probability": token_probability,
                    "top_tokens": top_tokens,
                }
            )

        return token_scores

    def _summary_from_nested_outputs(self, outputs: Any) -> dict[str, Any] | None:
        if not outputs:
            return None

        summary: dict[str, Any] = {"steps": len(outputs)}
        first_step = outputs[0]
        if isinstance(first_step, (list, tuple)):
            summary["layers_in_first_step"] = len(first_step)
            if first_step and hasattr(first_step[-1], "shape"):
                summary["last_tensor_shape"] = list(first_step[-1].shape)
        return summary

    def _generate(self, messages: Sequence[dict[str, Any]]) -> tuple[str, str, dict[str, Any]]:
        chat_template_kwargs: dict[str, Any] = {}
        if self.enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = self.enable_thinking

        prompt_text = self.tokenizer.apply_chat_template(
            list(messages),
            tools=self.tool_schemas,
            add_generation_prompt=True,
            tokenize=False,
            **chat_template_kwargs,
        )
        model_inputs = self.tokenizer.apply_chat_template(
            list(messages),
            tools=self.tool_schemas,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **chat_template_kwargs,
        )
        model_inputs = model_inputs.to(self.input_device)
        input_token_count = int(model_inputs["input_ids"].shape[-1])

        generation_kwargs = {
            **model_inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": self.capture_scores,
            "output_hidden_states": self.capture_hidden_states,
            "output_attentions": self.capture_attentions,
        }

        with torch.inference_mode():
            generation_output = self.model.generate(**generation_kwargs)

        generated_ids = generation_output.sequences[0][input_token_count:]
        raw_completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        metadata = {
            "prompt_text": prompt_text,
            "input_token_count": input_token_count,
            "generated_token_count": int(generated_ids.shape[0]),
            "token_scores": self._extract_token_scores(generated_ids, getattr(generation_output, "scores", None)),
            "hidden_state_summary": self._summary_from_nested_outputs(
                getattr(generation_output, "hidden_states", None)
            ),
            "attention_summary": self._summary_from_nested_outputs(
                getattr(generation_output, "attentions", None)
            ),
        }
        return prompt_text, raw_completion, metadata

    def run_with_artifacts(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
    ) -> AgentRunArtifacts:
        system_prompt = self.system_prompt or ""
        model_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        agentdojo_messages: list[ChatMessage] = [
            {"role": "system", "content": [text_content_block_from_string(system_prompt)]},
            {"role": "user", "content": [text_content_block_from_string(query)]},
        ]
        tool_trace: list[FunctionCall] = []
        generation_steps: list[GenerationStepArtifact] = []

        for iteration in range(1, self.max_iters + 1):
            prompt_text, raw_completion, generation_metadata = self._generate(model_messages)
            parsed_content, function_calls, parser_used = self._parse_assistant_output(raw_completion)
            tool_trace.extend(function_calls)

            generation_steps.append(
                GenerationStepArtifact(
                    iteration=iteration,
                    messages=[dict(message) for message in model_messages],
                    prompt_text=prompt_text,
                    raw_completion=raw_completion,
                    parsed_content=parsed_content,
                    parsed_tool_calls=[call.model_dump() for call in function_calls],
                    parser_used=parser_used,
                    input_token_count=generation_metadata["input_token_count"],
                    generated_token_count=generation_metadata["generated_token_count"],
                    token_scores=generation_metadata["token_scores"],
                    hidden_state_summary=generation_metadata["hidden_state_summary"],
                    attention_summary=generation_metadata["attention_summary"],
                )
            )

            assistant_content = parsed_content if function_calls else (parsed_content or raw_completion)
            assistant_chat_message: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_content,
            }
            if function_calls:
                assistant_chat_message["tool_calls"] = [
                    self._make_chat_template_tool_call(tool_call) for tool_call in function_calls
                ]
            model_messages.append(assistant_chat_message)
            agentdojo_messages.append(
                self._assistant_message(assistant_content, function_calls)
            )

            if not function_calls:
                final_output = assistant_content
                return AgentRunArtifacts(
                    final_output=final_output,
                    messages=agentdojo_messages,
                    tool_trace=tool_trace,
                    generation_steps=generation_steps,
                    iterations=iteration,
                )

            for function_call in function_calls:
                tool_result_text = ""
                tool_error: str | None = None

                if function_call.function not in runtime.functions:
                    tool_error = (
                        f"ToolNotFoundError: The requested function `{function_call.function}` is not available."
                    )
                else:
                    tool_result, runtime_error = runtime.run_function(
                        env,
                        function_call.function,
                        function_call.args,
                    )
                    tool_error = runtime_error
                    tool_result_text = tool_result_to_str(tool_result)

                model_messages.append(
                    self._tool_result_as_chat_message(function_call, tool_result_text, tool_error)
                )
                agentdojo_messages.append(
                    self._tool_message(function_call, tool_result_text, tool_error)
                )

        max_iter_message = "Maximum tool iterations reached before the agent produced a final answer."
        agentdojo_messages.append(self._assistant_message(max_iter_message, []))
        return AgentRunArtifacts(
            final_output=max_iter_message,
            messages=agentdojo_messages,
            tool_trace=tool_trace,
            generation_steps=generation_steps,
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
                "TransformersSimpleAgentPipeline only supports fresh task execution and does not accept pre-populated messages."
            )
        if extra_args is None:
            extra_args = {}

        run = self.run_with_artifacts(query, runtime, env)
        extra_args = {
            **extra_args,
            "final_output": run.final_output,
            "tool_trace": [call.model_dump() for call in run.tool_trace],
            "generation_steps": [_json_safe(asdict(step)) for step in run.generation_steps],
            "iterations": run.iterations,
        }
        return query, runtime, env, run.messages, extra_args


def _supported_task(
    task: BaseUserTask | BaseInjectionTask,
    suite: TaskSuite,
    pipeline: TransformersSimpleAgentPipeline,
) -> tuple[bool, str | None]:
    probe_environment = suite.load_and_inject_default_environment({})
    required_tools = _task_required_tools(task, probe_environment)
    missing_tools = sorted(required_tools - pipeline.available_tools)
    if missing_tools:
        return False, f"Unsupported by transformers agent toolset: {', '.join(missing_tools)}"
    return True, None


def _evaluate_run(
    suite: TaskSuite,
    pipeline: TransformersSimpleAgentPipeline,
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
        generation_steps=[_json_safe(asdict(step)) for step in run.generation_steps],
        agentdojo_messages=list(run.messages),
    )


def _write_trace(trace_dir: Path, record: EvaluationRecord) -> None:
    task_trace_dir = trace_dir / _base_user_task_id(record.task_id)
    task_trace_dir.mkdir(parents=True, exist_ok=True)
    if record.injection_task_id == "none":
        filename = f"{record.task_id}.json"
    else:
        filename = f"{record.task_id}_{record.injection_task_id}.json"
    path = task_trace_dir / filename
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


def _evaluate_prompt_variations(
    suite: TaskSuite,
    pipeline: TransformersSimpleAgentPipeline,
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
            logging.warning(
                "Skipping prompt file %s because task %s is not in suite %s",
                prompt_file,
                task_id,
                suite.name,
            )
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
                                generation_steps=[],
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
                                    generation_steps=[],
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
                            generation_steps=[],
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
                                generation_steps=[],
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
                            generation_steps=[],
                            agentdojo_messages=[],
                            error=str(exc),
                        )
                    records.append(injection_record)
                    _write_trace(output_dir / "traces" / "injection", injection_record)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ambiguity prompt variations by running a local transformers model "
            "with `src` system prompts and tool schemas against real AgentDojo environments."
        )
    )
    parser.add_argument("--agent", type=str, default="workspace", help="Agent suite to evaluate.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the local Hugging Face model directory.",
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
        default=Path("runs_transformers_src_eval"),
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
        "--user-tasks",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional list of user task numbers to evaluate, for example "
            "--user-tasks 0 3 12 will only test user_task_0, user_task_3, and user_task_12."
        ),
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
        help="Maximum tool-call iterations allowed for the transformers runner.",
    )
    parser.add_argument(
        "--n-injection-tasks",
        type=int,
        default=3,
        help="Number of injection tasks to evaluate when --mode includes injections. Use 0 for all.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate per assistant turn.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map passed to AutoModelForCausalLM.from_pretrained.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        help="Torch dtype for model loading: auto, float16, bfloat16, or float32.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Enable trust_remote_code when loading the tokenizer and model.",
    )
    parser.add_argument(
        "--enable-thinking",
        dest="enable_thinking",
        action="store_true",
        default=None,
        help="Request thinking mode through the tokenizer chat template when the model supports it.",
    )
    parser.add_argument(
        "--disable-thinking",
        dest="enable_thinking",
        action="store_false",
        help="Disable thinking mode through the tokenizer chat template when the model supports it.",
    )
    parser.add_argument(
        "--capture-scores",
        action="store_true",
        help="Capture generated-token probabilities and top-k alternatives for each step.",
    )
    parser.add_argument(
        "--capture-hidden-states",
        action="store_true",
        help="Request hidden states from generation and store lightweight summaries.",
    )
    parser.add_argument(
        "--capture-attentions",
        action="store_true",
        help="Request attentions from generation and store lightweight summaries.",
    )
    parser.add_argument(
        "--score-top-k",
        type=int,
        default=5,
        help="Number of top candidate tokens to store per generated position when --capture-scores is enabled.",
    )
    args = parser.parse_args()

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

    suite = get_suite(BENCHMARK_VERSION, args.agent)
    pipeline = TransformersSimpleAgentPipeline(
        agent_type=args.agent,
        model_path=args.model_path,
        max_iters=args.max_iters,
        max_new_tokens=args.max_new_tokens,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        enable_thinking=args.enable_thinking,
        capture_scores=args.capture_scores,
        capture_hidden_states=args.capture_hidden_states,
        capture_attentions=args.capture_attentions,
        score_top_k=args.score_top_k,
    )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / f"results_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Model path: %s", args.model_path)
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
        "model_path": args.model_path,
        "mode": args.mode,
        "attack": args.attack,
        "n_tasks": args.n_tasks,
        "user_tasks": args.user_tasks,
        "n_runs": args.n_runs,
        "n_injection_tasks": args.n_injection_tasks,
        "pipeline_name": pipeline.name,
        "benchmark_version": BENCHMARK_VERSION,
        "timestamp": timestamp,
        "max_new_tokens": args.max_new_tokens,
        "device_map": args.device_map,
        "torch_dtype": args.torch_dtype,
        "enable_thinking": args.enable_thinking,
        "capture_scores": args.capture_scores,
        "capture_hidden_states": args.capture_hidden_states,
        "capture_attentions": args.capture_attentions,
        "score_top_k": args.score_top_k,
    }
    with (output_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    logging.info("Saved results to %s", output_dir)


if __name__ == "__main__":
    main()
