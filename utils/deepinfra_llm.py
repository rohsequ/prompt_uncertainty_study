from collections.abc import Sequence
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionMessageParam,
)
from agentdojo.agent_pipeline.llms.openai_llm import (
    OpenAILLM,
    _message_to_openai as original_message_to_openai,
    _function_to_openai,
    chat_completion_request,
    _openai_to_assistant_message,
)
from agentdojo.functions_runtime import FunctionsRuntime, EmptyEnv, Env
from agentdojo.types import ChatMessage


def _content_blocks_to_openai_content_blocks(message):
    if message["content"] is None:
        return None
    from openai.types.chat import ChatCompletionContentPartTextParam

    return [
        ChatCompletionContentPartTextParam(type="text", text=el["content"] or "")
        for el in message["content"]
    ]


def _deepinfra_message_to_openai(
    message: ChatMessage, model_name: str
) -> ChatCompletionMessageParam:
    if message["role"] == "system":
        return ChatCompletionSystemMessageParam(
            role="system",
            content=_content_blocks_to_openai_content_blocks(message),
        )
    return original_message_to_openai(message, model_name)


class DeepInfraLLM(OpenAILLM):
    """
    A custom AgentDojo Pipeline Element that properly maps 'system' roles natively
    instead of defaulting to 'developer', which is required for DeepInfra Llama APIs.
    """

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict | None = None,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if extra_args is None:
            extra_args = {}

        openai_messages = [
            _deepinfra_message_to_openai(message, self.model) for message in messages
        ]
        openai_tools = [
            _function_to_openai(tool) for tool in runtime.functions.values()
        ]
        completion = chat_completion_request(
            self.client,
            self.model,
            openai_messages,
            openai_tools,
            self.reasoning_effort,
            self.temperature,
        )
        output = _openai_to_assistant_message(completion.choices[0].message)

        # Fallback patch: DeepInfra occasionally fails to translate Llama-3 XML tool syntax
        # back into standard OpenAI tool_calls arrays, leaking `<function=...>` into text
        if not output.get("tool_calls", []):
            content_text = ""
            for block in output.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    content_text += block.get("content", "")

            if "<function=" in content_text:
                from agentdojo.agent_pipeline.llms.local_llm import _parse_model_output

                parsed_output = _parse_model_output(content_text)
                if parsed_output.get("tool_calls"):
                    output["tool_calls"] = parsed_output["tool_calls"]

        messages_out = list(messages) + [output]
        return query, runtime, env, messages_out, extra_args
