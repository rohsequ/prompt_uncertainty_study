import json
import os
import inspect
from typing import Optional, Dict, Any, List, Callable
from pydantic import BaseModel
import openai


class LLMConfig(BaseModel):
    provider: str  # "openai", "ollama", "ollama-cloud", or "deepinfra"
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None


class ToolCallOutput(BaseModel):
    name: str
    arguments: dict


class LLMessageResponse(BaseModel):
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCallOutput]] = None


def function_to_openai_tool(func: Callable) -> dict:
    sig = inspect.signature(func)
    parameters = {"type": "object", "properties": {}, "required": []}
    for name, param in sig.parameters.items():
        parameters["properties"][name] = {"type": "string"}
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": parameters,
        },
    }


class BaseLLMProvider:
    """Base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config

    def invoke(
        self,
        messages: List[Dict[str, str]],
        response_schema: type[BaseModel] | None = None,
        tools: Optional[List[Any]] = None,
    ) -> LLMessageResponse:
        """
        Invokes the LLM with the given messages, optional response schema, and optional tools.
        Returns an LLMessageResponse containing textual content and any tool calls.
        """
        raise NotImplementedError("Subclasses must implement the invoke method.")


class OpenAILLMProvider(BaseLLMProvider):
    def invoke(
        self,
        messages: List[Dict[str, str]],
        response_schema: type[BaseModel] | None = None,
        tools: Optional[List[Any]] = None,
    ) -> LLMessageResponse:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return self._invoke_with_client(client, messages, response_schema, tools)

    def _invoke_with_client(
        self,
        client: openai.OpenAI,
        messages: List[Dict[str, str]],
        response_schema: type[BaseModel] | None = None,
        tools: Optional[List[Any]] = None,
    ) -> LLMessageResponse:
        kwargs = {}

        formatted_tools = []
        if tools:
            for t in tools:
                if callable(t):
                    formatted_tools.append(function_to_openai_tool(t))
                else:
                    formatted_tools.append(t)
            kwargs["tools"] = formatted_tools

        token_kwargs = {}
        if (
            self.config.model.startswith("o1")
            or self.config.model.startswith("o3")
            or "gpt-5" in self.config.model
        ):
            if self.config.max_tokens is not None:
                token_kwargs["max_completion_tokens"] = self.config.max_tokens
            token_kwargs["temperature"] = 1.0
        else:
            if self.config.max_tokens is not None:
                token_kwargs["max_tokens"] = self.config.max_tokens
            token_kwargs["temperature"] = self.config.temperature

        if response_schema:
            try:
                # Use strict native pydantic parsing if running a newer OpenAI version
                completion = client.beta.chat.completions.parse(
                    model=self.config.model,
                    messages=messages,
                    response_format=response_schema,
                    **token_kwargs,
                    **kwargs,
                )
            except AttributeError:
                # Fallback to standard json formatting
                kwargs["response_format"] = {"type": "json_object"}
                completion = client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    **token_kwargs,
                    **kwargs,
                )
        else:
            completion = client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                **token_kwargs,
                **kwargs,
            )

        message = completion.choices[0].message
        content = message.content

        t_calls = []
        if getattr(message, "tool_calls", None):
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                t_calls.append(ToolCallOutput(name=tc.function.name, arguments=args))

        return LLMessageResponse(content=content, tool_calls=t_calls)


class OllamaLLMProviderBase(BaseLLMProvider):
    def __init__(self, config: LLMConfig, host: str):
        super().__init__(config)
        self.host = host

    def invoke(
        self,
        messages: List[Dict[str, str]],
        response_schema: type[BaseModel] | None = None,
        tools: Optional[List[Any]] = None,
    ) -> LLMessageResponse:
        from ollama import Client

        api_key = (
            os.environ.get("OLLAMA_API_KEY") if "ollama.com" in self.host else None
        )
        headers = {"Authorization": "Bearer " + api_key} if api_key else {}

        client = Client(host=self.host, headers=headers)

        kwargs = {}
        if response_schema:
            kwargs["format"] = response_schema.model_json_schema()
        if tools:
            kwargs["tools"] = tools

        options = {"temperature": self.config.temperature}
        if self.config.max_tokens is not None:
            options["num_predict"] = self.config.max_tokens
        else:
            options["num_predict"] = -1  # Indicates infinite generation for Ollama
        kwargs["options"] = options

        response = client.chat(
            self.config.model, messages=messages, stream=False, **kwargs
        )

        t_calls = []
        if getattr(response.message, "tool_calls", None):
            for tc in response.message.tool_calls:
                t_calls.append(
                    ToolCallOutput(
                        name=tc.function.name, arguments=tc.function.arguments
                    )
                )

        return LLMessageResponse(content=response.message.content, tool_calls=t_calls)


class DeepInfraLLMProvider(OpenAILLMProvider):
    def invoke(
        self,
        messages: List[Dict[str, str]],
        response_schema: type[BaseModel] | None = None,
        tools: Optional[List[Any]] = None,
    ) -> LLMessageResponse:
        client = openai.OpenAI(
            base_url="https://api.deepinfra.com/v1/openai",
            api_key=os.environ.get("DEEPINFRA_API_KEY"),
        )
        return self._invoke_with_client(client, messages, response_schema, tools)


class OllamaLLMProvider(OllamaLLMProviderBase):
    def __init__(self, config: LLMConfig):
        super().__init__(config, host="http://localhost:11434")


class OllamaCloudLLMProvider(OllamaLLMProviderBase):
    def __init__(self, config: LLMConfig):
        super().__init__(config, host="https://ollama.com")


def get_llm_provider(config: LLMConfig) -> BaseLLMProvider:
    """Factory function to instantiate the correct provider."""
    if config.provider == "openai":
        return OpenAILLMProvider(config)
    elif config.provider == "ollama":
        return OllamaLLMProvider(config)
    elif config.provider == "ollama-cloud":
        return OllamaCloudLLMProvider(config)
    elif config.provider == "deepinfra":
        return DeepInfraLLMProvider(config)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")


def test_ollama_connection():
    """Simple test to verify Ollama is working."""
    config = LLMConfig(provider="ollama", model="qwen3:8b", max_tokens=50)
    provider = get_llm_provider(config)
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Ollama is working!'"},
        ]
        response = provider.invoke(messages)
        print(f"Ollama Test Response: {response.content}")
        return True
    except Exception as e:
        print(f"Ollama Test Failed: {e}")
        return False


if __name__ == "__main__":
    # Test block
    print("Testing Ollama Connection...")
    test_ollama_connection()
