import os
import json
from pydantic import BaseModel
from utils.llm_utils import get_llm_provider, LLMConfig

# ==========================================
# Test Definitions
# ==========================================


class PersonInfo(BaseModel):
    name: str
    age: int


def get_weather(location: str) -> str:
    """Get the current weather in a given location.

    Args:
        location: The city and state, e.g. 'Seattle, WA'
    """
    return f"The weather in {location} is 72F and sunny."


# ==========================================
# Test Runner
# ==========================================


def run_tests():
    providers = [
        {"provider": "openai", "model": "gpt-4o-mini"},  # fast and cheap for testing
        {"provider": "ollama", "model": "qwen3:8b"},
        {"provider": "ollama-cloud", "model": "gpt-oss:120b"},
        {"provider": "deepinfra", "model": "meta-llama/Meta-Llama-3.1-70B-Instruct"},
    ]

    for p in providers:
        print(f"\n{'='*50}")
        print(f"TESTING PROVIDER: {p['provider']} (Model: {p['model']})")
        print(f"{'='*50}")

        try:
            config = LLMConfig(
                provider=p["provider"], model=p["model"], temperature=0.0
            )
            llm = get_llm_provider(config)

            # ---------------------------------------------------------
            # 1. Chat Response
            # ---------------------------------------------------------
            print("\n--- 1. Basic Chat Test ---")
            messages = [{"role": "user", "content": "Say EXACTLY 'Hello World!'"}]
            response = llm.invoke(messages)
            print(f"Result: {response.content}")

            # ---------------------------------------------------------
            # 2. Structured Response
            # ---------------------------------------------------------
            print("\n--- 2. Structured JSON Test ---")
            messages = [
                {
                    "role": "user",
                    "content": "Extract the following: John Doe is 30 years old.",
                }
            ]
            response = llm.invoke(messages, response_schema=PersonInfo)
            print(f"Result: {response.content}")

            # ---------------------------------------------------------
            # 3. Tool Calling
            # ---------------------------------------------------------
            print("\n--- 3. Tool Calling Test ---")
            messages = [
                {"role": "user", "content": "What is the weather like in Seattle, WA?"}
            ]
            response = llm.invoke(messages, tools=[get_weather])

            if response.tool_calls:
                for tc in response.tool_calls:
                    print(f"Tool Called: {tc.name}")
                    print(f"Arguments: {tc.arguments}")
            else:
                print(
                    f"FAILED TO CALL TOOL. Content output instead: {response.content}"
                )

        except Exception as e:
            print(f"ERROR testing {p['provider']}: {e}")


if __name__ == "__main__":
    run_tests()
