from utils.llm_utils import get_llm_provider, LLMConfig


def get_temperature(city: str) -> str:
    """Get the current temperature for a city

    Args:
        city: The name of the city
    """
    return "22°C"


config = LLMConfig(provider="ollama-cloud", model="gpt-oss:120b", max_tokens=100)
provider = get_llm_provider(config)

messages = [{"role": "user", "content": "What is the temperature in New York?"}]
print("Calling LLM...")
response = provider.invoke(messages, tools=[get_temperature])

print("Tool Calls:")
for tc in response.tool_calls or []:
    print(f"Name: {tc.name}, Args: {tc.arguments}")
