from utils.llm_utils import get_llm_provider, LLMConfig
from pydantic import BaseModel


class Country(BaseModel):
    name: str
    capital: str


config = LLMConfig(provider="ollama-cloud", model="gpt-oss:120b", max_tokens=50)
provider = get_llm_provider(config)

messages = [
    {"role": "system", "content": "You are an assistant. Return only JSON."},
    {"role": "user", "content": "Tell me about Canada."},
]

response = provider.invoke(messages, response_schema=Country)
print(response.content)
