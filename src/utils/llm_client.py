"""
LLM client for generating tool call sequences.

Reuses OpenAI client pattern from src/target_environment.py
Uses structured output with Pydantic models.
"""

from typing import Dict, Any, Optional
from openai import OpenAI
from pydantic import BaseModel


class LLMClient:
    """
    Client for interacting with LLM providers.

    Pattern based on TargetEnvironment's eval_client initialization.
    Supports structured output with Pydantic models.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM client.

        Args:
            config: LLM configuration dictionary with keys:
                - model_provider: Provider name (e.g., 'ollama', 'openai')
                - model_name: Model identifier
                - base_url: API base URL (for Ollama/custom endpoints)
                - api_key: API key
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens to generate
        """
        self.config = config
        self.provider = config["model_provider"]
        self.model_name = config["model_name"]

        # Initialize OpenAI-compatible client directly
        # For dict-based config, use the inline approach since it already has the values we need
        if self.provider.lower() == "ollama" or self.provider.lower() == "deepinfra":
            self.client = OpenAI(base_url=config["base_url"], api_key=config["api_key"])
        elif self.provider.lower() == "openai":
            self.client = OpenAI(api_key=config["api_key"])
        else:
            # Assume OpenAI-compatible API
            self.client = OpenAI(
                base_url=config.get("base_url"), api_key=config["api_key"]
            )

    def generate_structured_output(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> BaseModel:
        """
        Generate structured output using Pydantic model.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt/query
            response_model: Pydantic model class for structured output
            temperature: Temperature parameter (overrides config if provided)
            max_tokens: Max tokens (overrides config if provided)

        Returns:
            Parsed Pydantic model instance
        """
        temp = (
            temperature
            if temperature is not None
            else float(self.config["temperature"])
        )
        max_tok = (
            max_tokens if max_tokens is not None else int(self.config["max_tokens"])
        )

        try:
            # Use beta.chat.completions.parse for structured output
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_model,
                temperature=temp,
                max_tokens=max_tok,
            )

            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise ValueError("Failed to parse structured output")

            return parsed

        except Exception as e:
            # Fallback to manual JSON parsing if structured output not supported
            print(f"⚠️  Structured output failed, falling back to manual parsing: {e}")
            return self._fallback_json_parsing(
                system_prompt, user_prompt, response_model, temp, max_tok
            )

    def _fallback_json_parsing(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        temperature: float,
        max_tokens: int,
    ) -> BaseModel:
        """
        Fallback method using manual JSON parsing.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            response_model: Pydantic model class
            temperature: Temperature
            max_tokens: Max tokens

        Returns:
            Parsed Pydantic model instance
        """
        import json
        import re

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from LLM")

        # Remove markdown code blocks if present
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]

        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        # Try to parse JSON
        try:
            data = json.loads(content)
            return response_model(**data)
        except json.JSONDecodeError:
            # Try to extract JSON object
            json_match = re.search(r"\{[\s\S]*\}", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                return response_model(**data)
            else:
                raise ValueError(
                    f"Failed to parse JSON from response: {content[:200]}..."
                )
