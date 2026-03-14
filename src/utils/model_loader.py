"""
Centralized Model Loading Utilities

This module provides unified functions for loading LLM models from configuration.
Supports multiple providers: Ollama, OpenAI, DeepInfra, and NVIDIA.

All model loading logic is centralized here to enable easy addition of new providers
and consistent configuration patterns across the codebase.
"""

from typing import Optional
from openai import OpenAI
from src.utils.config_loader import ConfigManager


# Global constants for provider base URLs
DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def load_chat_model(
    config: ConfigManager,
    config_section: str,
    model_key_prefix: str = "",
):
    """
    Load a LangChain chat model from configuration.

    This function reads model configuration from a config section and initializes
    a LangChain chat model with the appropriate provider settings.

    Args:
        config: ConfigManager instance containing configuration
        config_section: Config section name (e.g., "simple_agents", "models")
        model_key_prefix: Optional prefix for model config keys (e.g., "tool_response_")
                         Allows loading multiple models from the same section.

    Returns:
        Initialized LangChain BaseChatModel instance

    Example:
        >>> config = ConfigManager("config.ini")
        >>> # Load main agent model from [simple_agents] section
        >>> agent_model = load_chat_model(config, "simple_agents")
        >>>
        >>> # Load tool response model with prefix
        >>> tool_model = load_chat_model(config, "simple_agents", "tool_response_")

    Config format:
        [simple_agents]
        model_provider = ollama
        model_name = llama3.3:70b
        base_url = http://10.15.30.24:11434/v1
        api_key = ollama

        tool_response_model_provider = ollama
        tool_response_model_name = llama3.3:70b
        tool_response_base_url = http://10.15.30.24:11434/v1
        tool_response_api_key = ollama
    """
    from langchain.chat_models import init_chat_model
    import os

    # Read model configuration
    provider_key = f"{model_key_prefix}model_provider"
    name_key = f"{model_key_prefix}model_name"
    base_url_key = f"{model_key_prefix}base_url"
    api_key_key = f"{model_key_prefix}api_key"

    model_provider = config.get(config_section, provider_key)
    model_name = config.get(config_section, name_key)

    # Provider-specific initialization with base URLs
    if model_provider.lower() == "ollama":
        # Ollama needs base_url from config (local deployment, varies)
        # Important: LangChain's Ollama client expects base URL WITHOUT /v1 suffix
        # The /v1 suffix is only for OpenAI-compatible API (used in load_openai_client)
        base_url = config.get(config_section, base_url_key)
        # Strip /v1 suffix if present
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]

        api_key = config.get(config_section, api_key_key, fallback="ollama")
        return init_chat_model(
            model=model_name,
            model_provider=model_provider,
            base_url=base_url,
            api_key=api_key,
        )

    elif model_provider.lower() == "deepinfra":
        # DeepInfra uses global base URL constant and reads API key from environment
        api_key = os.getenv("DEEPINFRA_API_KEY")
        if not api_key:
            raise ValueError(
                "DeepInfra API key not found. Please set the DEEPINFRA_API_KEY environment variable.\n"
                "Example: export DEEPINFRA_API_KEY=your_key_here"
            )
        return init_chat_model(
            model=model_name,
            model_provider="openai",  # Use OpenAI compatible provider
            base_url=DEEPINFRA_BASE_URL,
            api_key=api_key,
        )

    elif model_provider.lower() == "nvidia":
        # NVIDIA uses global base URL constant and reads API key from environment
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError(
                "NVIDIA API key not found. Please set the NVIDIA_API_KEY environment variable.\n"
                "Example: export NVIDIA_API_KEY=your_key_here"
            )
        return init_chat_model(
            model=model_name,
            model_provider="openai",  # Use OpenAI compatible provider
            base_url=NVIDIA_BASE_URL,
            api_key=api_key,
        )

    elif model_provider.lower() == "openai":
        # OpenAI reads API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.\n"
                "Example: export OPENAI_API_KEY=your_key_here"
            )
        return init_chat_model(
            model=model_name,
            model_provider=model_provider,
            api_key=api_key,
        )

    # Default initialization for other providers
    return init_chat_model(model=model_name, model_provider=model_provider)


def load_openai_client(
    config: ConfigManager,
    config_section: str,
    model_key_prefix: str = "",
) -> OpenAI:
    """
    Load an OpenAI-compatible client from configuration.

    This function reads API configuration and returns an initialized OpenAI client.
    Works with OpenAI, Ollama (OpenAI-compatible API), DeepInfra, and NVIDIA.

    Base URLs are defined as global constants in this module:
    - DeepInfra: DEEPINFRA_BASE_URL
    - NVIDIA: NVIDIA_BASE_URL
    - Ollama: Read from config (local deployment, varies by setup)
    - OpenAI: Default (no custom base URL needed)

    API keys are ONLY read from environment variables (for security):
    - OpenAI: OPENAI_API_KEY (required)
    - DeepInfra: DEEPINFRA_API_KEY (required)
    - NVIDIA: NVIDIA_API_KEY (required)
    - Ollama: Read from config (not sensitive, local deployment)

    Args:
        config: ConfigManager instance containing configuration
        config_section: Config section name (e.g., "models", "pair_attack")
        model_key_prefix: Optional prefix for config keys (e.g., "attack_eval_")
                         Allows loading multiple clients from the same section.

    Returns:
        Initialized OpenAI client instance

    Example:
        >>> config = ConfigManager("config.ini")
        >>> # Load attack evaluation client (reads OPENAI_API_KEY from env if available)
        >>> eval_client = load_openai_client(config, "models", "attack_eval_")
        >>>
        >>> # Load attack generation client
        >>> gen_client = load_openai_client(config, "models", "attack_gen_")

    Config format (only provider and model name needed, base URLs are global):
        [models]
        attack_eval_provider = openai
        attack_eval_model = gpt-4
        # No base_url or api_key needed - uses globals and environment!

        attack_gen_provider = nvidia
        attack_gen_model = meta/llama-3.1-70b-instruct
        # Uses NVIDIA_BASE_URL global and NVIDIA_API_KEY from environment

        # Ollama still needs base_url since it's local
        attack_gen_provider = ollama
        attack_gen_model = llama3.3:70b
        attack_gen_base_url = http://10.15.30.24:11434/v1
        attack_gen_api_key = ollama
    """
    import os

    # Read provider configuration
    provider_key = f"{model_key_prefix}provider"
    base_url_key = f"{model_key_prefix}base_url"
    api_key_key = f"{model_key_prefix}api_key"

    model_provider = config.get(config_section, provider_key)

    # Provider-specific client initialization
    if model_provider.lower() == "ollama":
        # Ollama with OpenAI-compatible API - needs base_url from config (local deployment)
        base_url = config.get(config_section, base_url_key)

        # Ensure protocol exists
        if base_url and not (
            base_url.startswith("http://") or base_url.startswith("https://")
        ):
            base_url = f"http://{base_url}"

        # Ollama doesn't use standard env vars, use config value
        api_key = config.get(config_section, api_key_key, fallback="ollama")
        return OpenAI(base_url=base_url, api_key=api_key)

    elif model_provider.lower() == "openai":
        # Standard OpenAI API - uses default base URL
        # ONLY reads from environment variable (no config fallback for security)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.\n"
                "Example: export OPENAI_API_KEY=your_key_here"
            )
        return OpenAI(api_key=api_key)

    elif model_provider.lower() == "deepinfra":
        # DeepInfra with OpenAI-compatible API - uses global base URL
        # ONLY reads from environment variable (no config fallback for security)
        api_key = os.getenv("DEEPINFRA_API_KEY")
        if not api_key:
            raise ValueError(
                "DeepInfra API key not found. Please set the DEEPINFRA_API_KEY environment variable.\n"
                "Example: export DEEPINFRA_API_KEY=your_key_here"
            )
        return OpenAI(base_url=DEEPINFRA_BASE_URL, api_key=api_key)

    elif model_provider.lower() == "nvidia":
        # NVIDIA with OpenAI-compatible API - uses global base URL
        # ONLY reads from environment variable (no config fallback for security)
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError(
                "NVIDIA API key not found. Please set the NVIDIA_API_KEY environment variable.\n"
                "Example: export NVIDIA_API_KEY=your_key_here"
            )
        return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)

    else:
        # Generic fallback - assume OpenAI-compatible API
        # For custom providers, try to get base_url from config
        base_url = config.get(config_section, base_url_key, fallback=None)
        if not base_url:
            raise ValueError(
                f"Base URL not found for provider '{model_provider}'. "
                f"Add '{base_url_key}' to [{config_section}] in config."
            )

        # Try environment variable based on provider name
        env_var_name = f"{model_provider.upper()}_API_KEY"
        api_key = os.getenv(env_var_name) or config.get(
            config_section, api_key_key, fallback=None
        )
        if not api_key:
            raise ValueError(
                f"{model_provider} API key not found. Set {env_var_name} environment variable "
                f"or add '{api_key_key}' to [{config_section}] in config."
            )
        return OpenAI(base_url=base_url, api_key=api_key)
