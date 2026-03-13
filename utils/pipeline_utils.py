import os
import agentdojo.models as models
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig


def setup_agentdojo_pipeline(
    model_path: str,
    system_message_name: str = "default",
    provider: str = "transformers",
) -> AgentPipeline:
    """
    Sets up an AgentPipeline configuration, handling LlaMA-3 instruct appending
    and custom HF path model registrations.
    """
    if "llama-3" in model_path.lower() and not model_path.endswith("-instruct"):
        if os.path.exists(model_path + "-instruct"):
            print(
                f"Warning: Base model provided ({model_path}). Auto-appending '-instruct' to ensure tool-calling behaves correctly."
            )
            model_path = model_path + "-instruct"

    if provider == "ollama":
        config = PipelineConfig(
            llm=model_path,
            model_id=model_path,
            system_message_name=system_message_name,
            system_message=None,
            defense=None,
            tool_delimiter="tool",
        )
        models.ModelsEnum._value2member_map_[model_path] = model_path
        models.MODEL_PROVIDERS[model_path] = "local"
        pipeline = AgentPipeline.from_config(config)
        return pipeline

    elif provider == "deepinfra":
        import openai
        from utils.deepinfra_llm import DeepInfraLLM
        from agentdojo.agent_pipeline.basic_elements import SystemMessage, InitQuery
        from agentdojo.agent_pipeline.tool_execution import (
            ToolsExecutionLoop,
            ToolsExecutor,
            tool_result_to_str,
        )

        models.ModelsEnum._value2member_map_[model_path] = model_path
        models.MODEL_PROVIDERS[model_path] = "openai"

        client = openai.OpenAI(
            base_url="https://api.deepinfra.com/v1/openai",
            api_key=os.environ.get("DEEPINFRA_API_KEY"),
        )

        llm = DeepInfraLLM(client, model_path)
        from agentdojo.agent_pipeline.agent_pipeline import load_system_message

        system_msg_content = load_system_message(system_message_name)
        system_message_component = SystemMessage(system_msg_content)
        init_query_component = InitQuery()
        tools_loop = ToolsExecutionLoop([ToolsExecutor(tool_result_to_str), llm])

        pipeline = AgentPipeline(
            elements=[system_message_component, init_query_component, llm, tools_loop]
        )
        pipeline.name = model_path
        return pipeline

    elif provider == "openai":
        models.ModelsEnum._value2member_map_[model_path] = model_path
        models.MODEL_PROVIDERS[model_path] = "openai"

        config = PipelineConfig(
            llm=model_path,
            model_id=None,
            system_message=None,
            defense=None,
            system_message_name=system_message_name,
        )
        return AgentPipeline.from_config(config)

    # Default to local transformers huggingface pipeline
    models.ModelsEnum._value2member_map_[model_path] = model_path
    models.MODEL_PROVIDERS[model_path] = "transformers"

    config = PipelineConfig(
        llm=model_path,
        model_id=None,
        system_message=None,
        defense=None,
        system_message_name=system_message_name,
    )
    return AgentPipeline.from_config(config)
