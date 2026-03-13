import logging

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.functions_runtime import EmptyEnv, make_function, FunctionsRuntime
from agentdojo.types import ChatUserMessage, text_content_block_from_string
import agentdojo.models as models

logging.basicConfig(level=logging.INFO)


def get_weather(location: str) -> str:
    """Gets the weather for a given location.

    args:
        location: The location to get the weather for.
    """
    return "Weather is Sunny"


function_dict = make_function(get_weather)
from utils.pipeline_utils import setup_agentdojo_pipeline

# Setup pipeline using our custom utilities wrapper which correctly populates empty PipelineConfig params
pipeline = setup_agentdojo_pipeline(
    model_path="/research-storage/lab/rohseque/huggingface_models/meta-llama-3-1-8b-instruct",
    system_message_name="default",
    provider="transformers",
)

runtime = FunctionsRuntime([function_dict])
messages = [
    ChatUserMessage(
        role="user",
        content=[text_content_block_from_string("What is the weather in Paris?")],
    )
]

print("Querying the pipeline...")
query, runtime, env, output_messages, extra = pipeline.query(
    "test", runtime, EmptyEnv(), messages, {}
)
print("Final generated messages:")
for msg in output_messages:
    if msg["role"] == "assistant":
        print(f"Assistant Output Content: {msg.get('content')}")
        print(f"Assistant Tool Calls: {msg.get('tool_calls')}")
