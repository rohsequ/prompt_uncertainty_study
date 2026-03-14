"""
Pydantic models for structured LLM responses in the simple agent system.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, ToolMessage
from typing_extensions import TypedDict


class ToolCallSpec(BaseModel):
    """Specification for a tool call."""

    name: str = Field(..., description="The name of the tool to call")
    args: Dict[str, Any] = Field(..., description="Arguments to pass to the tool")
    id: str = Field(..., description="Unique identifier for this tool call")


class ToolResponseSpec(BaseModel):
    """Specification for a tool response."""

    content: str = Field(..., description="The content of the tool response")
    tool_call_id: str = Field(
        ..., description="ID of the tool call this response is for"
    )


class AgentInteractionData(BaseModel):
    """Data structure for agent interaction from JSON files."""

    user_message: str = Field(..., description="The benign user prompt")
    tool_call: ToolCallSpec = Field(..., description="The tool call specification")
    payload_template: str = Field(
        ...,
        description="Template for tool response with attack instruction placeholder",
    )
    tool_response_format: ToolResponseSpec = Field(
        ..., description="Format for the tool response"
    )


class AgentState(TypedDict):
    """State for the LangGraph agent workflow."""

    messages: List[Union[AIMessage, ToolMessage]]
    current_tool_call: Optional[ToolCallSpec]
    final_response: Optional[str]
