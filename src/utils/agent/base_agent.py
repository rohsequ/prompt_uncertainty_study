"""
Base Agent class containing shared functionality for both GeneralAgent and SimpleAgent.

This module provides the common methods that both agent types use for tool execution
with gatekeeper functionality and defense checking.
"""

import inspect
import traceback
from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolMessage,
)

# Rich imports for the gatekeeper function
from rich.console import Console
from rich.panel import Panel

from src.utils.models import DefenseEvaluation
from src.utils.config_loader import ConfigManager

console = Console()


class BaseAgent:
    """
    Base class containing shared agent functionality for tool execution and defense.

    This class provides common methods that can be used by both GeneralAgent and SimpleAgent.
    """

    def __init__(self, config_path: Optional[str] = None, use_gatekeeper: bool = False):
        """
        Initialize the base agent with gatekeeper configuration.

        Args:
            config_path: Optional path to configuration file
            use_gatekeeper: Whether to enable gatekeeper functionality
        """
        self.use_gatekeeper = use_gatekeeper

        # Load configuration
        self.config = ConfigManager(config_path) if config_path else ConfigManager()

    def _init_chat_model_with_config(self, model_name: str, model_provider: str):
        """
        Initialize a chat model with configuration support for custom base URLs.

        This method is deprecated. It exists for backward compatibility with code
        that passes model_name and model_provider directly. New code should use
        load_chat_model() from model_loader directly.

        Args:
            model_name: Name of the model to use
            model_provider: Provider of the model (e.g., "ollama", "openai")

        Returns:
            Initialized chat model
        """
        from langchain.chat_models import init_chat_model

        if model_provider.lower() == "ollama":
            # Get Ollama base URL from config if available
            ollama_base_url = self.config.get("ollama", "base_url", fallback=None)
            if ollama_base_url:
                return init_chat_model(
                    model=model_name,
                    model_provider=model_provider,
                    base_url=f"http://{ollama_base_url}",
                )

        # Default initialization for other providers or when no custom base URL is set
        return init_chat_model(model=model_name, model_provider=model_provider)

    def run_defenses(self, state: Dict[str, Any]) -> DefenseEvaluation:
        """
        Run defense checking code from the state.

        Args:
            state: Agent state containing defense_code and messages

        Returns:
            DefenseEvaluation with safety result and reason
        """
        try:
            code = state["defense_code"]
            messages = state["messages"]
            namespace = {}
            exec(code, namespace)
            (safe, reason) = namespace["safety_check_fn"](messages)
            result = DefenseEvaluation(safe=safe, reason=reason)
            return result
        except Exception as e:
            reason = f"Error running defense checking code. Assuming safe by default.\nError was:\n{traceback.format_exc()}"
            result = DefenseEvaluation(safe=True, reason=reason)
            return result

    def _check_tool_approval_with_gatekeeper(
        self, state: Dict[str, Any], tool_call: Dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Check if a tool call is approved by the gatekeeper without executing it.

        Args:
            state: Agent state containing messages and other data
            tool_call: Single tool call to check

        Returns:
            Tuple of (is_approved, reason)
        """
        user_prompt = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                user_prompt = str(msg.content) if msg.content else ""
                break

        is_approved = True  # Default to approved
        reason = "No checks enabled"

        # Run gatekeeper if enabled
        if self.use_gatekeeper:
            # Use the function based run_defenses method
            console.print(
                Panel(
                    "[bold cyan]Using Function Based Defense Model[/bold cyan]",
                    border_style="cyan",
                    title_align="left",
                )
            )

            blue_agent_result = self.run_defenses(state)
            blue_agent_info = (
                f"[bold cyan]Safe:[/bold cyan] {blue_agent_result.safe}\n"
                f"[bold cyan]Reason:[/bold cyan] {blue_agent_result.reason}"
            )
            console.print(
                Panel(
                    blue_agent_info,
                    title="Function Based Defense Result",
                    border_style="#0000FF",
                )
            )
            is_approved = blue_agent_result.safe
            reason = blue_agent_result.reason
        else:
            # If all are off, print a notice that the check was skipped
            console.print(
                Panel(
                    "[bold yellow]GATEKEEPER CHECK SKIPPED[/bold yellow]",
                    border_style="dim",
                    title_align="left",
                )
            )

        return is_approved, reason

    def _execute_tools_with_gatekeeper(
        self, state: Dict[str, Any], tools: List[Any]
    ) -> Dict[str, List[ToolMessage]]:
        """
        Execute tools with gatekeeper protection.

        Args:
            state: Agent state containing messages and other data
            tools: List of available tools for the agent

        Returns:
            Dictionary containing list of tool messages
        """
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": []}

        user_prompt = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                user_prompt = str(msg.content) if msg.content else ""
                break

        tool_messages = []
        # Create tool map with proper name handling for different tool types
        tool_map = {}
        for tool in tools:
            if hasattr(tool, "name") and tool.name:
                tool_name = tool.name
            elif hasattr(tool, "__name__"):
                tool_name = tool.__name__
            else:
                tool_name = f"unknown_tool_{id(tool)}"
            tool_map[tool_name] = tool

        for tool_call in last_message.tool_calls:
            # Convert tool_call to dict format expected by gatekeeper
            tool_call_dict = {
                "name": tool_call["name"],
                "args": tool_call["args"],
                "id": tool_call["id"],
            }

            # Use the new approval checking function
            is_approved, reason = self._check_tool_approval_with_gatekeeper(
                state, tool_call_dict
            )

            if is_approved:
                tool_to_run = tool_map.get(tool_call["name"])
                if tool_to_run:
                    tool_args = tool_call["args"].copy()
                    tool_signature = inspect.signature(tool_to_run)
                    if "environment" in tool_signature.parameters:
                        tool_args["environment"] = state.get("environment")

                    try:
                        if hasattr(tool_to_run, "invoke"):
                            output = tool_to_run.invoke(tool_args)
                        else:
                            output = tool_to_run(**tool_args)
                        tool_messages.append(
                            ToolMessage(
                                content=str(output), tool_call_id=tool_call["id"]
                            )
                        )
                    except Exception as e:
                        tool_messages.append(
                            ToolMessage(
                                content=f"Error running tool {tool_call['name']}: {e}",
                                tool_call_id=tool_call["id"],
                            )
                        )
                else:
                    error_msg = f"Error: Tool '{tool_call['name']}' not found."
                    tool_messages.append(
                        ToolMessage(content=error_msg, tool_call_id=tool_call["id"])
                    )
            else:
                # If the gatekeeper denied the call, create a denial message
                # Construct the denial message using the specific reason
                denial_message = f"Error: The tool call to '{tool_call['name']}' was denied by the gatekeeper. Reason: {reason}."
                tool_messages.append(
                    ToolMessage(content=denial_message, tool_call_id=tool_call["id"])
                )

        return {"messages": tool_messages}
