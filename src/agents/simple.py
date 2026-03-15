"""
Simple agent runtime built on LangGraph and JSON-backed agent definitions.
"""

import json
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

from .registry import AgentRegistry, get_agent_registry
from .schemas import AgentInteractionData, AgentState
from src.utils.config_loader import ConfigManager


class SimpleAgent:
    """Simple tool-using agent backed by JSON tool and prompt definitions."""

    def __init__(self, agent_type: str, config_path: Optional[str] = None):
        self.base_dir = Path(__file__).parent
        self.definitions_dir = self.base_dir / "definitions"
        self.agent_type = agent_type
        self.config = ConfigManager(config_path) if config_path else ConfigManager()

        cfg_section = "simple_agents"
        try:
            self.config.get(cfg_section, "model_name")
            self.config.get(cfg_section, "model_provider")
            self.config.get(cfg_section, "tool_response_model_name")
            self.config.get(cfg_section, "tool_response_model_provider")
        except Exception as exc:
            raise ValueError(
                f"SimpleAgent model configuration is incomplete in section '{cfg_section}'"
            ) from exc

        self.registry: AgentRegistry = get_agent_registry()
        self.system_message: Optional[str] = self.registry.get_system_prompt(
            self.agent_type
        )
        self._tools_cache: Dict[str, List[tool]] = {}

        from src.utils.model_loader import load_chat_model

        self.agent_model = load_chat_model(self.config, cfg_section)
        self.tool_response_model = load_chat_model(
            self.config, cfg_section, "tool_response_"
        )

        tools = self._get_tools_for_agent()
        if tools:
            self.agent_model = self.agent_model.bind_tools(tools)
            print(f"🔧 Bound {len(tools)} tools to {self.agent_type} agent model")
        else:
            print(f"⚠️  No tools found for {self.agent_type} agent")

        self.prompt_mapping = self._load_prompt_mapping()
        self.graph = self._build_graph()

    def _load_prompt_mapping(self) -> List[Dict[str, Any]]:
        """Load prompt guidance for simulated tool responses."""
        mapping_path = self.definitions_dir / self.agent_type / "prompt_mapping.json"
        if not mapping_path.exists():
            print(f"ℹ️  No prompt mapping found for {self.agent_type} at {mapping_path}")
            return []

        try:
            with open(mapping_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            print(f"⚠️  Error loading prompt mapping: {exc}")
            return []

    def _get_tool_prompt_config(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Find the prompt mapping entry for a specific tool."""
        for group in self.prompt_mapping:
            if tool_name in group.get("tools", []):
                return group
        return None

    def _convert_tool_spec_to_langchain(self, tool_spec: Dict[str, Any]) -> tool:
        """Convert a JSON tool spec into a LangChain tool."""
        name = tool_spec["name"]
        description = tool_spec["description"]
        parameters = tool_spec["parameters"]

        def tool_function(**kwargs):
            return f"Tool {name} called with parameters: {kwargs}"

        tool_function.__name__ = name
        tool_function.__doc__ = description

        lc_tool = tool(tool_function)
        lc_tool.name = name
        lc_tool.description = description
        lc_tool.args_schema = self._create_pydantic_schema(parameters)
        return lc_tool

    def _create_pydantic_schema(self, parameters: Dict[str, Any]):
        """Create a Pydantic args schema from an OpenAPI-style parameter block."""
        from pydantic import BaseModel, Field
        from typing import Any as TypingAny
        from typing import List as TypingList
        from typing import Optional as TypingOptional

        properties = parameters.get("properties", {})
        required = parameters.get("required", [])
        fields = {}

        for prop_name, prop_spec in properties.items():
            prop_type = prop_spec.get("type", "string")
            prop_description = prop_spec.get("description", "")
            is_required = prop_name in required

            if prop_type == "string":
                field_type = str
            elif prop_type == "integer":
                field_type = int
            elif prop_type == "number":
                field_type = float
            elif prop_type == "boolean":
                field_type = bool
            elif prop_type == "array":
                items_type = prop_spec.get("items", {}).get("type", "string")
                field_type = TypingList[str] if items_type == "string" else TypingList[TypingAny]
            else:
                field_type = TypingAny

            if is_required:
                fields[prop_name] = (field_type, Field(description=prop_description))
            else:
                fields[prop_name] = (
                    TypingOptional[field_type],
                    Field(default=None, description=prop_description),
                )

        model_name = f"ToolParameters_{parameters.get('title', 'Default')}"
        return type(
            model_name,
            (BaseModel,),
            {
                "__annotations__": {name: typ for name, (typ, _) in fields.items()},
                **{name: field for name, (_, field) in fields.items()},
                "model_config": {"arbitrary_types_allowed": True},
            },
        )

    def _get_tools_for_agent(self) -> List[tool]:
        """Load and cache LangChain tool wrappers for the current agent."""
        if self.agent_type in self._tools_cache:
            return self._tools_cache[self.agent_type]

        tools: List[tool] = []
        for spec in self.registry.get_tool_specs(self.agent_type):
            try:
                tools.append(self._convert_tool_spec_to_langchain(spec))
            except Exception as exc:
                print(f"Error converting tool spec {spec.get('name', 'unknown')}: {exc}")

        self._tools_cache[self.agent_type] = tools
        return tools

    def _normalize_tool_call(self, tool_call: Any) -> Dict[str, Any]:
        """Normalize LangChain/OpenAI-style tool calls into a dict shape."""
        if isinstance(tool_call, dict):
            name = tool_call.get("name")
            args = tool_call.get("args", {})
            tool_id = tool_call.get("id")
        else:
            name = getattr(tool_call, "name", None)
            args = getattr(tool_call, "args", {})
            tool_id = getattr(tool_call, "id", None)

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}

        if not isinstance(args, dict):
            args = {}

        return {"name": name, "args": args, "id": tool_id}

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("agent_llm", self._call_agent_llm)
        workflow.add_node("tool_response_llm", self._call_tool_response_llm)
        workflow.add_node("generate_final", self._generate_final_response)
        workflow.set_entry_point("agent_llm")
        workflow.add_conditional_edges(
            "agent_llm",
            self._should_continue,
            {"tool_call": "tool_response_llm", "final": "generate_final"},
        )
        workflow.add_edge("tool_response_llm", "agent_llm")
        workflow.add_edge("generate_final", END)
        return workflow.compile()

    def _call_agent_llm(self, state) -> Dict[str, Any]:
        messages = state["messages"]
        if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
            return {"messages": messages}

        response = self.agent_model.invoke(messages)
        return {"messages": messages + [response]}

    def _call_tool_response_llm(self, state) -> Dict[str, Any]:
        """Simulate tool execution results with the configured tool-response model."""
        messages = state["messages"]
        if not messages:
            return {"messages": messages}

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {"messages": messages}

        tool_responses = []

        for raw_tool_call in last_message.tool_calls:
            tool_call = self._normalize_tool_call(raw_tool_call)
            if not tool_call["name"]:
                continue

            prompt_config = self._get_tool_prompt_config(tool_call["name"])
            base_directives = dedent(
                """
                    **CRITICAL RESPONSE DIRECTIVES**

                    **1. Output Formatting:**
                    - **Raw Output ONLY**: You MUST return only the tool's direct output.
                    - **NO Explanations**: Do NOT include any conversational text.

                    **2. Parameter Handling Logic:**
                    - **EMPTY PARAMETERS ARE VALID**: Treat empty filters as a request for all matching data.
                    - **MISSING PARAMETERS ARE ERRORS**: Return `ERROR: Missing required parameter 'name'`.

                    **3. Contextual Generation Guidelines:**
                    - Match the expected schema shape.
                    - Keep values semantically realistic for the domain.
                """
            )

            if prompt_config:
                system_instruction = prompt_config.get(
                    "prompt_text", "You are a tool execution simulator."
                )
                realism_rules = "\n".join(
                    f"- {rule}" for rule in prompt_config.get("realism_rules", [])
                )
                tool_response_prompt = dedent(
                    f"""
                    {system_instruction}

                    TOOL DETAILS:
                    - Name: {tool_call['name']}
                    - Arguments: {json.dumps(tool_call['args'], indent=2)}

                    {base_directives}

                    **4. REALISM STANDARDS (STRICT ENFORCEMENT):**
                    {realism_rules}

                    Generate the realistic tool response now:
                    """
                )
            else:
                system_instruction = (
                    "You are a precise tool execution simulator. Return only the exact output a real tool would produce."
                )
                tool_response_prompt = dedent(
                    f"""
                    You are a tool execution simulator. Generate only the raw response that the tool would return.

                    TOOL DETAILS:
                    - Name: {tool_call['name']}
                    - Arguments: {json.dumps(tool_call['args'], indent=2)}

                    {base_directives}

                    **4. Realism & Relevance Requirements:**
                    - Use plausible values, names, and timestamps.
                    - Preserve structural and semantic realism.

                    Generate the contextual tool response now:
                    """
                )

            response = self.tool_response_model.invoke(
                [
                    SystemMessage(content=system_instruction),
                    HumanMessage(content=tool_response_prompt),
                ]
            )
            tool_responses.append(
                ToolMessage(
                    content=response.content,
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                )
            )

        return {"messages": messages + tool_responses}

    def _generate_final_response(self, state) -> Dict[str, Any]:
        messages = state["messages"]
        final_ai_message = None
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
                final_ai_message = message
                break

        return {
            "messages": messages,
            "final_response": (
                final_ai_message.content if final_ai_message else "No response generated"
            ),
        }

    def _should_continue(self, state) -> str:
        messages = state["messages"]
        if not messages:
            return "final"

        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
            return "tool_call"
        return "final"

    def load_agent_data(self, interaction_file: str) -> AgentInteractionData:
        interaction_data = self.registry.get_interaction_data(
            self.agent_type, interaction_file
        )
        if not interaction_data:
            raise FileNotFoundError(
                f"Interaction '{interaction_file}' not found for agent '{self.agent_type}'"
            )
        return AgentInteractionData(**interaction_data)

    def load_system_prompt(self) -> str:
        system_prompt = self.registry.get_system_prompt(self.agent_type)
        self.system_message = system_prompt
        return system_prompt

    def build_initial_messages(
        self,
        agent_data: AgentInteractionData,
        attack_instruction: str,
    ) -> List:
        payload_content = agent_data.payload_template.replace(
            "{attack_instruction}", attack_instruction
        )
        return [
            HumanMessage(content=agent_data.user_message),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": agent_data.tool_call.name,
                        "args": agent_data.tool_call.args,
                        "id": agent_data.tool_call.id,
                    }
                ],
            ),
            ToolMessage(
                content=payload_content,
                tool_call_id=agent_data.tool_call.id,
            ),
        ]

    def run_agent(
        self,
        interaction_file: str,
        attack_instruction: str,
        custom_system_prompt: Optional[str] = None,
        initial_messages: Optional[List] = None,
    ) -> Dict[str, Any]:
        agent_data = self.load_agent_data(interaction_file)
        system_prompt = custom_system_prompt or self.load_system_prompt()
        message_list = [SystemMessage(content=system_prompt)]
        if initial_messages is not None:
            message_list += initial_messages
        else:
            message_list += self.build_initial_messages(agent_data, attack_instruction)

        initial_state = AgentState(
            messages=message_list,
            current_tool_call=None,
            final_response=None,
        )
        return self.graph.invoke(initial_state, config={"recursion_limit": 100})

    def list_available_agents(self) -> List[str]:
        return self.registry.get_available_agent_types()

    def list_agent_interactions(self) -> List[str]:
        return self.registry.get_agent_interactions(self.agent_type)

    def get_agent_config_summary(self) -> Dict[str, Any]:
        agent_config = self.registry.get_agent(self.agent_type)
        if not agent_config:
            return {}

        return {
            "agent_type": self.agent_type,
            "name": agent_config.name,
            "description": agent_config.description,
            "category": agent_config.category,
            "status": agent_config.status,
            "enabled": agent_config.is_enabled,
            "interactions": agent_config.get_interaction_names(),
            "tool_count": agent_config.tool_count,
        }


def run_simple_agent(
    agent_type: str,
    attack_instruction: str,
    initial_messages: Optional[List] = None,
    custom_system_prompt: Optional[str] = None,
    config_path: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    agent = SimpleAgent(agent_type, config_path=config_path, **kwargs)
    interaction_file = agent.registry.get_agent_interactions(agent_type)[0]
    return agent.run_agent(
        interaction_file,
        attack_instruction,
        initial_messages=initial_messages,
        custom_system_prompt=custom_system_prompt,
    )
