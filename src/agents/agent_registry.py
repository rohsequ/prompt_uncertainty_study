"""
Agent Registry - Centralized agent configuration and management.

This module provides a centralized way to manage agent configurations,
capabilities, and metadata for the simple agent system.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class AgentInteraction:
    """Represents a single interaction for an agent."""

    user_message: str
    tool_call: Dict[str, Any]
    payload_template: str
    tool_response_format: Dict[str, Any]


@dataclass
class AgentConfig:
    """Configuration for a single agent."""

    type: str
    name: str
    description: str
    category: str
    status: str
    interactions: Dict[str, AgentInteraction]
    tool_count: int

    @property
    def is_enabled(self) -> bool:
        """Check if the agent is enabled."""
        return self.status.lower() == "enabled"

    def get_interaction_names(self) -> List[str]:
        """Get list of interaction names."""
        return list(self.interactions.keys())

    def get_interaction(self, name: str) -> Optional[AgentInteraction]:
        """Get a specific interaction by name."""
        return self.interactions.get(name)


class AgentRegistry:
    """
    Centralized registry for managing agent configurations.

    This class loads basic agent info from config and discovers detailed
    information from the directory structure.
    """

    def __init__(self, agents_dir: str = "agents", base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(__file__).parent
        self.agents_dir = self.base_dir / agents_dir

        self._agents: Dict[str, AgentConfig] = {}
        self._categories: Dict[str, List[str]] = {}
        self._tool_specs_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._system_prompts_cache: Dict[str, str] = {}

        self._load_agents()

    def _load_agents(self):
        """Load all agent data from the consolidated agents directory."""
        if not self.agents_dir.exists():
            print(f"⚠️  Agents directory not found: {self.agents_dir}")
            return

        # Load each agent from its directory
        for agent_dir in self.agents_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            agent_type = agent_dir.name
            config_file = agent_dir / "config.json"
            interactions_file = agent_dir / "interactions.json"
            tools_file = agent_dir / "tools.json"

            # Load basic config
            if not config_file.exists():
                print(f"⚠️  Config file not found for {agent_type}: {config_file}")
                continue

            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)

                # Load interactions
                interactions = {}
                if interactions_file.exists():
                    with open(interactions_file, "r", encoding="utf-8") as f:
                        interactions_data = json.load(f)
                        for interaction_data in interactions_data:
                            interaction_name = interaction_data.get(
                                "tool_call", {}
                            ).get("name", "unknown")
                            interactions[interaction_name] = AgentInteraction(
                                user_message=interaction_data.get("user_message", ""),
                                tool_call=interaction_data.get("tool_call", {}),
                                payload_template=interaction_data.get(
                                    "payload_template", ""
                                ),
                                tool_response_format=interaction_data.get(
                                    "tool_response_format", {}
                                ),
                            )

                # Load tool count
                tool_count = 0
                if tools_file.exists():
                    with open(tools_file, "r", encoding="utf-8") as f:
                        tools_data = json.load(f)
                        tool_count = len(tools_data.get("tools", []))

                # Create agent config
                agent_config = AgentConfig(
                    type=agent_type,
                    name=config_data["name"],
                    description=config_data["description"],
                    category=config_data["category"],
                    status=config_data["status"],
                    interactions=interactions,
                    tool_count=tool_count,
                )

                self._agents[agent_type] = agent_config

                # Add to category
                category = config_data["category"]
                if category not in self._categories:
                    self._categories[category] = []
                self._categories[category].append(agent_type)

            except Exception as e:
                print(f"❌ Error loading agent {agent_type}: {e}")

        print(f"✅ Loaded {len(self._agents)} agents from consolidated structure")

    def get_agent(self, agent_type: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent."""
        return self._agents.get(agent_type)

    def get_all_agents(self) -> Dict[str, AgentConfig]:
        """Get all agent configurations."""
        return self._agents.copy()

    def get_enabled_agents(self) -> Dict[str, AgentConfig]:
        """Get only enabled agents."""
        return {k: v for k, v in self._agents.items() if v.is_enabled}

    def get_available_agent_types(self) -> List[str]:
        """Get list of all available agent types."""
        return list(self._agents.keys())

    def get_enabled_agent_types(self) -> List[str]:
        """Get list of enabled agent types."""
        return [k for k, v in self._agents.items() if v.is_enabled]

    def get_agent_interactions(self, agent_type: str) -> List[str]:
        """Get list of interactions for a specific agent."""
        agent = self.get_agent(agent_type)
        if not agent:
            return []
        return agent.get_interaction_names()

    def get_agents_by_category(self, category: str) -> List[str]:
        """Get agent types for a specific category."""
        return self._categories.get(category, [])

    def get_all_categories(self) -> Dict[str, List[str]]:
        """Get all categories and their agents."""
        return self._categories.copy()

    def get_agent_summary(self, agent_type: str) -> Optional[Dict[str, Any]]:
        """Get a summary of agent information."""
        agent = self.get_agent(agent_type)
        if not agent:
            return None

        return {
            "type": agent.type,
            "name": agent.name,
            "description": agent.description,
            "category": agent.category,
            "status": agent.status,
            "interactions_count": len(agent.interactions),
            "tool_count": agent.tool_count,
            "interactions": list(agent.interactions.keys()),
        }

    def get_full_summary(self) -> Dict[str, Any]:
        """Get a full summary of all agents and metadata."""
        agent_summaries = {}
        for agent_type, agent in self._agents.items():
            agent_summaries[agent_type] = self.get_agent_summary(agent_type)

        return {
            "categories": self._categories,
            "agents": agent_summaries,
            "enabled_agents": self.get_enabled_agent_types(),
            "total_agents": len(self._agents),
            "enabled_count": len(self.get_enabled_agents()),
        }

    def validate_agent_exists(self, agent_type: str) -> bool:
        """Check if an agent exists."""
        return agent_type in self._agents

    def validate_interaction_exists(self, agent_type: str, interaction: str) -> bool:
        """Check if an interaction exists for an agent."""
        agent = self.get_agent(agent_type)
        if not agent:
            return False
        return interaction in agent.interactions

    def get_interaction_data(
        self, agent_type: str, interaction_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get the full interaction data for a specific interaction."""
        agent = self.get_agent(agent_type)
        if not agent:
            return None

        interaction_obj = agent.get_interaction(interaction_name)
        if not interaction_obj:
            return None

        return {
            "user_message": interaction_obj.user_message,
            "tool_call": interaction_obj.tool_call,
            "payload_template": interaction_obj.payload_template,
            "tool_response_format": interaction_obj.tool_response_format,
        }

    def get_tool_specs(self, agent_type: str) -> List[Dict[str, Any]]:
        """Get tool specifications for an agent type."""
        # Check cache first
        if agent_type in self._tool_specs_cache:
            return self._tool_specs_cache[agent_type]

        # Load from consolidated structure
        tool_spec_file = self.agents_dir / agent_type / "tools.json"
        if not tool_spec_file.exists():
            return []

        try:
            with open(tool_spec_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                tool_specs = data.get("tools", [])

            # Cache the result
            self._tool_specs_cache[agent_type] = tool_specs
            return tool_specs
        except Exception as e:
            print(f"Error loading tool specs for {agent_type}: {e}")
            return []

    def get_system_prompt(self, agent_type: str) -> str:
        """Get system prompt for an agent type."""
        # Check cache first
        if agent_type in self._system_prompts_cache:
            return self._system_prompts_cache[agent_type]

        # Load from consolidated structure
        system_prompt_file = self.agents_dir / agent_type / "system_prompt.txt"
        if not system_prompt_file.exists():
            return ""

        try:
            with open(system_prompt_file, "r", encoding="utf-8") as f:
                system_prompt = f.read().strip()

            # Cache the result
            self._system_prompts_cache[agent_type] = system_prompt
            return system_prompt
        except Exception as e:
            print(f"Error loading system prompt for {agent_type}: {e}")
            return ""

    def reload_config(self):
        """Reload the configuration from file."""
        self._agents.clear()
        self._categories.clear()
        self._load_agents()

    def __str__(self) -> str:
        """String representation of the registry."""
        enabled = len(self.get_enabled_agents())
        total = len(self._agents)
        return f"AgentRegistry: {enabled}/{total} agents enabled"

    def __repr__(self) -> str:
        return f"AgentRegistry(agents={list(self._agents.keys())})"


# Global registry instance
_agent_registry = None


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry instance."""
    global _agent_registry
    if _agent_registry is None:
        _agent_registry = AgentRegistry()
    return _agent_registry
