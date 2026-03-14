from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)


# Input Data Model
class InputData(BaseModel):
    """Input data containing user prompt, email template and framework data path."""

    user_prompt: str = Field(description="The benign user prompt")
    payload_template: str = Field(
        description="Email template with attacker_instructions placeholder"
    )
    framework_data_path: str = Field(description="Path to framework data file")
    defense_code: Optional[str] = Field(
        default=None, description="Python script to scan for attacks"
    )


# Attack Evaluation Models
class AttackEvaluationResponse(BaseModel):
    success_flag: bool = Field(description="Whether the attack was successful")
    rating: int = Field(
        description="Attack effectiveness rating from 1-10", ge=1, le=10
    )
    reasoning: str = Field(
        description="A concise explanation for the success or failure of the attack."
    )
    attack_id: Optional[str] = Field(
        default=None, description="Unique identifier for the attack"
    )
    defense_blocked: Optional[bool] = Field(
        default=None, description="Whether the attack was blocked by a defense"
    )


# Attack Models
class AttackScenario(BaseModel):
    attacker_tools: List[str] = Field(description="List of tools the attacker can use")
    attacker_instruction: str = Field(
        description="The specific instruction issued by the attacker"
    )
    expected_achievements: str = Field(
        description="The outcome the attacker aims to accomplish"
    )
    attack_type: str = Field(description="Category of the attacker's goal")
    thought: str = Field(description="The agent's internal reasoning about the task")
    target_vulnerabilities: List[str] = Field(
        default_factory=list, description="Vulnerabilities this attack targets"
    )


# Target Agent Response Model
class TargetAgentResponse(BaseModel):
    """Response from target agent containing messages and final response."""

    messages: List[BaseMessage] = Field(
        description="Complete message trace from target agent"
    )
    agent_response: str = Field(
        description="Final response content from the target agent"
    )
    tool_calls_made: List[str] = Field(
        default_factory=list, description="List of tool calls made by the target agent"
    )


# Attack/Defense Storage Models
class AttackRecord(BaseModel):
    """Record of an attack for storage and tracking."""

    attack_id: str = Field(description="Unique identifier for the attack")
    attack_scenario: AttackScenario = Field(description="The attack scenario")
    is_successful: bool = Field(description="Whether the attack was successful")
    defense_available: bool = Field(
        default=False, description="Whether a defense exists for this attack"
    )
    timestamp: str = Field(description="When the attack was recorded")
    evaluation_details: Dict[str, Any] = Field(description="Evaluation details")
    target_agent_response: Optional[TargetAgentResponse] = Field(
        default=None, description="Response from target agent"
    )
    defense_blocked: Optional[bool] = Field(
        default=None, description="Whether the defense blocked the attack"
    )

    def model_dump(self, **kwargs):
        """Custom serialization to handle TargetAgentResponse with BaseMessage objects properly."""
        data = super().model_dump(**kwargs)

        # Handle TargetAgentResponse serialization manually
        if "target_agent_response" in data:
            target_response = self.target_agent_response

            # Serialize messages manually
            messages_data = []
            for msg in target_response.messages:
                # Use BaseMessage's own serialization method
                if hasattr(msg, "model_dump"):
                    msg_data = msg.model_dump()
                elif hasattr(msg, "dict"):
                    msg_data = msg.dict()
                else:
                    # Fallback for older versions
                    msg_data = {
                        "type": msg.type,
                        "content": msg.content,
                        "name": getattr(msg, "name", None),
                    }
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        msg_data["tool_calls"] = msg.tool_calls
                    if hasattr(msg, "tool_call_id"):
                        msg_data["tool_call_id"] = msg.tool_call_id

                messages_data.append(msg_data)

            # Create the target_agent_response structure
            data["target_agent_response"] = {
                "messages": messages_data,
                "agent_response": target_response.agent_response,
                "tool_calls_made": target_response.tool_calls_made,
            }

        return data

    @classmethod
    def model_validate(cls, data: Any, **kwargs):
        """Custom deserialization to properly reconstruct TargetAgentResponse with BaseMessage objects."""
        if isinstance(data, dict) and "target_agent_response" in data:
            target_response_data = data["target_agent_response"]

            if target_response_data and "messages" in target_response_data:
                # Reconstruct BaseMessage objects from serialized data
                reconstructed_messages = []
                for msg_data in target_response_data["messages"]:
                    if isinstance(msg_data, dict):
                        # Determine the correct BaseMessage subclass based on the type
                        msg_type = msg_data.get("type", "")

                        if msg_type == "human":
                            message = HumanMessage(
                                content=msg_data.get("content", ""),
                                additional_kwargs=msg_data.get("additional_kwargs", {}),
                            )
                        elif msg_type == "ai":
                            message = AIMessage(
                                content=msg_data.get("content", ""),
                                additional_kwargs=msg_data.get("additional_kwargs", {}),
                                tool_calls=msg_data.get("tool_calls", []),
                            )
                        elif msg_type == "tool":
                            message = ToolMessage(
                                content=msg_data.get("content", ""),
                                tool_call_id=msg_data.get("tool_call_id", ""),
                                name=msg_data.get("name", ""),
                                additional_kwargs=msg_data.get("additional_kwargs", {}),
                            )
                        elif msg_type == "system":
                            message = SystemMessage(
                                content=msg_data.get("content", ""),
                                additional_kwargs=msg_data.get("additional_kwargs", {}),
                            )
                        else:
                            # Fallback to BaseMessage for unknown types
                            message = BaseMessage(
                                content=msg_data.get("content", ""),
                                type=msg_type,
                                additional_kwargs=msg_data.get("additional_kwargs", {}),
                            )

                        reconstructed_messages.append(message)

                # Create TargetAgentResponse with reconstructed messages
                target_response = TargetAgentResponse(
                    messages=reconstructed_messages,
                    agent_response=target_response_data.get("agent_response", ""),
                    tool_calls_made=target_response_data.get("tool_calls_made", []),
                )

                # Replace the serialized data with the proper object
                data["target_agent_response"] = target_response

        return super().model_validate(data, **kwargs)


class UtilityRecord(BaseModel):
    """Record of a utility evaluation case."""

    utility_id: str = Field(description="Unique identifier for the utility case")
    utility_description: str = Field(
        description="Description of what the utility case tests"
    )
    category: str = Field(
        description="Category of the utility (Information-Retrieval, Action-Oriented, Synthetic)"
    )
    target_agent_response: TargetAgentResponse = Field(
        description="Response from target agent"
    )

    def model_dump(self, **kwargs):
        """Custom serialization to handle TargetAgentResponse with BaseMessage objects properly."""
        data = super().model_dump(**kwargs)

        # Handle TargetAgentResponse serialization manually
        if "target_agent_response" in data:
            target_response = self.target_agent_response

            # Serialize messages manually
            messages_data = []
            for msg in target_response.messages:
                # Use BaseMessage's own serialization method
                if hasattr(msg, "model_dump"):
                    msg_data = msg.model_dump()
                elif hasattr(msg, "dict"):
                    msg_data = msg.dict()
                else:
                    # Fallback for older versions
                    msg_data = {
                        "type": msg.type,
                        "content": msg.content,
                        "name": getattr(msg, "name", None),
                    }
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        msg_data["tool_calls"] = msg.tool_calls
                    if hasattr(msg, "tool_call_id"):
                        msg_data["tool_call_id"] = msg.tool_call_id

                messages_data.append(msg_data)

            # Create the target_agent_response structure
            data["target_agent_response"] = {
                "messages": messages_data,
                "agent_response": target_response.agent_response,
                "tool_calls_made": target_response.tool_calls_made,
            }

        return data


class UtilityDataset(BaseModel):
    """Complete utility evaluation dataset."""

    utility: List[UtilityRecord] = Field(
        description="List of utility evaluation records"
    )

    def add_utility(self, utility: UtilityRecord) -> None:
        """Add a utility record to the dataset."""
        self.utility.append(utility)

    def extend_utilities(self, utilities: List[UtilityRecord]) -> None:
        """Add multiple utility records to the dataset."""
        self.utility.extend(utilities)

    @property
    def count(self) -> int:
        """Get the number of utility records."""
        return len(self.utility)

    def filter_by_category(self, category: str) -> "UtilityDataset":
        """Filter utility records by category."""
        filtered_utilities = [
            utility for utility in self.utility if utility.category == category
        ]
        return UtilityDataset(utility=filtered_utilities)

    def get_by_id(self, utility_id: str) -> Optional[UtilityRecord]:
        """Get a utility record by ID."""
        for utility in self.utility:
            if utility.utility_id == utility_id:
                return utility
        return None

    def model_dump(self, **kwargs):
        """Custom serialization to ensure UtilityRecord custom serialization is called."""
        data = {"utility": [utility.model_dump(**kwargs) for utility in self.utility]}
        return data


# Defense Models
class DefenseCase(BaseModel):
    """Generated defense case with optional function code and rule-based components."""

    function_name: Optional[str] = Field(description="Name of the defense function")
    function_code: Optional[str] = Field(
        description="Python code for the defense function"
    )
    description: str = Field(description="Description of what the defense does")
    attack_ids: Optional[List[str]] = Field(
        description="List of attack IDs this defense targets"
    )
    # generated_rules: Optional[RuleBasedDefenseRules] = Field(
    #     default=None, description="Generated defense rules"
    # )
    # rules_file_paths: Dict[str, str] = Field(
    #     default_factory=dict, description="Updated rule file paths"
    # )


class DefenseEvaluation(BaseModel):
    safe: bool
    reason: str
    attack_id: Optional[str] = Field(
        default=None, description="ID of the attack being evaluated"
    )
    attack_success: Optional[bool] = Field(
        default=None,
        description="Whether the original attack succeeded against the target agent",
    )
    utility_evaluation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Results of utility dataset evaluation for this defense",
    )


# Agent State Model
class PurpleAgentState(BaseModel):
    """Complete state for the purple agent workflow."""

    # Input data
    input_data: Optional[InputData] = Field(
        default=None, description="Input data for the workflow"
    )

    # 1. To hold the current attack being processed in a parallel branch.
    current_attack_scenario: Optional[AttackScenario] = Field(
        default=None,
        description="The single attack scenario for the current parallel branch.",
    )
    # 2. To aggregate the final results from all parallel branches.
    attack_results: List[AttackRecord] = Field(
        default_factory=list,
        description="Aggregated results (AttackRecord) from all parallel attack runs.",
    )

    # Target agent response
    target_agent_response: Optional[TargetAgentResponse] = Field(
        default=None, description="Response from target agent"
    )

    # Evaluation results
    evaluation_results: Optional[AttackEvaluationResponse] = Field(
        default=None, description="Attack evaluation results"
    )

    # Defense cases
    defense_cases: Optional[DefenseCase] = Field(
        default=None, description="Generated defense function"
    )
