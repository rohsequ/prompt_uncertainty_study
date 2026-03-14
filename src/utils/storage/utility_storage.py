"""
Utility storage utility for managing utility evaluation datasets.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.utils.models import TargetAgentResponse, UtilityRecord, UtilityDataset


class UtilityStorage:
    """Manages storage and retrieval of utility evaluation datasets."""

    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize storage file if it doesn't exist
        if not self.storage_path.exists():
            self._initialize_storage()

    def _initialize_storage(self):
        """Initialize empty storage file."""
        initial_data = UtilityDataset(utility=[])
        with open(self.storage_path, "w") as f:
            json.dump(initial_data.model_dump(), f, indent=2)

    def load_dataset(self) -> UtilityDataset:
        """Load the complete utility dataset from storage."""
        with open(self.storage_path, "r") as f:
            data = json.load(f)
        return data

    def save_dataset(self, dataset: UtilityDataset):
        """Save the complete utility dataset to storage."""
        with open(self.storage_path, "w") as f:
            json.dump(dataset.model_dump(), f, indent=2)

    def add_utility_record(
        self,
        utility_id: str,
        utility_description: str,
        category: str,
        target_agent_response: TargetAgentResponse,
    ) -> None:
        """
        Add a new utility record to storage.

        Args:
            utility_id: Unique identifier for the utility case
            utility_description: Description of what the utility case tests
            category: Category of the utility
            target_agent_response: TargetAgentResponse object with messages and final response
        """
        utility_record = UtilityRecord(
            utility_id=utility_id,
            utility_description=utility_description,
            category=category,
            target_agent_response=target_agent_response,
        )

        dataset = self.load_dataset()
        dataset = UtilityDataset(**dataset)
        dataset.add_utility(utility_record)
        self.save_dataset(dataset)

    def get_utilities_by_category(self, category: str) -> List[UtilityRecord]:
        """Get all utility records for a specific category."""
        dataset = self.load_dataset()
        dataset = UtilityDataset(**dataset)
        return dataset.filter_by_category(category).utility

    def get_all_utilities(self) -> List[UtilityRecord]:
        """Get all utility records."""
        dataset = self.load_dataset()
        dataset = UtilityDataset(**dataset)
        return dataset.utility

    def get_all_utilities_raw(self) -> List[Dict[str, Any]]:
        """Get raw data of all utility records."""
        dataset = self.load_dataset()
        return dataset["utility"]

    def get_utility_by_id(self, utility_id: str) -> Optional[UtilityRecord]:
        """Get a specific utility record by ID."""
        dataset = self.load_dataset()
        dataset = UtilityDataset(**dataset)
        return dataset.get_by_id(utility_id)

    @classmethod
    def from_json_file(cls, json_file_path: str, storage_path: str) -> "UtilityStorage":
        """
        Create a UtilityStorage instance and load data from an existing JSON file.

        Args:
            json_file_path: Path to the existing JSON file with utility data
            storage_path: Path where the storage file should be created/updated

        Returns:
            UtilityStorage instance with loaded data
        """
        # Load the existing JSON data
        with open(json_file_path, "r") as f:
            data = json.load(f)

        # Convert to UtilityDataset
        dataset = UtilityDataset(**data)

        # Create storage instance
        storage = cls(storage_path)

        # Save the dataset
        storage.save_dataset(dataset)

        return storage

    @classmethod
    def create_from_dict_data(
        cls, json_data: Dict[str, Any], storage_path: str
    ) -> "UtilityStorage":
        """
        Create a UtilityStorage instance from dictionary data (e.g., loaded JSON).

        Args:
            json_data: Dictionary containing utility data in the expected format
            storage_path: Path where the storage file should be created/updated

        Returns:
            UtilityStorage instance with loaded data
        """
        from langchain_core.messages import (
            SystemMessage,
            HumanMessage,
            AIMessage,
            ToolMessage,
        )

        # Create storage instance
        storage = cls(storage_path)

        # Convert the dictionary data to UtilityDataset
        utilities = []
        for utility_data in json_data.get("utility", []):
            # Convert message dictionaries back to BaseMessage objects
            messages = []
            for msg_data in utility_data["target_agent_response"]["messages"]:
                msg_type = msg_data["type"]

                if msg_type == "SystemMessage":
                    msg = SystemMessage(
                        content=msg_data["content"], name=msg_data.get("name")
                    )
                elif msg_type == "HumanMessage":
                    msg = HumanMessage(
                        content=msg_data["content"], name=msg_data.get("name")
                    )
                elif msg_type == "AIMessage":
                    # Handle tool_calls properly - they can be in the message data directly
                    tool_calls = msg_data.get("tool_calls", [])
                    msg = AIMessage(
                        content=msg_data["content"],
                        name=msg_data.get("name"),
                        tool_calls=tool_calls,
                    )
                elif msg_type == "ToolMessage":
                    msg = ToolMessage(
                        content=msg_data["content"],
                        name=msg_data.get("name"),
                        tool_call_id=msg_data.get("tool_call_id"),
                    )
                else:
                    # Fallback for unknown message types
                    msg = SystemMessage(content=msg_data["content"])

                messages.append(msg)

            # Create TargetAgentResponse
            target_agent_response_data = utility_data["target_agent_response"]
            target_response = TargetAgentResponse(
                messages=messages,
                agent_response=target_agent_response_data.get("agent_response", ""),
                tool_calls_made=target_agent_response_data.get("tool_calls_made", []),
            )

            # Create UtilityRecord
            utility_record = UtilityRecord(
                utility_id=utility_data["utility_id"],
                utility_description=utility_data["utility_description"],
                category=utility_data["category"],
                target_agent_response=target_response,
            )
            utilities.append(utility_record)

        # Create and save dataset
        dataset = UtilityDataset(utility=utilities)
        storage.save_dataset(dataset)

        return storage
