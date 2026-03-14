"""
Attack storage utility for managing successful and failed attacks.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.utils.models import (
    AttackRecord,
    AttackScenario,
    AttackEvaluationResponse,
)


class AttackStorage:
    """Manages storage and retrieval of attack records."""

    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize storage file if it doesn't exist
        if not self.storage_path.exists():
            initial_data = {"successful_attacks": [], "failed_attacks": []}
            with open(self.storage_path, "w") as f:
                json.dump(initial_data, f, indent=2)

        # Load existing data
        self.data_store = self._load_data()

        # Track the current attack ID
        self.current_attack_id = None

    def _load_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load data from storage file."""
        with open(self.storage_path, "r") as f:
            return json.load(f)

    def _save_data(self, data: Dict[str, List[Dict[str, Any]]]):
        """Save data to storage file."""
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_attack_record(
        self,
        attack_scenario: AttackScenario,
        evaluation: AttackEvaluationResponse,
        target_agent_response=None,
    ) -> str:
        """
        Add a new attack record to storage.

        Args:
            attack_scenario: The attack scenario
            evaluation: The evaluation results
            target_agent_response: The response from the target agent

        Returns:
            The attack ID
        """
        attack_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        record = AttackRecord(
            attack_id=attack_id,
            attack_scenario=attack_scenario,
            is_successful=evaluation.success_flag,
            defense_available=False,
            timestamp=timestamp,
            evaluation_details={"reasoning": evaluation.reasoning},
            target_agent_response=target_agent_response,
        )

        if evaluation.success_flag:
            self.data_store["successful_attacks"].append(record.model_dump())
        else:
            self.data_store["failed_attacks"].append(record.model_dump())
            # Keep only the top 10 most recent failed attempts
            if len(self.data_store["failed_attacks"]) > 10:
                self.data_store["failed_attacks"] = self.data_store["failed_attacks"][
                    -10:
                ]

        self._save_data(self.data_store)

        # Store the current attack ID
        self.current_attack_id = attack_id
        return attack_id

    def update_defense_status(
        self, attack_id: Optional[str] = None, defense_available: bool = True
    ):
        """
        Update the defense status for an attack.

        Args:
            attack_id: The attack ID to update. If None, uses the current attack ID.
            defense_available: Whether defense is available
        """
        # Use current_attack_id if attack_id is not provided
        if attack_id is None:
            attack_id = self.current_attack_id
            if attack_id is None:
                raise ValueError(
                    "No attack ID provided and no current attack ID available"
                )

        # Update in successful attacks
        for attack in self.data_store["successful_attacks"]:
            if attack["attack_id"] == attack_id:
                attack["defense_available"] = defense_available
                if defense_available:
                    # Move from successful to failed if defense is created
                    attack["is_successful"] = False
                    self.data_store["failed_attacks"].append(attack)
                    self.data_store["successful_attacks"].remove(attack)
                break

        self._save_data(self.data_store)

    def get_successful_attacks(self) -> List[AttackRecord]:
        """Get all successful attacks."""
        return [
            AttackRecord(**record) for record in self.data_store["successful_attacks"]
        ]

    def get_successful_attacks_raw(self) -> List[Dict[str, Any]]:
        """Get raw data of all successful attacks."""
        return self.data_store["successful_attacks"]

    def get_failed_attacks(self) -> List[AttackRecord]:
        """Get all failed attacks."""
        return [AttackRecord(**record) for record in self.data_store["failed_attacks"]]

    def get_current_attack_id(self) -> Optional[str]:
        """
        Get the ID of the most recently created attack.

        Returns:
            The current attack ID or None if no attacks have been created
        """
        return self.current_attack_id

    def get_attack_examples_for_feedback(self) -> Dict[str, List[AttackRecord]]:
        """Get attack examples for feedback generation."""
        return {
            "successful": self.get_successful_attacks(),
            "failed": self.get_failed_attacks(),
        }
