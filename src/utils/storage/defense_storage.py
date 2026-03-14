"""
Defense storage utility for managing the latest approved defense case.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from textwrap import dedent
from src.utils.models import DefenseCase


class DefenseStorage:
    """Manages storage and retrieval of the latest defense case and defense rules."""

    def __init__(
        self,
        storage_path: str,
        default_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        # Initialize config for rules
        self.storage_path = Path(storage_path)
        self.default_path = Path(default_path) if default_path else None
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a directory for storing human-readable defense code files
        self.code_storage_dir = self.storage_path.parent / "defense_code"
        self.code_storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize storage file if it doesn't exist
        if not self.storage_path.exists():
            self._initialize_storage()

    def _initialize_storage(self):
        """Initialize storage file with default defense case if available."""
        if self.default_path and self.default_path.exists():
            # Load from default file
            default_data = self._load_data_from_file(self.default_path)
            self._save_data(default_data)

            # If there's a defense case in the default data, save it as a readable file too
            if default_data.get("defense_case") and default_data["defense_case"].get(
                "defense_case"
            ):
                defense_case_data = default_data["defense_case"]["defense_case"]
                defense_case = DefenseCase(**defense_case_data)
                defense_id = default_data["defense_case"].get(
                    "defense_id", "default_defense"
                )
                self._save_defense_code_file(defense_case, defense_id)
        else:
            # Create empty storage
            initial_data = {
                "defense_case": None,
                "metadata": {"last_updated": None, "version": "1.0.0"},
            }
            self._save_data(initial_data)

    def _load_data_from_file(self, file_path: Path) -> Dict[str, Any]:
        """Load data from a specific file."""
        with open(file_path, "r") as f:
            return json.load(f)

    def _load_data(self) -> Dict[str, Any]:
        """Load data from storage file."""
        return self._load_data_from_file(self.storage_path)

    def _save_data(self, data: Dict[str, Any]):
        """Save data to storage file."""
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_defense_code_file(self, defense_case: DefenseCase, defense_id: str):
        """Save defense function code as a readable .py file."""
        code_file_path = self.code_storage_dir / f"{defense_id}.py"

        # Create a nicely formatted Python file with metadata
        header = dedent(
            f'''"""
                Defense Function: {defense_case.function_name}
                Defense ID: {defense_id}
                Generated: {datetime.now().isoformat()}

                Description:
                {defense_case.description}

                Targeted Attack IDs:
                {', '.join(defense_case.attack_ids) if defense_case.attack_ids else 'None'}
                """

                '''
        )
        if hasattr(defense_case, "function_code"):
            full_content = header + defense_case.function_code
        else:
            full_content = header + "# No function code provided.\n"

        with open(code_file_path, "w") as f:
            f.write(full_content)

    def _load_defense_code_file(self, defense_id: str) -> Optional[str]:
        """Load defense code from the readable .py file (for reference only)."""
        code_file_path = self.code_storage_dir / f"{defense_id}.py"
        if code_file_path.exists():
            with open(code_file_path, "r") as f:
                return f.read()
        return None

    def save_defense_case(
        self, defense_case: DefenseCase, approved: bool = True
    ) -> str:
        """
        Save a new defense case, overwriting the previous one.
        Also saves the function code as a readable .py file.

        Args:
            defense_case: The defense case to save
            approved: Whether the defense case is approved (default: True)

        Returns:
            The defense case ID (timestamp-based)
        """
        if not approved:
            raise ValueError("Only approved defense cases can be saved")

        defense_id = f"defense_{int(datetime.now().timestamp())}"
        timestamp = datetime.now().isoformat()

        # Create defense record with metadata
        defense_record = {
            "defense_id": defense_id,
            "defense_case": defense_case.model_dump(),
            "timestamp": timestamp,
            "approved": approved,
            "status": "active",
        }

        data = {
            "defense_case": defense_record,
            "metadata": {
                "last_updated": timestamp,
                "version": "1.0.0",
                "total_defenses_created": self._get_total_defenses_count() + 1,
            },
        }

        # Save JSON data (for pipeline use)
        self._save_data(data)

        # Save human-readable Python file
        self._save_defense_code_file(defense_case, defense_id)

        return defense_id

    def load_defense_case(self) -> Optional[DefenseCase]:
        """
        Load the current active defense case.

        Returns:
            The current defense case or None if no defense exists
        """
        data = self._load_data()

        if data.get("defense_case") and data["defense_case"].get("defense_case"):
            defense_data = data["defense_case"]["defense_case"]
            return DefenseCase(**defense_data)

        return None

    def get_defense_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the current defense.

        Returns:
            Dictionary containing defense metadata
        """
        data = self._load_data()
        metadata = data.get("metadata", {})

        defense_info = {}
        if data.get("defense_case"):
            defense_info = {
                "defense_id": data["defense_case"].get("defense_id"),
                "timestamp": data["defense_case"].get("timestamp"),
                "approved": data["defense_case"].get("approved"),
                "status": data["defense_case"].get("status"),
                "function_name": (
                    data["defense_case"]["defense_case"].get("function_name")
                    if data["defense_case"].get("defense_case")
                    else None
                ),
            }

        return {**metadata, "current_defense": defense_info}

    def clear_defense_case(self):
        """Clear the current defense case and remove the code file."""
        # Get current defense ID to remove the code file
        metadata = self.get_defense_metadata()
        defense_id = metadata.get("current_defense", {}).get("defense_id")

        if defense_id:
            code_file_path = self.code_storage_dir / f"{defense_id}.py"
            if code_file_path.exists():
                code_file_path.unlink()  # Delete the file

        data = {
            "defense_case": None,
            "metadata": {
                "last_updated": datetime.now().isoformat(),
                "version": "1.0.0",
                "total_defenses_created": self._get_total_defenses_count(),
            },
        }
        self._save_data(data)

    def _get_total_defenses_count(self) -> int:
        """Get the total number of defenses created so far."""
        try:
            data = self._load_data()
            return data.get("metadata", {}).get("total_defenses_created", 0)
        except (FileNotFoundError, json.JSONDecodeError):
            return 0

    def has_active_defense(self) -> bool:
        """
        Check if there's an active defense case.

        Returns:
            True if there's an active defense case, False otherwise
        """
        data = self._load_data()
        return (
            data.get("defense_case") is not None
            and data["defense_case"].get("status") == "active"
            and data["defense_case"].get("approved", False)
        )

    def get_defense_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current defense case.

        Returns:
            Dictionary containing defense summary information
        """
        defense_case = self.load_defense_case()
        metadata = self.get_defense_metadata()

        summary = {"has_defense": defense_case is not None, "metadata": metadata}

        if defense_case:
            defense_id = metadata.get("current_defense", {}).get("defense_id")
            code_file_path = None
            if defense_id:
                code_file_path = str(self.code_storage_dir / f"{defense_id}.py")

            summary["defense_info"] = {
                "function_name": defense_case.function_name,
                "description": defense_case.description,
                "attack_ids_count": len(defense_case.attack_ids),
                "code_length": len(defense_case.function_code),
                "human_readable_code_file": code_file_path,
            }

        return summary

    def get_defense_code_file_path(self) -> Optional[str]:
        """
        Get the path to the human-readable defense code file.

        Returns:
            Path to the .py file containing the defense code, or None if no active defense
        """
        metadata = self.get_defense_metadata()
        defense_id = metadata.get("current_defense", {}).get("defense_id")

        if defense_id:
            code_file_path = self.code_storage_dir / f"{defense_id}.py"
            if code_file_path.exists():
                return str(code_file_path)

        return None
