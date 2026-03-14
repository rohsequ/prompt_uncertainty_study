"""
Configuration loader for the purple agent framework using ConfigParser.
"""

import configparser
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Configuration manager for the purple agent framework."""

    def __init__(self, config_file_path: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_file_path: Path to the config.ini file. If None, uses the default path.
        """
        self.config = configparser.ConfigParser()

        if config_file_path is None:
            # Default config path
            base_dir = Path(__file__).parent.parent.parent
            config_file_path = str(base_dir / "src" / "configs" / "base_config.ini")

        self.config_file_path = config_file_path
        self._load_config()

    def _load_config(self):
        """Load configuration from the INI file."""
        if not Path(self.config_file_path).exists():
            raise FileNotFoundError(f"Config file not found at {self.config_file_path}")

        self.config.read(self.config_file_path)

    def get(self, section: str, option: str, fallback: str = "") -> str:
        """
        Get a configuration value.

        Args:
            section: The section in the config file
            option: The option name
            fallback: Default value if option is not found (default empty string)

        Returns:
            The configuration value or fallback, never None
        """
        value = self.config.get(section, option, fallback=fallback)
        # Ensure we never return None
        return value if value is not None else fallback

    def getint(self, section: str, option: str, fallback: int = 0) -> int:
        """
        Get an integer configuration value.

        Args:
            section: The section in the config file
            option: The option name
            fallback: Default value if option is not found (default 0)

        Returns:
            The configuration value as int or fallback, never None
        """
        value = self.config.getint(section, option, fallback=fallback)
        return value if value is not None else fallback

    def getfloat(self, section: str, option: str, fallback: float = 0.0) -> float:
        """
        Get a float configuration value.

        Args:
            section: The section in the config file
            option: The option name
            fallback: Default value if option is not found (default 0.0)

        Returns:
            The configuration value as float or fallback, never None
        """
        value = self.config.getfloat(section, option, fallback=fallback)
        return value if value is not None else fallback

    def getboolean(self, section: str, option: str, fallback: bool = False) -> bool:
        """
        Get a boolean configuration value.

        Args:
            section: The section in the config file
            option: The option name
            fallback: Default value if option is not found (default False)

        Returns:
            The configuration value as boolean or fallback, never None
        """
        value = self.config.getboolean(section, option, fallback=fallback)
        return value if value is not None else fallback

    def has_option(self, section: str, option: str) -> bool:
        """Check if option exists in the given section."""
        return self.config.has_option(section, option)

    def has_section(self, section: str) -> bool:
        """Check if section exists."""
        return self.config.has_section(section)

    def get_section(self, section: str) -> Dict[str, str]:
        """Get all options in a section as a dictionary."""
        if not self.has_section(section):
            return {}
        return dict(self.config[section])
