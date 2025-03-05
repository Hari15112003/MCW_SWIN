import json

class ConfigManager:
    _config = None  # Singleton storage

    @classmethod
    def load_config(cls, config_path):
        """Loads the configuration from a JSON file only once."""
        if cls._config is None:
            print(f"Loading config from: {config_path}")
            try:
                with open(config_path, "r") as f:
                    cls._config = json.load(f)
                if not isinstance(cls._config, dict):
                    raise ValueError("Invalid config format")
                print("Config loaded successfully!")
            except Exception as e:
                print(f"Error loading config: {e}")
                cls._config = None
        return cls._config

    @classmethod
    def get_config(cls):
        """Returns the global config."""
        if cls._config is None:
            raise RuntimeError("Config is not loaded! Call load_config() first.")
        return cls._config
