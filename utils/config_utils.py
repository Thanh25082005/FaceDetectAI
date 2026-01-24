import json
from pathlib import Path

# Default configuration values
DEFAULT_CONFIG = {
    "COMPANY_LOCATION": [21.0285, 105.8542],
    "MAX_CHECKIN_DISTANCE": 1000
}

# Path to the dynamic configuration file
DYNAMIC_CONFIG_PATH = Path(__file__).parent.parent / "data" / "config.json"

def load_dynamic_config():
    """Load dynamic configuration from JSON file."""
    if not DYNAMIC_CONFIG_PATH.exists():
        save_dynamic_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG
    
    try:
        with open(DYNAMIC_CONFIG_PATH, "r") as f:
            data = json.load(f)
            # Ensure all default keys exist
            updated = False
            for k, v in DEFAULT_CONFIG.items():
                if k not in data:
                    data[k] = v
                    updated = True
            if updated:
                save_dynamic_config(data)
            return data
    except Exception as e:
        print(f"⚠️ Error loading dynamic config: {e}")
        return {}

def save_dynamic_config(config_data):
    """Save dynamic configuration to JSON file."""
    # Ensure data directory exists
    DYNAMIC_CONFIG_PATH.parent.mkdir(exist_ok=True)
    
    try:
        with open(DYNAMIC_CONFIG_PATH, "w") as f:
            json.dump(config_data, f, indent=4)
        return True
    except Exception as e:
        print(f"⚠️ Error saving dynamic config: {e}")
        return False

def update_config_value(key, value):
    """Update a specific configuration value."""
    config = load_dynamic_config()
    config[key] = value
    return save_dynamic_config(config)
