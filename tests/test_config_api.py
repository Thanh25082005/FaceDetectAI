import requests
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path to import config
sys.path.append(str(Path(__file__).parent.parent))

BASE_URL = "http://localhost:8000/api/v1"

def test_config_workflow():
    print("--- Testing Config Workflow ---")
    
    # 1. Get initial config
    print("\n1. GET /config")
    try:
        response = requests.get(f"{BASE_URL}/config")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        initial_config = response.json()
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Update config
    print("\n2. POST /config")
    new_location = [21.03, 105.86]
    new_distance = 500
    payload = {
        "company_location": new_location,
        "max_checkin_distance": new_distance
    }
    try:
        response = requests.post(f"{BASE_URL}/config", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
        return

    # 3. Verify update
    print("\n3. GET /config (Verify)")
    try:
        response = requests.get(f"{BASE_URL}/config")
        data = response.json()
        print(f"Response: {data}")
        if data["company_location"] == new_location and data["max_checkin_distance"] == new_distance:
            print("✅ Config updated successfully in memory")
        else:
            print("❌ Config update failed")
    except Exception as e:
        print(f"Error: {e}")
        return

    # 4. Check persistence file
    print("\n4. Check persistence file (data/config.json)")
    config_file = Path("data/config.json")
    if config_file.exists():
        with open(config_file, "r") as f:
            persisted_data = json.load(f)
            print(f"File content: {persisted_data}")
            if persisted_data.get("COMPANY_LOCATION") == new_location and persisted_data.get("MAX_CHECKIN_DISTANCE") == new_distance:
                print("✅ Config persisted correctly to file")
            else:
                print("❌ Config persistence failed")
    else:
        print("❌ Persistence file not found")

if __name__ == "__main__":
    test_config_workflow()
