from pathlib import Path
import json

def get_base_url():
    return "https://api.clashroyale.com/v1"

def load_api_key():
    api_key_path = Path(__file__).resolve().parent / "api_key.txt"
    with open(api_key_path, "r") as f:
        return f.readline()

def save_to_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)