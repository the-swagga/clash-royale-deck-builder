import json

def get_base_url():
    return "https://api.clashroyale.com/v1"

def load_api_key(path="api_key.txt"):
    with open(path, "r") as f:
        return f.readline()

def save_to_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)