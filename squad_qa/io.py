"""JSON input and output shared by the experiment commands."""
import json
import os


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, value):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
