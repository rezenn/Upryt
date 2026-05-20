import json
import time
from datetime import datetime
import os


def ensure_dir(directory):
    """Ensure directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def format_duration(seconds):
    """Convert seconds to human-readable time"""
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {seconds}s"


def save_json(data, filepath):
    """Save data to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath):
    """Load data from JSON file"""
    if os.path.exists(filepath):
        with open(filepath, encoding='utf-8') as f:
            return json.load(f)
    return None
