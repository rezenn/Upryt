import pickle
import os
from datetime import datetime

PROFILE_PATH = "user_profile.pkl"
HISTORY_PATH = "posture_history.pkl"


def load_user_profile():
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, "rb") as f:
            return pickle.load(f)
    else:
        return {
            "username": "DefaultUser",
            "language": "english",
            "preferred_posture_model": "v1",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


def save_user_profile(profile):
    with open(PROFILE_PATH, "wb") as f:
        pickle.dump(profile, f)


def load_posture_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "rb") as f:
            return pickle.load(f)
    else:
        return {
            "timestamp": [],
            "classification": [],
            "corrections": [],
            "shoulder_angles": [],
            "neck_angles": []
        }


def update_posture_history(history, classification, correction, shoulder_angle, neck_angle):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history["timestamp"].append(now)
    history["classification"].append(classification)
    history["corrections"].append(correction)
    history["shoulder_angles"].append(shoulder_angle)
    history["neck_angles"].append(neck_angle)


def save_posture_history(history):
    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(history, f)
