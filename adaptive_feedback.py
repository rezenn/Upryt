import numpy as np
import joblib
import time
import os
from collections import deque
import matplotlib.pyplot as plt


class AdaptivePostureFeedback:
    def __init__(self, user_id="default", window_size=100, save_interval=50):
        self.user_id = user_id
        self.profile_file = f"user_{user_id}_posture_profile.pkl"
        self.history_file = f"user_{user_id}_posture_history.pkl"
        self.posture_history = deque(maxlen=window_size)
        self.save_interval = save_interval
        self.counter = 0

        # Initialize with reasonable defaults
        self.user_profile = {
            'shoulder_angle': {'mean': 90, 'std': 5, 'min': 80, 'max': 100},
            'neck_angle': {'mean': 30, 'std': 5, 'min': 20, 'max': 40},
            'spine_angle': {'mean': 145, 'std': 5, 'min': 135, 'max': 155},
            'symmetry_diff': {'mean': 5, 'std': 3, 'min': 0, 'max': 10}
        }

        # Load existing data if available
        self.load_data()

    def load_data(self):
        """Load user profile and history from files"""
        if os.path.exists(self.profile_file):
            self.user_profile = joblib.load(self.profile_file)
        if os.path.exists(self.history_file):
            self.posture_history = joblib.load(self.history_file)

    def save_data(self):
        """Save current profile and history to files"""
        joblib.dump(self.user_profile, self.profile_file)
        joblib.dump(self.posture_history, self.history_file)

    def update_profile(self, angles):
        """Update profile with new posture data"""
        shoulder, neck, spine, symmetry = angles

        self.posture_history.append({
            'timestamp': time.time(),
            'shoulder': shoulder,
            'neck': neck,
            'spine': spine,
            'symmetry': symmetry
        })

        self.counter += 1
        if self.counter % self.save_interval == 0:
            self._calculate_profile()
            self.save_data()

    def _calculate_profile(self):
        """Recalculate profile statistics from history"""
        if len(self.posture_history) < 10:  # Need minimum data points
            return

        shoulders = [p['shoulder'] for p in self.posture_history]
        necks = [p['neck'] for p in self.posture_history]
        spines = [p['spine'] for p in self.posture_history]
        symmetries = [p['symmetry'] for p in self.posture_history]

        # Update statistics for each measurement
        for name, values in zip(['shoulder_angle', 'neck_angle', 'spine_angle', 'symmetry_diff'],
                                [shoulders, necks, spines, symmetries]):
            self.user_profile[name]['mean'] = np.mean(values)
            self.user_profile[name]['std'] = np.std(values)
            self.user_profile[name]['min'] = min(values)
            self.user_profile[name]['max'] = max(values)

    def get_personalized_feedback(self, current_angles):
        """Generate easy-to-understand feedback"""
        shoulder, neck, spine, symmetry = current_angles
        feedback = []

        # Shoulder feedback
        s_mean = self.user_profile['shoulder_angle']['mean']
        if shoulder < s_mean - 1.5*self.user_profile['shoulder_angle']['std']:
            feedback.append("Try raising your left shoulder slightly")
        elif shoulder > s_mean + 1.5*self.user_profile['shoulder_angle']['std']:
            feedback.append("Try lowering your right shoulder slightly")

        # Neck feedback
        n_mean = self.user_profile['neck_angle']['mean']
        if neck < n_mean - 1.5*self.user_profile['neck_angle']['std']:
            feedback.append("Gently tuck your chin to straighten neck")

        # Spine feedback
        sp_mean = self.user_profile['spine_angle']['mean']
        if spine < sp_mean - 1.5*self.user_profile['spine_angle']['std']:
            feedback.append("Engage your core to straighten your back")

        # Symmetry feedback
        sym_mean = self.user_profile['symmetry_diff']['mean']
        if symmetry > sym_mean + 2*self.user_profile['symmetry_diff']['std']:
            feedback.append("Adjust your hips to balance your posture")

        return feedback

    def visualize_profile(self, save_path=None):
        """Generate visualization of posture patterns"""
        if not self.posture_history:
            return None

        timestamps = [p['timestamp'] for p in self.posture_history]
        shoulders = [p['shoulder'] for p in self.posture_history]
        necks = [p['neck'] for p in self.posture_history]
        spines = [p['spine'] for p in self.posture_history]

        plt.figure(figsize=(12, 8))

        # Shoulder angle plot
        plt.subplot(3, 1, 1)
        plt.plot(timestamps, shoulders, label='Shoulder Angle')
        plt.axhline(self.user_profile['shoulder_angle']['mean'],
                    color='r', linestyle='--', label='Your Average')
        plt.ylabel('Degrees')
        plt.title('Your Shoulder Posture Over Time')
        plt.legend()

        # Neck angle plot
        plt.subplot(3, 1, 2)
        plt.plot(timestamps, necks, label='Neck Angle')
        plt.axhline(self.user_profile['neck_angle']['mean'],
                    color='r', linestyle='--', label='Your Average')
        plt.ylabel('Degrees')
        plt.title('Your Neck Posture Over Time')
        plt.legend()

        # Spine angle plot
        plt.subplot(3, 1, 3)
        plt.plot(timestamps, spines, label='Spine Angle')
        plt.axhline(self.user_profile['spine_angle']['mean'],
                    color='r', linestyle='--', label='Your Average')
        plt.ylabel('Degrees')
        plt.title('Your Spine Posture Over Time')
        plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
            return save_path
        else:
            return plt
