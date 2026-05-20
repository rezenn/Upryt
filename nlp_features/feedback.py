from transformers import pipeline, set_seed
import random
import time
from datetime import datetime
import json
from .utils import ensure_dir


class NLPFeedbackGenerator:
    def __init__(self, language="English"):
        # Initialize with CPU-only configuration
        self.generator = pipeline(
            'text-generation',
            model='distilgpt2',
            device='cpu',
            torch_dtype='auto'  # Automatically handles data type
        )

        set_seed(42)  # For reproducible results
        self.language = language
        self._init_feedback_templates()
        ensure_dir('user_data/feedback_logs')

    def _init_feedback_templates(self):
        """Initialize feedback templates and advice database"""
        self.feedback_templates = {
            'shoulder': [
                "Your shoulders seem {comparison} than usual. {advice}",
                "I notice your shoulder angle is {comparison}. {advice}"
            ],
            'neck': [
                "Your neck position appears {comparison}. {advice}",
                "The neck angle is {comparison} your average. {advice}"
            ],
            'spine': [
                "Your spine alignment looks {comparison}. {advice}",
                "Spine posture is {comparison} normal. {advice}"
            ]
        }

        self.advice_db = {
            'shoulder': {
                'worse': "Try doing shoulder rolls every 30 minutes.",
                'better': "Great job keeping your shoulders relaxed!"
            },
            'neck': {
                'worse': "Remember to keep your screen at eye level.",
                'better': "Your neck alignment looks perfect!"
            },
            'spine': {
                'worse': "Consider adjusting your chair height for better support.",
                'better': "Excellent spinal alignment - keep it up!"
            }
        }

    def generate_feedback(self, current_angles, user_profile):
        """
        Generate personalized feedback based on current posture and user history

        Args:
            current_angles: dict with keys 'shoulder', 'neck', 'spine'
            user_profile: dict with mean and std for each angle
        """
        feedback = []

        for angle_type in ['shoulder', 'neck', 'spine']:
            current = current_angles[angle_type]
            mean = user_profile[f'{angle_type}_angle']['mean']
            std = user_profile[f'{angle_type}_angle']['std']

            # Determine comparison
            if current < mean - 1.5*std:
                comparison = "worse"
                advice = self.advice_db[angle_type]['worse']
            elif current > mean + 1.5*std:
                comparison = "better"
                advice = self.advice_db[angle_type]['better']
            else:
                continue

            # Select random template
            template = random.choice(self.feedback_templates[angle_type])
            message = template.format(comparison=comparison, advice=advice)

            # Enhance with GPT-2
            try:
                enhanced = self.generator(
                    message,
                    max_length=50,
                    num_return_sequences=1,
                    temperature=0.7,
                    truncation=True
                )[0]['generated_text']
                feedback.append(enhanced)
            except Exception as e:
                print(f"Feedback generation error: {e}")
                feedback.append(message)  # Fallback to original message

        return feedback if feedback else ["Your posture looks good overall!"]

    def log_feedback(self, feedback_text, posture_data):
        """Log generated feedback for analysis"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'feedback': feedback_text,
            'posture_data': posture_data,
            'language': self.language
        }

        log_file = f"user_data/feedback_logs/{datetime.now().strftime('%Y-%m-%d')}.json"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
