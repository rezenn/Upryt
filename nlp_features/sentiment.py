from transformers import pipeline
import random
import json
from datetime import datetime
from .utils import ensure_dir


class PostureSentimentAnalyzer:
    def __init__(self):
        # Use a smaller, faster model with CPU-only
        self.analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device='cpu'
        )

        self.feedback_responses = {
            'POSITIVE': [
                "Glad you're finding this helpful!",
                "Great to hear! Keep up the good posture."
            ],
            'NEGATIVE': [
                "Sorry to hear that. We'll try to improve.",
                "Your feedback is appreciated. We'll work on better suggestions."
            ],
            'NEUTRAL': [
                "Thanks for your feedback!",
                "We appreciate your input."
            ]
        }

        ensure_dir('user_data/sentiment_logs')

    def analyze_response(self, text):
        try:
            # Truncate to model max length and clean text
            clean_text = text[:512].strip()
            if not clean_text:
                return None

            result = self.analyzer(clean_text)
            return {
                'sentiment': result[0]['label'],
                'confidence': result[0]['score'],
                'response': random.choice(self.feedback_responses[result[0]['label']])
            }
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return None

    def log_user_feedback(self, text, session_data):
        """Log feedback with sentiment analysis for later review"""
        analysis = self.analyze_response(text)

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'feedback': text,
            'sentiment': analysis['sentiment'] if analysis else 'UNKNOWN',
            'confidence': analysis['confidence'] if analysis else 0,
            'session_data': session_data
        }

        log_file = f"user_data/sentiment_logs/{datetime.now().strftime('%Y-%m-%d')}.json"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

        return analysis
