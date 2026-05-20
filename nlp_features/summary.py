from transformers import pipeline
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json
import random
from .utils import ensure_dir, format_duration


class DailySummaryGenerator:
    def __init__(self):
        # Initialize summarization model with CPU fallback
        self.summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device='cpu'
        )
        ensure_dir('user_data/summaries')
        self.last_summary_hash = None  # Track last summary to avoid repeats

    def generate_summary(self, session_data):
        """Generates a unique summary with each call using multiple techniques"""
        max_attempts = 3
        attempt = 0

        while attempt < max_attempts:
            try:
                # Generate candidate summary
                candidate = self._generate_candidate_summary(session_data)
                current_hash = hash(candidate['text'])

                # Ensure variation from last summary
                if current_hash != self.last_summary_hash:
                    self.last_summary_hash = current_hash

                    # Generate visualization
                    img_path = self._generate_summary_chart(session_data)
                    candidate['chart'] = img_path

                    # Save summary
                    self._save_daily_summary(
                        candidate['text'], session_data, img_path)

                    return candidate

                attempt += 1
            except Exception as e:
                print(f"Summary generation attempt {attempt + 1} failed: {e}")
                attempt += 1

        # Fallback if all attempts fail
        return {
            'text': self._create_fallback_summary(session_data),
            'chart': None
        }

    def _generate_candidate_summary(self, data):
        """Generates one candidate summary using layered approaches"""
        # 1. Choose a random template structure
        templates = [
            "Posture Report for {date}:\n{stats}\n{insight}\n{advice}\n{note}",
            "Your {date} Analysis:\n{stats}\n{insight}\n{advice}",
            "Summary ({date}):\n{stats}\nKey Observation: {insight}\nRecommendation: {advice}",
            "Daily Posture Review ({date}):\n{stats}\n{insight}\n{note}"
        ]

        # 2. Generate all components
        components = {
            'date': datetime.now().strftime("%A, %B %d"),
            'stats': self._generate_stats_component(data),
            'insight': self._generate_insight_component(data),
            'advice': self._generate_advice_component(data),
            'note': self._generate_motivational_note(data)
        }

        # 3. Select and format template
        template = random.choice(templates)
        summary_text = template.format(**components)

        # 4. Optional AI refinement (30% chance)
        if random.random() < 0.3:
            try:
                refined = self.summarizer(
                    summary_text,
                    max_length=len(summary_text.split()) + 15,
                    min_length=max(30, len(summary_text.split()) - 10),
                    do_sample=True,
                    temperature=0.7  # Controls randomness
                )
                summary_text = refined[0]['summary_text']
            except:
                pass  # Keep original if refinement fails

        return {'text': summary_text}

    def _generate_stats_component(self, data):
        """Generates varied statistical presentations"""
        variants = [
            f"- Session Duration: {format_duration(data['total_duration'])}\n"
            f"- Good Posture: {data['good_percent']}% ({format_duration(data['good_time'])})\n"
            f"- Corrections Needed: {data['corrections']}\n"
            f"- Best Streak: {format_duration(data['max_good_streak'])}",

            f"📊 Metrics:\n"
            f"Good Posture: {data['good_percent']}%\n"
            f"Total Corrections: {data['corrections']}\n"
            f"Longest Good Streak: {format_duration(data['max_good_streak'])}\n"
            f"Common Issue: {data['common_issue']}",

            f"• {data['good_percent']}% good posture time\n"
            f"• Required {data['corrections']} posture corrections\n"
            f"• {format_duration(data['max_good_streak'])} longest maintained posture\n"
            f"• Primary concern: {data['common_issue']}"
        ]
        return random.choice(variants)

    def _generate_insight_component(self, data):
        """Generates context-aware insights"""
        percent = data['good_percent']
        issue = data['common_issue'].lower()

        if percent > 85:
            insights = [
                "Excellent posture maintenance throughout the day!",
                "Your posture discipline is truly impressive.",
                "Professional-level posture consistency achieved!",
                "You're setting a great example for ergonomic health."
            ]
        elif percent > 65:
            insights = [
                "Good overall posture with room for refinement.",
                "Your posture awareness is developing well.",
                "Solid performance with occasional lapses.",
                "Maintaining decent posture with some variability."
            ]
        else:
            insights = [
                "Your posture needs significant attention.",
                "Frequent slouching detected during sessions.",
                "Posture maintenance was challenging today.",
                "Ergonomic adjustments are strongly recommended."
            ]

        # Add issue-specific insights
        if 'shoulder' in issue:
            insights.append("Shoulder alignment needs particular focus.")
        if 'neck' in issue:
            insights.append("Forward head posture was noticeable.")
        if 'spine' in issue:
            insights.append("Spinal curvature requires attention.")

        return random.choice(insights)

    def _generate_advice_component(self, data):
        """Generates personalized recommendations"""
        issue = data['common_issue'].lower()
        percent = data['good_percent']

        # General advice based on performance
        if percent > 85:
            general_advice = [
                "Maintain your excellent habits!",
                "Keep up your current routine.",
                "Your methods are working well."
            ]
        elif percent > 65:
            general_advice = [
                "Try periodic posture checks.",
                "Set reminders every 30 minutes.",
                "Consider ergonomic adjustments."
            ]
        else:
            general_advice = [
                "Implement structured posture breaks.",
                "Consult an ergonomics specialist.",
                "Try posture-correcting exercises."
            ]

        # Issue-specific advice
        specific_advice = []
        if 'shoulder' in issue:
            specific_advice.extend([
                "Practice shoulder blade squeezes.",
                "Adjust chair and desk heights.",
                "Relax your shoulders periodically."
            ])
        if 'neck' in issue:
            specific_advice.extend([
                "Position monitor at eye level.",
                "Try chin tuck exercises.",
                "Be mindful of forward head posture."
            ])
        if 'spine' in issue:
            specific_advice.extend([
                "Use lumbar support cushions.",
                "Alternate between sitting and standing.",
                "Engage your core while sitting."
            ])

        # Combine advice
        all_advice = general_advice + (specific_advice if specific_advice else [
            "Regular posture checks help build habits.",
            "Small improvements compound over time."
        ])

        return random.choice(all_advice)

    def _generate_motivational_note(self, data):
        """Generates psychological boosters"""
        percent = data['good_percent']

        if percent > 85:
            notes = [
                "You're mastering posture discipline!",
                "Your consistency is inspiring!",
                "Perfect posture becomes natural with this dedication."
            ]
        elif percent > 65:
            notes = [
                "Every day of practice counts!",
                "You're building healthier habits.",
                "Notice how good posture improves your energy."
            ]
        else:
            notes = [
                "Tomorrow is a new opportunity!",
                "Progress happens one correction at a time.",
                "Your posture awareness is growing."
            ]

        return random.choice(notes)

    def _create_fallback_summary(self, data):
        """Creates a basic summary when generation fails"""
        return (
            f"Posture Summary for {datetime.now().strftime('%A, %B %d')}:\n"
            f"- Good Posture: {data['good_percent']}%\n"
            f"- Session Duration: {format_duration(data['total_duration'])}\n"
            f"- Corrections: {data['corrections']}\n"
            f"- Main Issue: {data['common_issue']}"
        )

    def _generate_summary_chart(self, data):
        """Generates visualization of daily data"""
        plt.figure(figsize=(10, 6))

        # Create more detailed chart
        categories = ['Good Posture', 'Poor Posture', 'Corrections']
        values = [
            data['good_time'] / 60,  # Convert to minutes
            data['poor_time'] / 60,
            data['corrections']
        ]

        colors = ['#4CAF50', '#F44336', '#FFC107']
        bars = plt.bar(categories, values, color=colors)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f}' if height >= 1 else f'{height:.2f}',
                     ha='center', va='bottom')

        plt.title(f"Posture Summary - {datetime.now().strftime('%Y-%m-%d')}")
        plt.ylabel('Minutes (Count for Corrections)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        img_path = f"user_data/summaries/{datetime.now().strftime('%Y-%m-%d')}.png"
        plt.savefig(img_path, bbox_inches='tight', dpi=100)
        plt.close()

        return img_path

    def _save_daily_summary(self, summary_text, raw_data, chart_path):
        """Saves complete summary data with metadata"""
        summary_data = {
            'meta': {
                'generated_at': datetime.now().isoformat(),
                'version': '2.1',
                'generator': 'dynamic_nlg'
            },
            'summary': summary_text,
            'visualization': chart_path,
            'raw_data': raw_data,
            'components': {
                'stats_hash': hash(self._generate_stats_component(raw_data)),
                'insight_hash': hash(self._generate_insight_component(raw_data)),
                'advice_hash': hash(self._generate_advice_component(raw_data))
            }
        }

        save_path = f"user_data/summaries/{datetime.now().strftime('%Y-%m-%d')}.json"
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
