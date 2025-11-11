import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from .config import config
from .utils.nlp_tools import nlp_tools

class TrendDetector:
    def __init__(self):
        self.trend_history = defaultdict(list)
        self.mwasifanwar = "mwasifanwar"
    
    def detect_cultural_trends(self, multi_modal_data):
        trends = []
        
        text_data = multi_modal_data.get('text', {})
        image_data = multi_modal_data.get('image', {})
        audio_data = multi_modal_data.get('audio', {})
        social_data = multi_modal_data.get('social', {})
        
        text_trends = self.analyze_text_trends(text_data)
        image_trends = self.analyze_visual_trends(image_data)
        audio_trends = self.analyze_audio_trends(audio_data)
        social_trends = self.analyze_social_dynamics(social_data)
        
        combined_trends = self.combine_modality_trends(
            text_trends, image_trends, audio_trends, social_trends
        )
        
        for trend in combined_trends:
            trend_confidence = self.calculate_trend_confidence(trend)
            trend_momentum = self.calculate_trend_momentum(trend)
            
            if trend_confidence >= config.TREND_THRESHOLDS['emerging']:
                trend_entry = {
                    'name': trend['name'],
                    'description': trend['description'],
                    'confidence': trend_confidence,
                    'momentum': trend_momentum,
                    'first_detected': datetime.now().isoformat(),
                    'modality_scores': trend['modality_scores'],
                    'key_indicators': trend['key_indicators']
                }
                trends.append(trend_entry)
        
        trends.sort(key=lambda x: x['confidence'], reverse=True)
        return trends
    
    def analyze_text_trends(self, text_data):
        trends = []
        
        for topic, analysis in text_data.items():
            patterns = analysis.get('patterns', {})
            
            significance = analysis.get('significance', 0.0)
            momentum = patterns.get('temporal_trend', {}).get('momentum', 0.0)
            
            if significance > 0.5 and momentum > 0:
                trend = {
                    'name': f"Text Trend: {topic}",
                    'description': f"Emerging discussion around {topic}",
                    'modality_scores': {'text': significance},
                    'key_indicators': {
                        'keywords': patterns.get('emerging_keywords', []),
                        'sentiment': patterns.get('average_sentiment', 0),
                        'virality': patterns.get('average_virality', 0)
                    }
                }
                trends.append(trend)
        
        return trends
    
    def analyze_visual_trends(self, image_data):
        trends = []
        
        for style, features in image_data.items():
            trend_score = features.get('trend_score', 0.0)
            
            if trend_score > 0.5:
                trend = {
                    'name': f"Visual Trend: {style}",
                    'description': f"Emerging visual style: {style}",
                    'modality_scores': {'image': trend_score},
                    'key_indicators': {
                        'color_palette': features.get('color_palette', []),
                        'style_scores': features.get('style_indicators', {}),
                        'composition': features.get('composition', {})
                    }
                }
                trends.append(trend)
        
        return trends
    
    def analyze_audio_trends(self, audio_data):
        trends = []
        
        for genre, features in audio_data.items():
            innovation_score = features.get('innovation_score', 0.0)
            
            if innovation_score > 0.5:
                trend = {
                    'name': f"Audio Trend: {genre}",
                    'description': f"Innovative developments in {genre} music",
                    'modality_scores': {'audio': innovation_score},
                    'key_indicators': {
                        'tempo': features.get('tempo', 0),
                        'genre_fusion': features.get('genre_characteristics', {}),
                        'rhythm_complexity': features.get('rhythm_patterns', {})
                    }
                }
                trends.append(trend)
        
        return trends
    
    def analyze_social_dynamics(self, social_data):
        trends = []
        
        for platform, data in social_data.items():
            engagement_trend = self.analyze_engagement_patterns(data)
            network_effects = self.analyze_network_effects(data)
            
            social_score = (engagement_trend + network_effects) / 2.0
            
            if social_score > 0.5:
                trend = {
                    'name': f"Social Trend: {platform}",
                    'description': f"Emerging social dynamics on {platform}",
                    'modality_scores': {'social': social_score},
                    'key_indicators': {
                        'engagement_trend': engagement_trend,
                        'network_effects': network_effects,
                        'user_diversity': len(data) / 100.0
                    }
                }
                trends.append(trend)
        
        return trends
    
    def analyze_engagement_patterns(self, social_data):
        if not social_data:
            return 0.0
        
        engagement_metrics = []
        for item in social_data:
            engagement = item.get('likes', 0) + item.get('shares', 0) + item.get('comments', 0)
            engagement_metrics.append(engagement)
        
        avg_engagement = np.mean(engagement_metrics)
        engagement_score = min(avg_engagement / 1000.0, 1.0)
        
        return engagement_score
    
    def analyze_network_effects(self, social_data):
        if not social_data:
            return 0.0
        
        unique_users = len(set(item.get('user', '') for item in social_data))
        user_diversity = min(unique_users / 50.0, 1.0)
        
        return user_diversity
    
    def combine_modality_trends(self, text_trends, image_trends, audio_trends, social_trends):
        all_trends = text_trends + image_trends + audio_trends + social_trends
        
        combined_trends = []
        trend_groups = defaultdict(list)
        
        for trend in all_trends:
            trend_name = trend['name'].split(': ')[1]
            trend_groups[trend_name].append(trend)
        
        for trend_name, modality_trends in trend_groups.items():
            if len(modality_trends) >= 2:
                combined_trend = self.merge_modality_trends(trend_name, modality_trends)
                combined_trends.append(combined_trend)
        
        return combined_trends
    
    def merge_modality_trends(self, trend_name, modality_trends):
        merged_scores = defaultdict(float)
        merged_indicators = defaultdict(list)
        
        for trend in modality_trends:
            for modality, score in trend['modality_scores'].items():
                merged_scores[modality] = max(merged_scores[modality], score)
            
            for key, value in trend['key_indicators'].items():
                if isinstance(value, list):
                    merged_indicators[key].extend(value)
                else:
                    merged_indicators[key].append(value)
        
        overall_confidence = sum(merged_scores.values()) / len(merged_scores)
        
        return {
            'name': f"Multi-modal Trend: {trend_name}",
            'description': f"Cross-platform cultural trend around {trend_name}",
            'modality_scores': dict(merged_scores),
            'key_indicators': dict(merged_indicators),
            'overall_confidence': overall_confidence
        }
    
    def calculate_trend_confidence(self, trend):
        modality_scores = trend['modality_scores']
        
        weighted_score = 0.0
        for modality, score in modality_scores.items():
            weight = config.MODALITY_WEIGHTS.get(modality, 0.2)
            weighted_score += score * weight
        
        cross_modality_bonus = len(modality_scores) * 0.1
        weighted_score += min(cross_modality_bonus, 0.3)
        
        return max(0.0, min(1.0, weighted_score))
    
    def calculate_trend_momentum(self, trend):
        trend_name = trend['name']
        current_time = datetime.now()
        
        self.trend_history[trend_name].append({
            'timestamp': current_time,
            'confidence': trend['confidence']
        })
        
        history = self.trend_history[trend_name]
        if len(history) < 2:
            return 0.0
        
        recent_history = [h for h in history if h['timestamp'] > current_time - timedelta(days=7)]
        if len(recent_history) < 2:
            return 0.0
        
        confidences = [h['confidence'] for h in recent_history]
        momentum = (confidences[-1] - confidences[0]) / len(confidences)
        
        return momentum
    
    def predict_trend_trajectory(self, trend, forecast_days=365):
        history = self.trend_history[trend['name']]
        
        if len(history) < 3:
            return {'prediction': 'insufficient_data', 'confidence': 0.0}
        
        confidences = [h['confidence'] for h in history]
        timestamps = [h['timestamp'] for h in history]
        
        time_deltas = [(ts - timestamps[0]).days for ts in timestamps]
        
        try:
            trend_line = np.polyfit(time_deltas, confidences, 1)
            future_confidence = np.polyval(trend_line, time_deltas[-1] + forecast_days)
            
            future_confidence = max(0.0, min(1.0, future_confidence))
            
            if future_confidence > 0.8:
                trajectory = 'mainstream'
            elif future_confidence > 0.6:
                trajectory = 'growing'
            else:
                trajectory = 'niche'
            
            return {
                'prediction': trajectory,
                'confidence': future_confidence,
                'estimated_peak': future_confidence,
                'time_to_peak': forecast_days
            }
        except:
            return {'prediction': 'uncertain', 'confidence': 0.0}

trend_detector = TrendDetector()