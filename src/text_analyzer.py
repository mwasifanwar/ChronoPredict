import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from .utils.nlp_tools import nlp_tools
from .config import config

class TextAnalyzer:
    def __init__(self):
        self.mwasifanwar = "mwasifanwar"
    
    def analyze_text_corpus(self, text_data):
        analysis_results = {
            'top_keywords': [],
            'sentiment_trends': [],
            'topic_distribution': defaultdict(int),
            'virality_scores': [],
            'temporal_patterns': defaultdict(list)
        }
        
        for item in text_data:
            text = item.get('text', '') or item.get('title', '') or item.get('description', '')
            if not text:
                continue
            
            keywords = nlp_tools.extract_keywords(text)
            analysis_results['top_keywords'].extend(keywords)
            
            sentiment = nlp_tools.analyze_sentiment(text)
            analysis_results['sentiment_trends'].append(sentiment['compound'])
            
            topics = nlp_tools.detect_topics(text)
            for topic in topics:
                analysis_results['topic_distribution'][topic] += 1
            
            engagement_metrics = {
                'likes': item.get('likes', 0) or item.get('upvotes', 0),
                'shares': item.get('retweets', 0) or item.get('comments', 0)
            }
            virality = nlp_tools.calculate_virality_score(text, engagement_metrics)
            analysis_results['virality_scores'].append(virality)
            
            created_at = item.get('created_at')
            if created_at:
                if isinstance(created_at, str):
                    try:
                        created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    except:
                        created_at = datetime.now()
                time_key = created_at.strftime('%Y-%m-%d')
                analysis_results['temporal_patterns'][time_key].append({
                    'sentiment': sentiment['compound'],
                    'virality': virality
                })
        
        return analysis_results
    
    def detect_emerging_patterns(self, analysis_results, time_window=30):
        patterns = {}
        
        keyword_counter = defaultdict(int)
        for keyword, count in analysis_results['top_keywords']:
            keyword_counter[keyword] += count
        
        emerging_keywords = []
        for keyword, count in keyword_counter.items():
            if count >= 3:
                emerging_keywords.append((keyword, count))
        
        emerging_keywords.sort(key=lambda x: x[1], reverse=True)
        patterns['emerging_keywords'] = emerging_keywords[:10]
        
        topic_distribution = dict(analysis_results['topic_distribution'])
        dominant_topics = sorted(topic_distribution.items(), key=lambda x: x[1], reverse=True)[:5]
        patterns['dominant_topics'] = dominant_topics
        
        avg_sentiment = np.mean(analysis_results['sentiment_trends']) if analysis_results['sentiment_trends'] else 0
        patterns['average_sentiment'] = avg_sentiment
        
        avg_virality = np.mean(analysis_results['virality_scores']) if analysis_results['virality_scores'] else 0
        patterns['average_virality'] = avg_virality
        
        temporal_trend = self.analyze_temporal_trends(analysis_results['temporal_patterns'])
        patterns['temporal_trend'] = temporal_trend
        
        return patterns
    
    def analyze_temporal_trends(self, temporal_patterns):
        if not temporal_patterns:
            return {'trend': 'stable', 'momentum': 0.0}
        
        dates = sorted(temporal_patterns.keys())
        if len(dates) < 2:
            return {'trend': 'stable', 'momentum': 0.0}
        
        sentiment_trend = []
        virality_trend = []
        
        for date in dates[-7:]:
            day_data = temporal_patterns[date]
            if day_data:
                avg_sentiment = np.mean([d['sentiment'] for d in day_data])
                avg_virality = np.mean([d['virality'] for d in day_data])
                sentiment_trend.append(avg_sentiment)
                virality_trend.append(avg_virality)
        
        if len(sentiment_trend) >= 2:
            sentiment_slope = self.calculate_slope(sentiment_trend)
            virality_slope = self.calculate_slope(virality_trend)
        else:
            sentiment_slope = virality_slope = 0.0
        
        overall_momentum = (sentiment_slope + virality_slope) / 2.0
        
        if overall_momentum > 0.1:
            trend = 'growing'
        elif overall_momentum < -0.1:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'momentum': overall_momentum,
            'sentiment_slope': sentiment_slope,
            'virality_slope': virality_slope
        }
    
    def calculate_slope(self, values):
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope
    
    def calculate_cultural_significance(self, patterns):
        significance_score = 0.0
        
        keyword_diversity = len(patterns['emerging_keywords']) / 10.0
        significance_score += keyword_diversity * 0.2
        
        topic_variety = len(patterns['dominant_topics']) / 5.0
        significance_score += topic_variety * 0.2
        
        sentiment_strength = abs(patterns['average_sentiment'])
        significance_score += sentiment_strength * 0.2
        
        virality_impact = patterns['average_virality']
        significance_score += virality_impact * 0.2
        
        momentum = abs(patterns['temporal_trend']['momentum'])
        significance_score += momentum * 0.2
        
        return max(0.0, min(1.0, significance_score))

text_analyzer = TextAnalyzer()