import numpy as np
import re
from collections import Counter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

class NLPTools:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.mwasifanwar = "mwasifanwar"
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
    
    def extract_keywords(self, text, top_n=10):
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = Counter(words)
        return word_freq.most_common(top_n)
    
    def analyze_sentiment(self, text):
        scores = self.sia.polarity_scores(text)
        return scores
    
    def detect_topics(self, text):
        topics = []
        
        cultural_keywords = {
            'fashion': ['style', 'fashion', 'outfit', 'trend', 'designer'],
            'music': ['song', 'album', 'artist', 'track', 'genre'],
            'art': ['painting', 'exhibition', 'gallery', 'artist', 'installation'],
            'technology': ['tech', 'innovation', 'digital', 'app', 'software'],
            'social': ['movement', 'community', 'activism', 'protest', 'rally']
        }
        
        text_lower = text.lower()
        for topic, keywords in cultural_keywords.items():
            keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
            if keyword_count >= 2:
                topics.append(topic)
        
        return topics
    
    def calculate_virality_score(self, text, engagement_metrics):
        base_score = 0.0
        
        sentiment_scores = self.analyze_sentiment(text)
        compound_sentiment = sentiment_scores['compound']
        
        keywords = self.extract_keywords(text, 5)
        keyword_diversity = len(keywords) / 5.0
        
        engagement_score = min(engagement_metrics.get('likes', 0) / 1000.0, 1.0)
        share_score = min(engagement_metrics.get('shares', 0) / 500.0, 1.0)
        
        virality = (compound_sentiment * 0.3 + 
                   keyword_diversity * 0.2 + 
                   engagement_score * 0.3 + 
                   share_score * 0.2)
        
        return max(0.0, min(1.0, virality))
    
    def extract_named_entities(self, text):
        entities = {
            'people': [],
            'organizations': [],
            'locations': [],
            'events': []
        }
        
        event_patterns = [
            r'(\b[A-Z][a-z]+ [A-Z][a-z]+ [Ff]estival\b)',
            r'(\b[A-Z][a-z]+ [Cc]onference\b)',
            r'(\b[A-Z][a-z]+ [Ee]xhibition\b)',
            r'(\b[A-Z][a-z]+ [Ww]eek\b)'
        ]
        
        for pattern in event_patterns:
            matches = re.findall(pattern, text)
            entities['events'].extend(matches)
        
        org_pattern = r'\b([A-Z][a-z]+ (?:Inc|Corp|Foundation|Studio|Records))\b'
        entities['organizations'] = re.findall(org_pattern, text)
        
        return entities
    
    def calculate_complexity_score(self, text):
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        unique_word_ratio = len(set(words)) / len(words)
        
        complexity = (min(avg_sentence_length / 20.0, 1.0) * 0.6 + 
                     unique_word_ratio * 0.4)
        
        return complexity

nlp_tools = NLPTools()