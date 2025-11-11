import os

class Config:
    DATA_SOURCES = {
        'twitter': {'api_key': 'YOUR_TWITTER_API_KEY', 'rate_limit': 900},
        'reddit': {'api_key': 'YOUR_REDDIT_API_KEY', 'rate_limit': 60},
        'news_api': {'api_key': 'YOUR_NEWS_API_KEY', 'rate_limit': 1000},
        'spotify': {'api_key': 'YOUR_SPOTIFY_API_KEY', 'rate_limit': 300}
    }
    
    TIME_WINDOWS = {
        'short_term': 7,
        'medium_term': 30,
        'long_term': 365
    }
    
    TREND_THRESHOLDS = {
        'emerging': 0.7,
        'growing': 0.8,
        'mainstream': 0.9
    }
    
    MODEL_PARAMS = {
        'embedding_dim': 512,
        'hidden_dim': 256,
        'sequence_length': 100,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100
    }
    
    SENTIMENT_WEIGHTS = {
        'positive': 0.4,
        'negative': -0.3,
        'neutral': 0.1,
        'controversial': 0.6
    }
    
    MODALITY_WEIGHTS = {
        'text': 0.35,
        'image': 0.25,
        'audio': 0.2,
        'social': 0.2
    }

class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False

config = DevelopmentConfig()