import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_collector import DataCollector
from src.text_analyzer import TextAnalyzer

def test_data_collection():
    collector = DataCollector()
    analyzer = TextAnalyzer()
    
    print("Testing data collection and analysis...")
    
    twitter_data = collector.collect_twitter_data('sustainable fashion', 10)
    print(f"Collected {len(twitter_data)} Twitter posts")
    
    reddit_data = collector.collect_reddit_data('sustainability', 5)
    print(f"Collected {len(reddit_data)} Reddit posts")
    
    all_text_data = twitter_data + reddit_data
    analysis = analyzer.analyze_text_corpus(all_text_data)
    patterns = analyzer.detect_emerging_patterns(analysis)
    
    print(f"Detected {len(patterns['emerging_keywords'])} emerging keywords")
    print(f"Dominant topics: {patterns['dominant_topics']}")
    print(f"Average sentiment: {patterns['average_sentiment']:.3f}")
    
    print("Data collection test completed successfully")

if __name__ == "__main__":
    test_data_collection()