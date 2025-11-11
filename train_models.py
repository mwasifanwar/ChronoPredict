import numpy as np
from src.data_collector import DataCollector
from src.text_analyzer import TextAnalyzer
from src.trend_detector import TrendDetector
from src.prediction_engine import PredictionEngine
from src.config import config

def generate_training_data():
    collector = DataCollector()
    text_analyzer = TextAnalyzer()
    trend_detector = TrendDetector()
    
    cultural_topics = [
        'sustainable fashion', 'digital art', 'indie music',
        'urban gardening', 'mindfulness', 'virtual reality'
    ]
    
    historical_trends = []
    actual_outcomes = []
    
    for topic in cultural_topics:
        print(f"Collecting data for: {topic}")
        
        data = collector.collect_cultural_data([topic], 'long_term')
        
        text_data = data[topic]['twitter'] + data[topic]['reddit'] + data[topic]['news']
        text_analysis = text_analyzer.analyze_text_corpus(text_data)
        patterns = text_analyzer.detect_emerging_patterns(text_analysis)
        
        significance = text_analyzer.calculate_cultural_significance(patterns)
        
        multi_modal_data = {
            'text': {topic: {'analysis': text_analysis, 'patterns': patterns, 'significance': significance}},
            'image': {},
            'audio': {},
            'social': data[topic]
        }
        
        trends = trend_detector.detect_cultural_trends(multi_modal_data)
        
        for trend in trends:
            historical_trends.append(trend)
            
            simulated_outcome = np.random.beta(
                trend['confidence'] * 10,
                (1 - trend['confidence']) * 10
            )
            actual_outcomes.append(simulated_outcome)
    
    return historical_trends, actual_outcomes

def main():
    print("Generating training data for ChronoPredict...")
    
    historical_trends, actual_outcomes = generate_training_data()
    
    print(f"Generated {len(historical_trends)} historical trend examples")
    print(f"Outcome range: {np.min(actual_outcomes):.3f} - {np.max(actual_outcomes):.3f}")
    
    prediction_engine = PredictionEngine()
    
    success = prediction_engine.train_prediction_model(historical_trends, actual_outcomes)
    
    if success:
        print("Prediction model trained successfully")
        
        test_trends = historical_trends[:5]
        predictions = prediction_engine.predict_trend_success(test_trends, 365)
        
        print("\nSample predictions:")
        for pred in predictions[:3]:
            print(f"Trend: {pred['trend_name']}")
            print(f"  Predicted success: {pred['predicted_success']:.3f}")
            print(f"  Risks: {pred['risk_factors']}")
    else:
        print("Insufficient data for model training")
    
    print("Training completed")

if __name__ == "__main__":
    main()