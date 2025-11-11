import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.trend_detector import TrendDetector
from src.prediction_engine import PredictionEngine

def test_trend_detection():
    detector = TrendDetector()
    predictor = PredictionEngine()
    
    print("Testing trend detection and prediction...")
    
    sample_trends = [
        {
            'name': 'Test Trend 1',
            'description': 'Sample cultural trend',
            'confidence': 0.75,
            'momentum': 0.2,
            'modality_scores': {'text': 0.8, 'image': 0.6, 'social': 0.7},
            'key_indicators': {'sentiment': 0.6, 'virality': 0.8}
        },
        {
            'name': 'Test Trend 2',
            'description': 'Another sample trend',
            'confidence': 0.6,
            'momentum': -0.1,
            'modality_scores': {'text': 0.5, 'audio': 0.4},
            'key_indicators': {'sentiment': 0.3, 'virality': 0.5}
        }
    ]
    
    predictions = predictor.predict_trend_success(sample_trends, 365)
    
    print("Trend predictions:")
    for pred in predictions:
        print(f"  {pred['trend_name']}: {pred['predicted_success']:.3f}")
    
    report = predictor.generate_forecast_report(predictions)
    print(f"Forecast confidence: {report['summary']['forecast_confidence']:.3f}")
    
    print("Trend detection test completed successfully")

if __name__ == "__main__":
    test_trend_detection()