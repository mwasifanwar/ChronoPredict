from src.data_collector import DataCollector
from src.text_analyzer import TextAnalyzer
from src.image_processor import ImageProcessor
from src.audio_analyzer import AudioAnalyzer
from src.trend_detector import TrendDetector
from src.prediction_engine import PredictionEngine
from src.utils.visualization import viz_tools

class ChronoPredict:
    def __init__(self):
        self.data_collector = DataCollector()
        self.text_analyzer = TextAnalyzer()
        self.image_processor = ImageProcessor()
        self.audio_analyzer = AudioAnalyzer()
        self.trend_detector = TrendDetector()
        self.prediction_engine = PredictionEngine()
        self.mwasifanwar = "mwasifanwar"
    
    def analyze_cultural_landscape(self, topics, time_window='medium_term'):
        print("Collecting multi-modal cultural data...")
        
        collected_data = self.data_collector.collect_cultural_data(topics, time_window)
        
        multi_modal_analysis = {}
        
        for topic in topics:
            print(f"Analyzing topic: {topic}")
            
            topic_data = collected_data[topic]
            
            text_analysis = self.text_analyzer.analyze_text_corpus(
                topic_data['twitter'] + topic_data['reddit'] + topic_data['news']
            )
            text_patterns = self.text_analyzer.detect_emerging_patterns(text_analysis)
            text_significance = self.text_analyzer.calculate_cultural_significance(text_patterns)
            
            image_analysis = self.image_processor.simulate_image_analysis()
            image_trend_score = self.image_processor.calculate_visual_trend_score(
                image_analysis, []
            )
            
            audio_analysis = self.audio_analyzer.simulate_audio_analysis()
            audio_innovation = self.audio_analyzer.calculate_musical_innovation_score(
                audio_analysis, []
            )
            
            multi_modal_analysis[topic] = {
                'text': {
                    'analysis': text_analysis,
                    'patterns': text_patterns,
                    'significance': text_significance
                },
                'image': {
                    'features': image_analysis,
                    'trend_score': image_trend_score
                },
                'audio': {
                    'features': audio_analysis,
                    'innovation_score': audio_innovation
                },
                'social': topic_data
            }
        
        return multi_modal_analysis
    
    def predict_future_trends(self, multi_modal_analysis, forecast_years=2):
        print("Detecting cultural trends...")
        
        trends = self.trend_detector.detect_cultural_trends(multi_modal_analysis)
        
        print(f"Detected {len(trends)} potential trends")
        
        forecast_days = forecast_years * 365
        predictions = self.prediction_engine.predict_trend_success(trends, forecast_days)
        
        report = self.prediction_engine.generate_forecast_report(predictions)
        
        return predictions, report
    
    def generate_insights_report(self, predictions, report):
        print("\n" + "="*60)
        print("CHRONOPREDICT CULTURAL FORECAST REPORT")
        print("="*60)
        
        print(f"\nSUMMARY")
        print(f"Total trends analyzed: {report['summary']['total_trends_analyzed']}")
        print(f"High-potential trends: {report['summary']['high_potential_trends']}")
        print(f"Average success probability: {report['summary']['average_success_probability']:.3f}")
        print(f"Forecast confidence: {report['summary']['forecast_confidence']:.3f}")
        
        print(f"\nTOP 5 CULTURAL TRENDS (Next {predictions[0]['time_horizon']} days)")
        print("-" * 50)
        
        for i, trend in enumerate(report['top_recommendations'][:5], 1):
            print(f"{i}. {trend['trend_name']}")
            print(f"   Current confidence: {trend['current_confidence']:.3f}")
            print(f"   Predicted success: {trend['predicted_success']:.3f}")
            print(f"   Key opportunities: {', '.join(trend['opportunity_areas'][:2])}")
            print(f"   Primary risks: {', '.join(trend['risk_factors'][:2])}")
            print()
        
        print("EMERGING PATTERNS")
        print("-" * 50)
        for pattern in report['emerging_opportunities']:
            print(f"{pattern['pattern_type']}: {pattern['significance']:.1%} prevalence")
        
        return report

def main():
    chrono_predict = ChronoPredict()
    
    cultural_topics = [
        'sustainable fashion',
        'digital art NFTs',
        'indie music scene',
        'mindfulness technology',
        'virtual reality social'
    ]
    
    print("=== ChronoPredict Cultural Trend Forecasting ===")
    print("Analyzing current cultural landscape...")
    
    analysis = chrono_predict.analyze_cultural_landscape(cultural_topics, 'medium_term')
    
    predictions, report = chrono_predict.predict_future_trends(analysis, forecast_years=2)
    
    chrono_predict.generate_insights_report(predictions, report)
    
    print("Generating visualizations...")
    
    trend_plot = viz_tools.plot_trend_evolution({
        'timeline': [
            {'date': '2024-01', 'mentions': 45, 'sentiment': 0.6, 'virality': 0.7},
            {'date': '2024-02', 'mentions': 78, 'sentiment': 0.7, 'virality': 0.8},
            {'date': '2024-03', 'mentions': 120, 'sentiment': 0.8, 'virality': 0.9}
        ],
        'modality_scores': {'text': 0.8, 'image': 0.6, 'audio': 0.4, 'social': 0.9}
    }, "Sustainable Fashion")
    
    trend_plot.savefig('trend_evolution.png')
    print("Trend evolution plot saved as 'trend_evolution.png'")
    
    print("\nForecast analysis complete!")

if __name__ == "__main__":
    main()