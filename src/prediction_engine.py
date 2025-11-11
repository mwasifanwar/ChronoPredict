import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from .config import config

class PredictionEngine:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.mwasifanwar = "mwasifanwar"
    
    def prepare_features(self, trend_data):
        features = []
        
        for trend in trend_data:
            feature_vector = []
            
            modality_scores = trend['modality_scores']
            feature_vector.append(modality_scores.get('text', 0))
            feature_vector.append(modality_scores.get('image', 0))
            feature_vector.append(modality_scores.get('audio', 0))
            feature_vector.append(modality_scores.get('social', 0))
            
            feature_vector.append(trend['confidence'])
            feature_vector.append(trend['momentum'])
            
            indicators = trend['key_indicators']
            feature_vector.append(indicators.get('sentiment', 0) if 'sentiment' in indicators else 0)
            feature_vector.append(indicators.get('virality', 0) if 'virality' in indicators else 0)
            
            feature_vector.append(len(modality_scores))
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def train_prediction_model(self, historical_trends, actual_outcomes):
        if len(historical_trends) < 10:
            return False
        
        X = self.prepare_features(historical_trends)
        y = np.array(actual_outcomes)
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        return True
    
    def predict_trend_success(self, current_trends, time_horizon_days=365):
        if not self.is_trained:
            return self.baseline_prediction(current_trends, time_horizon_days)
        
        predictions = []
        
        X = self.prepare_features(current_trends)
        X_scaled = self.scaler.transform(X)
        
        raw_predictions = self.model.predict(X_scaled)
        
        for i, trend in enumerate(current_trends):
            raw_pred = raw_predictions[i]
            adjusted_pred = self.adjust_prediction(raw_pred, trend, time_horizon_days)
            
            prediction = {
                'trend_name': trend['name'],
                'current_confidence': trend['confidence'],
                'predicted_success': adjusted_pred,
                'time_horizon': time_horizon_days,
                'risk_factors': self.identify_risk_factors(trend),
                'opportunity_areas': self.identify_opportunities(trend)
            }
            
            predictions.append(prediction)
        
        predictions.sort(key=lambda x: x['predicted_success'], reverse=True)
        return predictions
    
    def baseline_prediction(self, current_trends, time_horizon_days):
        predictions = []
        
        for trend in current_trends:
            base_score = trend['confidence']
            momentum_bonus = trend['momentum'] * 0.2
            modality_bonus = len(trend['modality_scores']) * 0.1
            
            predicted_success = base_score + momentum_bonus + modality_bonus
            predicted_success = max(0.0, min(1.0, predicted_success))
            
            time_adjustment = self.calculate_time_adjustment(time_horizon_days)
            predicted_success *= time_adjustment
            
            prediction = {
                'trend_name': trend['name'],
                'current_confidence': trend['confidence'],
                'predicted_success': predicted_success,
                'time_horizon': time_horizon_days,
                'risk_factors': ['untrained_model', 'limited_historical_data'],
                'opportunity_areas': ['cross_modality', 'early_adoption']
            }
            
            predictions.append(prediction)
        
        return predictions
    
    def adjust_prediction(self, raw_prediction, trend, time_horizon_days):
        adjusted = raw_prediction
        
        momentum_factor = 1.0 + (trend['momentum'] * 0.5)
        adjusted *= momentum_factor
        
        modality_diversity = len(trend['modality_scores'])
        diversity_bonus = 1.0 + (modality_diversity * 0.1)
        adjusted *= diversity_bonus
        
        time_factor = self.calculate_time_adjustment(time_horizon_days)
        adjusted *= time_factor
        
        return max(0.0, min(1.0, adjusted))
    
    def calculate_time_adjustment(self, time_horizon_days):
        if time_horizon_days <= 30:
            return 1.0
        elif time_horizon_days <= 90:
            return 0.8
        elif time_horizon_days <= 365:
            return 0.6
        else:
            return 0.4
    
    def identify_risk_factors(self, trend):
        risks = []
        
        if trend['confidence'] < 0.7:
            risks.append('low_confidence')
        
        if trend['momentum'] < 0:
            risks.append('negative_momentum')
        
        modality_scores = trend['modality_scores']
        if len(modality_scores) < 2:
            risks.append('single_modality')
        
        if modality_scores.get('social', 0) < 0.3:
            risks.append('limited_social_spread')
        
        return risks
    
    def identify_opportunities(self, trend):
        opportunities = []
        
        if trend['momentum'] > 0.2:
            opportunities.append('strong_growth_trajectory')
        
        modality_scores = trend['modality_scores']
        if len(modality_scores) >= 3:
            opportunities.append('cross_platform_potential')
        
        if modality_scores.get('text', 0) > 0.7:
            opportunities.append('strong_narrative')
        
        if modality_scores.get('image', 0) > 0.7:
            opportunities.append('visual_appeal')
        
        return opportunities
    
    def generate_forecast_report(self, predictions):
        report = {
            'summary': {
                'total_trends_analyzed': len(predictions),
                'high_potential_trends': len([p for p in predictions if p['predicted_success'] > 0.7]),
                'average_success_probability': np.mean([p['predicted_success'] for p in predictions]),
                'forecast_confidence': self.calculate_forecast_confidence(predictions)
            },
            'top_recommendations': predictions[:5],
            'emerging_opportunities': self.identify_emerging_patterns(predictions),
            'risk_assessment': self.assemble_risk_assessment(predictions)
        }
        
        return report
    
    def calculate_forecast_confidence(self, predictions):
        if not predictions:
            return 0.0
        
        confidence_scores = [p['current_confidence'] for p in predictions]
        avg_confidence = np.mean(confidence_scores)
        
        model_confidence = 0.8 if self.is_trained else 0.5
        
        overall_confidence = (avg_confidence + model_confidence) / 2.0
        return overall_confidence
    
    def identify_emerging_patterns(self, predictions):
        patterns = []
        
        high_potential = [p for p in predictions if p['predicted_success'] > 0.7]
        
        modality_patterns = defaultdict(int)
        for trend in high_potential:
            for opportunity in trend['opportunity_areas']:
                modality_patterns[opportunity] += 1
        
        dominant_patterns = sorted(modality_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for pattern, count in dominant_patterns:
            patterns.append({
                'pattern_type': pattern,
                'frequency': count,
                'significance': count / len(high_potential)
            })
        
        return patterns
    
    def assemble_risk_assessment(self, predictions):
        risk_counts = defaultdict(int)
        
        for trend in predictions:
            for risk in trend['risk_factors']:
                risk_counts[risk] += 1
        
        total_trends = len(predictions)
        risk_assessment = []
        
        for risk, count in risk_counts.items():
            risk_assessment.append({
                'risk_factor': risk,
                'prevalence': count / total_trends,
                'severity': self.assess_risk_severity(risk)
            })
        
        return risk_assessment
    
    def assess_risk_severity(self, risk_factor):
        severity_scores = {
            'low_confidence': 0.7,
            'negative_momentum': 0.9,
            'single_modality': 0.5,
            'limited_social_spread': 0.6
        }
        
        return severity_scores.get(risk_factor, 0.5)

prediction_engine = PredictionEngine()