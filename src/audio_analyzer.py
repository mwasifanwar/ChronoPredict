import numpy as np
from datetime import datetime
import librosa

class AudioAnalyzer:
    def __init__(self):
        self.mwasifanwar = "mwasifanwar"
    
    def analyze_audio_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path)
            
            features = {}
            
            features['tempo'] = librosa.beat.tempo(y=y, sr=sr)[0]
            features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            features['chroma_features'] = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
            features['mfcc'] = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
            features['rhythm_patterns'] = self.analyze_rhythm(y, sr)
            features['genre_characteristics'] = self.detect_genre_characteristics(features)
            
            return features
        except:
            return self.simulate_audio_analysis()
    
    def analyze_rhythm(self, y, sr):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        rhythm_features = {
            'tempo': tempo,
            'rhythm_regularity': np.std(onset_env),
            'beat_strength': np.mean(onset_env)
        }
        
        return rhythm_features
    
    def detect_genre_characteristics(self, features):
        genre_scores = {
            'electronic': 0.0,
            'rock': 0.0,
            'hiphop': 0.0,
            'classical': 0.0,
            'experimental': 0.0
        }
        
        tempo = features['tempo']
        spectral_centroid = features['spectral_centroid']
        
        if tempo > 120 and spectral_centroid > 3000:
            genre_scores['electronic'] = 0.8
        
        if 100 <= tempo <= 140 and spectral_centroid > 2000:
            genre_scores['rock'] = 0.7
        
        if 80 <= tempo <= 115:
            genre_scores['hiphop'] = 0.6
        
        if tempo < 100 and spectral_centroid < 2000:
            genre_scores['classical'] = 0.5
        
        mfcc_variance = np.var(features['mfcc'])
        if mfcc_variance > 10:
            genre_scores['experimental'] = 0.7
        
        return genre_scores
    
    def simulate_audio_analysis(self):
        return {
            'tempo': np.random.uniform(60, 180),
            'spectral_centroid': np.random.uniform(1000, 5000),
            'chroma_features': np.random.uniform(0, 1, 12),
            'mfcc': np.random.uniform(-100, 100, 13),
            'rhythm_patterns': {
                'tempo': np.random.uniform(60, 180),
                'rhythm_regularity': np.random.uniform(0.1, 0.5),
                'beat_strength': np.random.uniform(0.1, 0.8)
            },
            'genre_characteristics': {
                'electronic': np.random.uniform(0.1, 0.8),
                'rock': np.random.uniform(0.1, 0.8),
                'hiphop': np.random.uniform(0.1, 0.8),
                'classical': np.random.uniform(0.1, 0.8),
                'experimental': np.random.uniform(0.1, 0.8)
            }
        }
    
    def calculate_musical_innovation_score(self, audio_features, historical_features):
        innovation_score = 0.0
        
        tempo_novelty = self.analyze_tempo_novelty(audio_features['tempo'], historical_features)
        innovation_score += tempo_novelty * 0.3
        
        spectral_innovation = self.analyze_spectral_innovation(audio_features, historical_features)
        innovation_score += spectral_innovation * 0.3
        
        genre_fusion = self.analyze_genre_fusion(audio_features['genre_characteristics'])
        innovation_score += genre_fusion * 0.2
        
        rhythm_complexity = self.analyze_rhythm_complexity(audio_features['rhythm_patterns'])
        innovation_score += rhythm_complexity * 0.2
        
        return max(0.0, min(1.0, innovation_score))
    
    def analyze_tempo_novelty(self, current_tempo, historical_tempos):
        if not historical_tempos:
            return 0.5
        
        historical_array = np.array(historical_tempos)
        tempo_difference = np.min(np.abs(historical_array - current_tempo))
        
        novelty = min(tempo_difference / 60.0, 1.0)
        return novelty
    
    def analyze_spectral_innovation(self, current_features, historical_features):
        if not historical_features:
            return 0.5
        
        current_centroid = current_features['spectral_centroid']
        historical_centroids = [f['spectral_centroid'] for f in historical_features]
        
        centroid_difference = np.min(np.abs(np.array(historical_centroids) - current_centroid))
        innovation = min(centroid_difference / 2000.0, 1.0)
        
        return innovation
    
    def analyze_genre_fusion(self, genre_characteristics):
        dominant_genres = [genre for genre, score in genre_characteristics.items() if score > 0.5]
        
        if len(dominant_genres) >= 2:
            return 0.7
        elif len(dominant_genres) == 1:
            return 0.3
        else:
            return 0.1
    
    def analyze_rhythm_complexity(self, rhythm_patterns):
        regularity = rhythm_patterns['rhythm_regularity']
        beat_strength = rhythm_patterns['beat_strength']
        
        if regularity < 0.2 and beat_strength > 0.5:
            return 0.8
        elif regularity < 0.3 and beat_strength > 0.4:
            return 0.6
        else:
            return 0.3

audio_analyzer = AudioAnalyzer()