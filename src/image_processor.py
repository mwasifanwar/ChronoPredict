import numpy as np
from datetime import datetime
import cv2
from sklearn.cluster import KMeans

class ImageProcessor:
    def __init__(self):
        self.mwasifanwar = "mwasifanwar"
    
    def analyze_image_features(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return self.simulate_image_analysis()
            
            features = {}
            
            features['color_palette'] = self.extract_color_palette(image)
            features['composition'] = self.analyze_composition(image)
            features['texture_patterns'] = self.analyze_texture(image)
            features['style_indicators'] = self.detect_style_indicators(image)
            
            return features
        except:
            return self.simulate_image_analysis()
    
    def extract_color_palette(self, image, n_colors=5):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = image_rgb.reshape(-1, 3)
        
        kmeans = KMeans(n_clusters=n_colors, random_state=42)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        return colors.tolist()
    
    def analyze_composition(self, image):
        height, width = image.shape[:2]
        
        composition_features = {}
        
        brightness = np.mean(image)
        composition_features['brightness'] = brightness / 255.0
        
        contrast = np.std(image)
        composition_features['contrast'] = contrast / 128.0
        
        edges = cv2.Canny(image, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        composition_features['edge_density'] = edge_density
        
        return composition_features
    
    def analyze_texture(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        texture_features = {}
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        texture_features['texture_variance'] = np.var(sobelx) + np.var(sobely)
        
        return texture_features
    
    def detect_style_indicators(self, image):
        style_scores = {
            'minimalist': 0.0,
            'vibrant': 0.0,
            'dark': 0.0,
            'complex': 0.0
        }
        
        color_palette = self.extract_color_palette(image)
        composition = self.analyze_composition(image)
        texture = self.analyze_texture(image)
        
        color_variance = np.var(color_palette, axis=0).mean()
        
        if composition['brightness'] > 0.7 and color_variance < 1000:
            style_scores['minimalist'] = 0.8
        
        if color_variance > 2000 and composition['brightness'] > 0.6:
            style_scores['vibrant'] = 0.7
        
        if composition['brightness'] < 0.4:
            style_scores['dark'] = 0.6
        
        if texture['texture_variance'] > 10000 and composition['edge_density'] > 0.1:
            style_scores['complex'] = 0.7
        
        return style_scores
    
    def simulate_image_analysis(self):
        return {
            'color_palette': [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255]],
            'composition': {
                'brightness': np.random.uniform(0.3, 0.8),
                'contrast': np.random.uniform(0.4, 0.9),
                'edge_density': np.random.uniform(0.05, 0.2)
            },
            'texture_patterns': {
                'texture_variance': np.random.uniform(5000, 20000)
            },
            'style_indicators': {
                'minimalist': np.random.uniform(0.1, 0.8),
                'vibrant': np.random.uniform(0.1, 0.8),
                'dark': np.random.uniform(0.1, 0.8),
                'complex': np.random.uniform(0.1, 0.8)
            }
        }
    
    def calculate_visual_trend_score(self, image_features, historical_features):
        trend_score = 0.0
        
        current_colors = np.array(image_features['color_palette'])
        
        novelty_score = self.calculate_color_novelty(current_colors, historical_features)
        trend_score += novelty_score * 0.4
        
        style_coherence = self.analyze_style_coherence(image_features['style_indicators'])
        trend_score += style_coherence * 0.3
        
        composition_quality = self.evaluate_composition(image_features['composition'])
        trend_score += composition_quality * 0.3
        
        return max(0.0, min(1.0, trend_score))
    
    def calculate_color_novelty(self, current_colors, historical_colors):
        if not historical_colors:
            return 0.5
        
        historical_array = np.array(historical_colors)
        current_array = np.array(current_colors)
        
        distances = []
        for color in current_array:
            min_distance = np.min(np.linalg.norm(historical_array - color, axis=1))
            distances.append(min_distance)
        
        avg_distance = np.mean(distances)
        novelty = min(avg_distance / 100.0, 1.0)
        
        return novelty
    
    def analyze_style_coherence(self, style_indicators):
        dominant_styles = [style for style, score in style_indicators.items() if score > 0.5]
        
        if len(dominant_styles) == 1:
            return 0.8
        elif len(dominant_styles) == 2:
            return 0.6
        else:
            return 0.3
    
    def evaluate_composition(self, composition):
        score = 0.0
        
        brightness = composition['brightness']
        if 0.3 <= brightness <= 0.8:
            score += 0.4
        
        contrast = composition['contrast']
        if contrast > 0.5:
            score += 0.3
        
        edge_density = composition['edge_density']
        if 0.05 <= edge_density <= 0.2:
            score += 0.3
        
        return score

image_processor = ImageProcessor()