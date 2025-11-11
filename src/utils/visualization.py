import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta

class VisualizationTools:
    def __init__(self):
        self.mwasifanwar = "mwasifanwar"
        plt.style.use('seaborn-v0_8')
    
    def plot_trend_evolution(self, trend_data, trend_name):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Trend Evolution: {trend_name}', fontsize=16)
        
        timelines = trend_data['timeline']
        dates = [item['date'] for item in timelines]
        mentions = [item['mentions'] for item in timelines]
        sentiment = [item['sentiment'] for item in timelines]
        virality = [item['virality'] for item in timelines]
        
        axes[0, 0].plot(dates, mentions, 'b-', linewidth=2)
        axes[0, 0].set_title('Mention Volume Over Time')
        axes[0, 0].set_ylabel('Number of Mentions')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].plot(dates, sentiment, 'g-', linewidth=2)
        axes[0, 1].set_title('Sentiment Trend')
        axes[0, 1].set_ylabel('Sentiment Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        axes[1, 0].plot(dates, virality, 'r-', linewidth=2)
        axes[1, 0].set_title('Virality Score')
        axes[1, 0].set_ylabel('Virality')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        modality_scores = trend_data['modality_scores']
        modalities = list(modality_scores.keys())
        scores = list(modality_scores.values())
        
        axes[1, 1].bar(modalities, scores, color=['blue', 'green', 'red', 'orange'])
        axes[1, 1].set_title('Modality Contribution')
        axes[1, 1].set_ylabel('Score')
        
        plt.tight_layout()
        return fig
    
    def create_trend_heatmap(self, trends_data, time_period='monthly'):
        trend_names = [trend['name'] for trend in trends_data]
        time_periods = list(set([trend['first_detected'][:7] for trend in trends_data]))
        time_periods.sort()
        
        heatmap_data = np.zeros((len(trend_names), len(time_periods)))
        
        for i, trend in enumerate(trends_data):
            for j, period in enumerate(time_periods):
                if period in trend['first_detected']:
                    heatmap_data[i, j] = trend['confidence']
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, 
                   xticklabels=time_periods,
                   yticklabels=trend_names,
                   cmap='YlOrRd',
                   annot=True,
                   fmt='.2f')
        plt.title('Trend Confidence Heatmap Over Time')
        plt.xlabel('Time Period')
        plt.ylabel('Trends')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_modality_correlation(self, trends_data):
        modalities = ['text', 'image', 'audio', 'social']
        correlation_matrix = np.zeros((len(modalities), len(modalities)))
        
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                scores1 = [trend['modality_scores'].get(mod1, 0) for trend in trends_data]
                scores2 = [trend['modality_scores'].get(mod2, 0) for trend in trends_data]
                correlation = np.corrcoef(scores1, scores2)[0, 1]
                correlation_matrix[i, j] = correlation if not np.isnan(correlation) else 0
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix,
                   xticklabels=modalities,
                   yticklabels=modalities,
                   cmap='coolwarm',
                   center=0,
                   annot=True,
                   fmt='.2f')
        plt.title('Modality Correlation Matrix')
        plt.tight_layout()
        return plt.gcf()
    
    def generate_trend_radar_chart(self, trend_profile):
        categories = ['Momentum', 'Diversity', 'Virality', 'Sentiment', 'Longevity']
        
        values = [
            trend_profile['momentum_score'],
            trend_profile['diversity_score'],
            trend_profile['virality_score'],
            trend_profile['sentiment_score'],
            trend_profile['longevity_score']
        ]
        
        values += values[:1]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title(f'Trend Profile: {trend_profile["name"]}', size=14, weight='bold')
        
        return fig

viz_tools = VisualizationTools()