import tweepy
import praw
import requests
import json
import time
from datetime import datetime, timedelta
from .config import config

class DataCollector:
    def __init__(self):
        self.twitter_client = self.setup_twitter()
        self.reddit_client = self.setup_reddit()
        self.news_client = self.setup_news_api()
        self.spotify_client = self.setup_spotify()
        self.mwasifanwar = "mwasifanwar"
    
    def setup_twitter(self):
        try:
            auth = tweepy.OAuthHandler(
                config.DATA_SOURCES['twitter']['api_key'],
                'YOUR_TWITTER_API_SECRET'
            )
            auth.set_access_token(
                'YOUR_TWITTER_ACCESS_TOKEN',
                'YOUR_TWITTER_ACCESS_SECRET'
            )
            return tweepy.API(auth, wait_on_rate_limit=True)
        except:
            return None
    
    def setup_reddit(self):
        try:
            return praw.Reddit(
                client_id=config.DATA_SOURCES['reddit']['api_key'],
                client_secret='YOUR_REDDIT_CLIENT_SECRET',
                user_agent='ChronoPredict Cultural Analytics'
            )
        except:
            return None
    
    def setup_news_api(self):
        return config.DATA_SOURCES['news_api']['api_key']
    
    def setup_spotify(self):
        return config.DATA_SOURCES['spotify']['api_key']
    
    def collect_twitter_data(self, query, count=100):
        if not self.twitter_client:
            return self.simulate_twitter_data(query, count)
        
        try:
            tweets = []
            for tweet in tweepy.Cursor(self.twitter_client.search_tweets,
                                     q=query,
                                     tweet_mode='extended',
                                     count=count).items(count):
                tweet_data = {
                    'text': tweet.full_text,
                    'created_at': tweet.created_at,
                    'user': tweet.user.screen_name,
                    'likes': tweet.favorite_count,
                    'retweets': tweet.retweet_count,
                    'source': 'twitter'
                }
                tweets.append(tweet_data)
            return tweets
        except:
            return self.simulate_twitter_data(query, count)
    
    def collect_reddit_data(self, subreddit, limit=100):
        if not self.reddit_client:
            return self.simulate_reddit_data(subreddit, limit)
        
        try:
            posts = []
            subreddit_obj = self.reddit_client.subreddit(subreddit)
            
            for post in subreddit_obj.hot(limit=limit):
                post_data = {
                    'title': post.title,
                    'text': post.selftext,
                    'created_at': datetime.fromtimestamp(post.created_utc),
                    'upvotes': post.score,
                    'comments': post.num_comments,
                    'subreddit': subreddit,
                    'source': 'reddit'
                }
                posts.append(post_data)
            return posts
        except:
            return self.simulate_reddit_data(subreddit, limit)
    
    def collect_news_data(self, query, days=7):
        api_key = self.news_client
        if not api_key:
            return self.simulate_news_data(query, days)
        
        try:
            url = f"https://newsapi.org/v2/everything?q={query}&from={days}daysago&sortBy=popularity&apiKey={api_key}"
            response = requests.get(url)
            articles = response.json().get('articles', [])
            
            news_data = []
            for article in articles:
                article_data = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'published_at': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'url': article.get('url', ''),
                    'source_type': 'news'
                }
                news_data.append(article_data)
            return news_data
        except:
            return self.simulate_news_data(query, days)
    
    def simulate_twitter_data(self, query, count):
        tweets = []
        base_time = datetime.now()
        
        for i in range(count):
            tweet_time = base_time - timedelta(hours=i)
            engagement = np.random.poisson(50)
            
            tweet_data = {
                'text': f"Simulated tweet about {query} with some cultural relevance #{query.replace(' ', '')}",
                'created_at': tweet_time,
                'user': f'user_{i}',
                'likes': engagement,
                'retweets': engagement // 2,
                'source': 'twitter_simulated'
            }
            tweets.append(tweet_data)
        
        return tweets
    
    def simulate_reddit_data(self, subreddit, limit):
        posts = []
        base_time = datetime.now()
        
        for i in range(limit):
            post_time = base_time - timedelta(hours=i*2)
            
            post_data = {
                'title': f"Discussion about {subreddit} and cultural trends",
                'text': f"This is a simulated post about cultural developments in {subreddit}. Interesting patterns emerging.",
                'created_at': post_time,
                'upvotes': np.random.poisson(100),
                'comments': np.random.poisson(25),
                'subreddit': subreddit,
                'source': 'reddit_simulated'
            }
            posts.append(post_data)
        
        return posts
    
    def simulate_news_data(self, query, days):
        articles = []
        base_time = datetime.now()
        
        for i in range(20):
            article_time = base_time - timedelta(days=np.random.randint(0, days))
            
            article_data = {
                'title': f"Cultural Analysis: {query} trends in modern society",
                'description': f"Exploring the emerging trends around {query} and their cultural impact",
                'content': f"Comprehensive analysis of {query} cultural movements and their significance in contemporary discourse.",
                'published_at': article_time.isoformat(),
                'source': 'Simulated News',
                'url': f'http://example.com/{query}_{i}',
                'source_type': 'news_simulated'
            }
            articles.append(article_data)
        
        return articles
    
    def collect_cultural_data(self, topics, time_window='medium_term'):
        all_data = {}
        
        for topic in topics:
            topic_data = {
                'twitter': self.collect_twitter_data(topic, 50),
                'reddit': self.collect_reddit_data(topic, 30),
                'news': self.collect_news_data(topic, config.TIME_WINDOWS[time_window])
            }
            all_data[topic] = topic_data
        
        return all_data

data_collector = DataCollector()