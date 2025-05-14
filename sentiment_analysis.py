import tweepy
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from config import API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_SECRET

nltk.download('vader_lexicon')

auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth)

def fetch_tweets(query, count=100):
    tweets = tweepy.Cursor(api.search_tweets, q=query, lang='en', tweet_mode='extended').items(count)
    tweet_data = [tweet.full_text for tweet in tweets]
    return pd.DataFrame(tweet_data, columns=['Tweet'])

def analyze_sentiments(df):
    sid = SentimentIntensityAnalyzer()
    sentiments = df['Tweet'].apply(sid.polarity_scores)
    sentiment_df = pd.DataFrame(list(sentiments))
    df = pd.concat([df, sentiment_df], axis=1)
    df['Sentiment'] = df['compound'].apply(lambda x: 'Positive' if x >= 0.05 else 'Negative' if x <= -0.05 else 'Neutral')
    return df

def visualize_sentiment_distribution(df):
    sns.countplot(x='Sentiment', data=df, palette='coolwarm')
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Tweet Count")
    plt.show()

if __name__ == "__main__":
    query = input("Enter keyword to analyze (e.g., 'climate change'): ")
    print(f"Fetching tweets for '{query}'...")
    tweets_df = fetch_tweets(query, count=200)
    analyzed_df = analyze_sentiments(tweets_df)
    print(analyzed_df[['Tweet', 'Sentiment']].head(10))
    visualize_sentiment_distribution(analyzed_df)
