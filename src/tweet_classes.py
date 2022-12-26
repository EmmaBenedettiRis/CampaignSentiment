import tweepy
from src.params import *
from datetime import datetime as dt
import pandas as pd
from langdetect import detect, LangDetectException
import time
import os


# Twitter Authenticator class
class TwitterAuthenticator():

    def auth_twitter_app(self):
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        auth.secure = True
        return auth


# Twitter Client class -- get tweets from a specific user
class TwitterClient():

    def __init__(self, leader: str, leader_username: str, date_from=date_from, date_to=date_to, election_date = election_date):
        self.auth = TwitterAuthenticator().auth_twitter_app()
        self.twitter_client = tweepy.API(self.auth)
        self.leader = leader
        self.leader_username = leader_username
        self.date_from = date_from
        self.date_to = date_to
        self.election_date = election_date

    # find if tweet publication date is between the set interval
    #@staticmethod
    #def between_dates(date_to_test,date_from=date_from, date_to=date_to) -> bool:
    #    return date_to_test >= date_from and date_to_test <= date_to
    
    def get_timeline_tweets(self, num_tweets = 1500, first_clean: bool = False, save: bool = True):

        # extend tweet text
        def extended_tweet(tweet: dict, first_clean = first_clean) -> str:
            """
            Extends tweet text
            """
            text = ''
            if 'extended_tweet' in tweet:
                text = tweet['extended_tweet']['full_text']
            else: text = tweet['full_text']
            return text
        
        # detect language of text
        def detect_language(text) -> str:
            try:
                lang = detect(text)
                if lang[:2]=="it": lang = "italian"
                else: lang = "other"
            except LangDetectException: lang = "other"
            return lang
        
        def collect_tweets(username:str):
            tweets_pre, tweets_post = [], []
            for tweet in tweepy.Cursor(self.twitter_client.user_timeline, screen_name = username, include_rts = False, tweet_mode= "extended").items(num_tweets):
                if detect_language(tweet._json['full_text'])!= "italian": pass
                if tweet._json['created_at'][-4:] == "2022":
                    tweet_time = dt.strptime(f"{tweet._json['created_at'][4:11]} 2022", "%b %d %Y")
                    tweet_text = extended_tweet(tweet._json)
                    if tweet_time >= date_from and tweet_time <= election_date: #between_dates(tweet_time):
                        tweets_pre.append(
                            {   'Date' : tweet_time,
                                'Tweet ID': tweet._json['id'],
                                'Leader Name': tweet.user._json['name'],
                                'Followers': tweet.user._json["followers_count"],
                                'Text': tweet_text,
                                'Hashtags': [d['text'].lower() for d in tweet._json['entities']['hashtags']]
                            }
                        )
                    elif tweet_time > election_date and tweet_time <= date_to:
                        tweets_post.append(
                                {   'Date' : tweet_time,
                                    'Tweet ID': tweet._json['id'],
                                    'Leader Name': tweet.user._json['name'],
                                    'Followers': tweet.user._json["followers_count"],
                                    'Text': tweet_text,
                                    'Hashtags': [d['text'].lower() for d in tweet._json['entities']['hashtags']]
                                }
                            )
            pre, post = pd.DataFrame(tweets_pre), pd.DataFrame(tweets_post)
            
            return pre, post

        # here starts the "actual" function
        print(f"Processing {self.leader}")
        pre, post = collect_tweets(self.leader_username)
        
        return pre, post

        
    @staticmethod
    #save tweets
    def save_tweets(pre: pd.DataFrame, post: pd.DataFrame, leader_name):
        """
        Saves dataframes in CSV files
        """
        leaders_pre, leaders_post = "./2022_pre_elections", "./2022_post_elections"

        paths = [leaders_pre,leaders_post]
        for path in paths:
            if not os.path.exists(path):
                os.mkdir(path)
        
        pre.to_csv(f"{leaders_pre}/{leader_name}.csv", index= False)
        print(f"pre-election df for {leader_name} saved as {leaders_pre}/{leader_name}.csv")
        post.to_csv(f"{leaders_post}/{leader_name}.csv", index= False)
        print(f"post-election df for {leader_name} saved as {leaders_post}/{leader_name}.csv")
        time.sleep(10)
