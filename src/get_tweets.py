from datetime import datetime as dt
import os
from src.tweet_classes import TwitterClient
import pandas as pd
from pandas.errors import EmptyDataError
from src.params import *

def get_tweets_df(save:bool=False, concat:bool=True):
    if "df_pre.csv" in os.listdir("./2022_pre_elections") and "df_post.csv" in os.listdir("./2022_post_elections"):
        df_pre, df_post = pd.read_csv("./2022_pre_elections/df_pre.csv", index_col = 0), pd.read_csv("./2022_post_elections/df_post.csv", index_col = 0)
    else:
        df_pre, df_post = pd.DataFrame(), pd.DataFrame()

        for leader in party_leaders.keys():

            if f'{leader}.csv'in os.listdir("./2022_pre_elections") and f'{leader}.csv'in os.listdir("./2022_post_elections"):
                try: pre = pd.read_csv(f'./2022_pre_elections/{leader}.csv')
                except EmptyDataError: pass
                try: post = pd.read_csv(f'./2022_post_elections/{leader}.csv')
                except EmptyDataError: pass
            else:
                try:
                    leader_username= party_leaders[leader]["username"]
                    twitter_client = TwitterClient(leader=leader, leader_username=leader_username)
                    pre, post = twitter_client.get_timeline_tweets()
                    twitter_client.save_tweets(pre, post, leader)
                except ValueError as e:
                    print(f"A ValueError happened while processing {leader}: {e}")
                    break

            df_pre = pd.concat([df_pre, pre], ignore_index=True)
            df_post = pd.concat([df_post, post], ignore_index=True)
  
        #pprint(twitter_client.get_timeline_tweets()[:5])
        if save:
            df_pre.to_csv("./2022_pre_elections/df_pre.csv")
            df_post.to_csv("./2022_post_elections/df_post.csv")

    print(f"Collected {df_pre.shape[0]} tweets during the electoral campaign")
    #print(f"Collected {df_post.shape[0]} tweets after the electoral campaign")
    
    # remove index column
    for df in [df_pre, df_post]:
        if "Unnamed: 0" in df.columns:
            df = df.drop(["Unnamed: 0"], axis = 1)
        df["Date"] = pd.to_datetime(df["Date"], format='%Y-%m-%d')
        df["Hashtags"] = df["Hashtags"].apply(lambda x: x.replace('[','').replace(']','').replace('\'','').split(', '))

    if concat:
        df_full = pd.concat([df_pre, df_post])
        print(f"Total tweets: {df_full.shape[0]}\n{df_full.shape[1]} columns in total: {list(df_full.columns)}")
        return df_full
    else:
        return df_pre, df_post