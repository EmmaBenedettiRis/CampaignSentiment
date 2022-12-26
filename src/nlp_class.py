#from datetime import datetime
import pandas as pd
import unicodedata
import re
import numpy as np
#from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem.snowball import ItalianStemmer
from nltk.stem import wordnet, WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#from typing import List
from src.params import *
#import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class NLPClassifier:
    def __init__(self):
        self.stemmer = ItalianStemmer(ignore_stopwords=True)
        self.stopwords = set(stopwords.words("italian"))
        self.lemmatizer = WordNetLemmatizer()
        self.increase_stopwords()

    def get_stopwords(self):
        return self.stopwords
    
    def tokenize_text(self, text: str, stem: bool = True, lem: bool = False, min_len: int =3) -> list or None:
        tokenized_text, res, tmp = word_tokenize(text=text, language="italian"), list(), list()
        for token in tokenized_text:
            if token=='litalia' or token=='unitalia': token='italia' #hard-coded. find alternative ASAP
            if token not in self.stopwords and len(token)>2:
                tmp.append(token)
        if len(tmp) < min_len:
            return None
        if stem: res = [self.stemmer.stem(token) for token in tmp]
        elif lem: res = [self.lemmatizer.lemmatize(token) for token in tmp]
        else: res = tmp
        return res


    def clean_text(self, text: str) ->str:
        def remove_emoji(text):
            re_pattern = re.compile(pattern="["
                                              u"\U0001F600-\U0001F64F"
                                              u"\U0001F300-\U0001F5FF"
                                              u"\U0001F680-\U0001F6FF"
                                              u"\U0001F1E0-\U0001F1FF"
                                              "]+", flags=re.UNICODE)
            return re_pattern.sub(r'', text).replace('\n', ' ')
        
        text = remove_emoji(text)
        for w in text.split(" "):
            if w.startswith("http"): text = text.replace(w, "")
        text = unicodedata.normalize('NFD', text)
        text = text.encode('ascii', 'ignore')
        text = text.decode("utf-8")
        text = text.replace("\n", '')
        text = text.lower()
        text = re.sub('[0-9$%]', ' ', text)
        text = re.sub("[^a-zA-Z;@#]+", ' ', text)
        for iel in range(4, 1, -1):
            text = text.replace(' ' * iel, ' ')
        #if "litalia" in text:
        #    text = text.replace("litalia", "italia")
        text = text.replace('  ', ' ')
        text = text.strip()
        return text

    def process_text_col(self, df: pd.DataFrame, stem: bool, lem: bool, min_len: int = 3) -> pd.DataFrame:
        pol_scores = df["Text"].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x))
        df["Polarity Score"] = pol_scores.apply(lambda d: d["compound"])    # For more info on the polarity score check https://github.com/EmmaBenedetti/vader-multi/blob/master/vaderSentiment/vaderSentiment.py
        df["Tokenized Text"] = df["Text"].apply(self.clean_text)
        df["Tokenized Text"] = df["Tokenized Text"].apply(self.tokenize_text, stem=stem, lem=lem, min_len= min_len)
        #print(type(df["Tokenized Text"]))
        return df[df["Tokenized Text"].notna()]

    @staticmethod
    def frequency_dist(df:pd.DataFrame, obj: str = "tweet") ->FreqDist:
        res= FreqDist()
        if obj=="tweet": bag= df["Tokenized Text"]
        elif obj == "hash": bag = df["Hashtags"]
        else: bag = df
        for text in bag:
            if text:
                for word in text: res[word] += 1
        return res

    def extract_keywords_from_tweets(self, tweets: pd.Series, sets, min_len: int = 1, stem: bool = False):
        def check_key(s, res):
            ret=set()
            for val in s:
                if val in res: ret.add(val)
            if ret: return list(ret)
            else: return None
        
        tokens, final = tweets.apply(self.tokenize_text, stem=stem, min_len=1), list()
        for t in tokens.apply(check_key, res=sets).dropna():
            if len(t) > min_len: final.append(t)
        print(f"Remaining documents: {len(final)}")
        return final

    def increase_stopwords(self) -> None:
        more_stopwords = {'ce', 'fa', 'tanto', 'comunque', 'ecco', 'sempre', 'perche', 'va', 'che', 'boh', 'fra',
                        'del', 'della', 'dello', 'dell', 'degli', 'delle', 'dei',
                        'al', 'alla', 'allo', 'all', 'agli', 'alle', 'ai',
                        'dal', 'dalla', 'dallo', 'dall', 'dagli', 'dalle', 'dai',
                        'col', 'colla', 'col', 'coi',
                        'sul', 'sulla', 'sullo', 'sull', 'sugli', 'sulle', 'sui',
                        'dopo', 'https', 'poi', 'vedere', 'te', 'quest', 'do', 'no', 'pero', 'piu', 'quando', 'state',
                        'adesso', 'ogni', 'so', 'essere', 'tutta', 'senza', 'fatto', 'essere', 'oggi', 'posso', 'tocca', 'vuoi',
                        'altri', 'quindi', 'gran', 'solo', 'ora', 'grazie', 'cosa', 'gia', 'me', '-', 'puoi',
                        'altro', 'prima', 'anno', 'pure', 'qui', 'sara', 'proprio', 'sa', 'de', 'fare', 'silvio', 'roberto', 'renzi', 'calenda', 'buongiorno', 'buonasera',
                        'nuova', 'molto', 'mette', 'dire', 'tali', 'puo', 'uso', 'cioe', 'alta', 'far', 'qualsiasi', 'tanto', 'tanta', 'stasera', 'serata',
                        'cosi', 'chiamano', 'capito', 'mai', 'avere', 'andare', 'invece', 'mesi', 'ancora','rtl', 'soprattutto', 'sopratutto', 'auguri',
                        'invece', 'parli', 'vai','allegri', 'qusta', 'qusto', 'anch', 'prch', 'com', 'snza', 'dir', 'qlli', 'no', 'detto','dice',
                        'qualcuno','qualche','quali', 'ieri','oggi', 'ile','cio','altra','via', 'appero', 'ore', 'facebook', 'sera',
                        'saro', 'intervistato', 'diretta', 'rai', 'zonabianca', 'mezzorainpiu', 'portaaporta', 'amici',
                        'quasi','andra','https','devo','avra','nun','non', 'accounthttps','ecc', 'vuole','sti','qua','neanche','oltre','vuol','chissa',
                        'questo', 'questa', 'seguite', 'seguitemi', 'domani', 'domenicavotolega', 'domanivotolega', #'elezionipolitiche', 'votafdi',
                        'piazza','roma','torino','milano', 'napoli', 'ancona','catania', 'sassari', 'firenze', 'palermo', 'perugia', 'sassari', 'genova',
                        'agosto', 'settembre','nulla','bene','sabato','domenica', 'pochi','anni','molti', 'due','tre', 'cinque', '25', 'vota', 'votare'}
        self.stopwords = self.stopwords.union(more_stopwords)
        self.stopwords = self.stopwords.union(set(map(str.upper, self.stopwords)))
