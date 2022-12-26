# Computational Social Science Final Project
## Sentiment analysis of Italian political candidates during the 2022 election campaign
This repository contains the code used in the Final Project of the Computational Social Science exam, held at the University of Trento (A.Y. 2021-2022). A complete description of the project can be found on the attached [report](https://github.com/EmmaBenedetti/CampaignSentiment/blob/main/Benedetti_CSS_Report.pdf).
## Abstract
This research analyses the Twitter posts of the political candidates of the 2022 Italian election campaign. The goal is to detect the tone of voice used by the candidates, in order to understand the general sentiment behind the text posts, as well as checking for the presence of multiple topics among these posts. To do so, we first collected a total of 2.829 tweets among six political leaders in the span of two months. Then, we retrieved the more relevant text information through the Term Frequency-Inverse Document Frequency (TF-IDF) method. Finally, we analysed the most-recurring keywords through the topic-modelling, Latent Dirichlet Analysis model.
## Prerequisites
In order to run this project, we suggest the following requirements: <br>
* Use Python>=[3.8](https://www.python.org/downloads/release/python-380/) <br>
* A [Twitter Developer](https://developer.twitter.com/en/docs/platform-overview) account, as well as an [active app](https://developer.twitter.com/en/portal/projects-and-apps) <br>
* Save the folder [nltk_data](./nltk_data) in the `share` folder of your virtual environment. 
## Installation
### Clone the Repository
Clone this repository by typing the following link in a CLI of your choice:
```
git clone https://github.com/EmmaBenedetti/CampaignSentiment.git
```
### Create a Virtual Environment
We strongly suggest to create a virtual environment. If not already installed, install virtualenv: <br>
* in Unix systems: <br>
```
python3 -m pip install --user virtualenv
```
* in Windows systems: <br>
```
python -m pip install --user virtualenv
```
Then, create and activate your virual environment:
* in Unix systems: <br>
```
python3 -m venv your_venv_name
source your_venv_name/bin/activate
```
* in Windows systems: <br>
```
python -m venv your_venv_name
your_venv_name/Scripts/activate
```
### Install the Requirements
After activating your virtual environment, install all libraries contained in the `requirements.txt` file.
```
pip install -r requirements.txt
```
### (Facultative) Add your Twitter Developer Credentials
This project will run even if you do not have a Twitter Developer account. However, if you want to run this project from scratch, please keep in mind that for tweet collection you will need to be authorised for OAuth v2. Because of that, please change the following variables in [params.py](https://github.com/EmmaBenedetti/CampaignSentiment/blob/main/src/params.py) with your personal Authentication keys:
```python
consumer_key= "your_consumer_key"
consumer_secret= "your_consumer_secret"
access_token= "your_access_token"
access_token_secret= "your_access_token_secret"
bearer_token= "your_bearer_token"
```
## Repository Structure
The repository is composed by: <br>
* `main_dashboard.ipynb`: the front-end dashboard of the project <br>
* `2022_pre_elections`: folder containing the tweets of political leaders during the two months of election campaign <br>
* `2022_post_elections`: folder containing the tweets of political leaders in the two months following the election date: not used in this project, but added for future modifications of the repository <br>
* `pics`: folder containing the pictures used in the report <br>
* `nltk_data`: folder containing the [NLTK])(https://www.nltk.org/) packages used in this project. To add in the `share` folder of the virtual environment <br>
* `requirements.txt`: collection of libraries used in the project <br>
* `src/get_tweets.py`: module containing the function to read the tweet dataset or collect new tweets within a specified interval <br>
* `src/params.py`: module containing the project parameters <br>
* `src/tweet_classes.py`: classes used for Twitter authentication and scraping <br>
* `src/nlp_class.py`: class used for preprocessing the text and NLP <br>
* `src/text_mining_class.py`: class used for text vectorization and topic modelling <br>
* `src/plots_class.py`: class used for plot and graph drawing <br>
### Overall repository structure:
```bash
├── 2022_post_elections
│   ├── Carlo Calenda.csv
│   ├── Enrico Letta.csv
│   ├── Giorgia Meloni.csv
│   ├── Giuseppe Conte.csv
│   ├── Matteo Renzi.csv
│   ├── Matteo Salvini.csv
│   ├── Silvio Berlusconi.csv
│   └── df_post.csv
├── 2022_pre_elections
│   ├── Carlo Calenda.csv
│   ├── Enrico Letta.csv
│   ├── Giorgia Meloni.csv
│   ├── Giuseppe Conte.csv
│   ├── Luigi Di Maio.csv
│   ├── Matteo Renzi.csv
│   ├── Matteo Salvini.csv
│   ├── Silvio Berlusconi.csv
│   ├── df_pre.csv
│   └── tweets_df_polsco.csv
├── Benedetti_CSS_Report.pdf
├── README.md
├── main_dashboard.ipynb
├── nltk_data
│   ├── corpora
│   │   ├── brown
│   │   │   ├── CONTENTS
│   │   │   ├── README
│   │   │   └── ...
|   |   |
|   |   └── stopwords
|   |       └── ...
│   ├── stemmers
│   │   ├── snowball_data
│   │   │   └── ...
│   └── tokenizers
│       ├── punkt
├── pics
│   ├── lda_topics.png
│   ├── main_centrality.png
│   ├── tweets_elections.png
│   └── wordcloud.png
├── requirements.txt
└── src
    ├── get_tweets.py
    ├── nlp_class.py
    ├── params.py
    ├── plots_class.py
    ├── text_mining_class.py
    └── tweet_classes.py
```
