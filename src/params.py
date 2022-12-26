from nltk import pos_tag, ne_chunk, RegexpParser
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from datetime import datetime as dt

# initialize variables


############################################
############################################
# Twitter credentials. REMEMBER TO DELETE THEM
consumer_key= "your_consumer_key"
consumer_secret= "your_consumer_secret"
access_token= "your_access_token"
access_token_secret= "your_access_token_secret"
bearer_token= "your_bearer_token"
############################################
############################################


party_leaders = {
    "Giorgia Meloni":{
        "party":    {"name": "Fratelli d'Italia",
                    "username": "FratellidItalia"},
        "leaning": "right-wing",
        "username": "GiorgiaMeloni",
    },
    "Matteo Salvini": {
        "party":    {"name": "Lega",
                    "username": "LegaSalvini"},
        "leaning": "right-wing",
        "username": "matteosalvinimi"
    },
    "Silvio Berlusconi": {
        "party":    {"name": "Forza Italia",
                    "username": "forza_italia"},
        "leaning": "right-wing",
        "username": "berlusconi"
    },
    "Enrico Letta": {
        "party":    {"name": "Partito Democratico", 
                    "username": "pdnetwork"},
        "leaning": "left-wing",
        "username": "EnricoLetta"
    },
    "Carlo Calenda": {
        "party":    {"name": "Azione",
                    "username": "Azione_it"},
        "leaning": "left-wing",
        "username": "CarloCalenda"
    },
    "Matteo Renzi": {
        "party":    {"name": "Italia Viva",
                    "username": "ItaliaViva"},
        "leaning": "left-wing",
        "username": "matteorenzi"
    },
    "Luigi Di Maio": {
        "party":    {"name": "Impegno Civico",
                    "username": "impegno_civico"},
        "leaning": "left-wing",
        "username": "luigidimaio"
    },
    "Giuseppe Conte": {
        "party":    {"name": "Movimento 5 Stelle",
                    "username": "Mov5Stelle"},
        "leaning": None,
        "username": "GiuseppeConteIT"
    }
}



date_from, date_to = "2022-07-21", "2022-11-27"
election_date = "2022-09-25"
# convert date strings in datetime
date_from, date_to = dt.strptime(date_from, '%Y-%m-%d'), dt.strptime(date_to, '%Y-%m-%d')
election_date = dt.strptime(election_date, '%Y-%m-%d')

stop_words = set(stopwords.words('italian'))