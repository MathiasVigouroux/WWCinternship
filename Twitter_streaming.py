## Récupération de la liste des followers sélectionnés par le Bureau du Conseil.

import pandas as pd
# Mettre en argument le chemin où est enregistré la liste des followers confectionnée par Kevin.
path_followers = "/Users/mathias/Desktop/Copy-of-WWCfollowersinfo-1-revKC.csv"
print("La liste des followers enregistré par Kévin est situé à:",path_followers )
followers = pd.read_csv(path_followers)
if 'Unnamed: 0' in followers.columns.values:
    del(followers['Unnamed: 0'])#supprime l'ancien index
print(followers.head()) #montre les 5 premieres lignes du tableau
print('\n','----------------------------------------------','\n')



liste_selected_followers = [] # contiendra la liste des tweets à suivre.
for i in range(followers.shape[0]):
    if followers.iloc[i,followers.shape[1]-1]=='Y': #j'assume ici que la colonne 'Important' est la dernière des colonnes
        liste_selected_followers.append(str(int(followers['follower_id'][i])))

for i in range(len(liste_selected_followers)):
    liste_selected_followers[i]=str(liste_selected_followers[i])
print('number of selected followers :',len(liste_selected_followers))
print('\n','----------------------------------------------','\n')
print('5 first id of the selected followers', '\n', liste_selected_followers[:5])
# import the modules

import tweepy
#Code pour l'installation du module tweepy (plus rarement déjà téléchargé)
#import pip
#package_name='tweepy'
#pip.main(['install',package_name])

import json
import time
import sys
import os
import csv
import numpy as np


consumer_key = 'dlP4bDmyCe4bDWBHPd5rtX0P6'
consumer_secret = 'xDQ6gu0Jxn3QDQki7ATTmHua2SrPoJ9ajtaL6OUU05wuG2l0gL'

# authorization of consumer key and consumer secret
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

access_token = '1288807957812183040-12hRAGudxmEUXFFr5jSohJmvFJJ1ut'
access_token_secret = 'Dqr3ZNSWvi7VquUHVAeIMxrkBg2QivYFDo5Lea01pCDMk'

# set access to user's access key and access secret
auth.set_access_token(access_token, access_token_secret)

# calling the api
api = tweepy.API(auth)

from tweepy.streaming import StreamListener
import json
import time
import sys

print("Vérifier que le dossier où s'enregistront les streams est bien le bon")
print("/Users/mathias/Desktop/Streaming_24082020/")
class SListener(StreamListener):
    def __init__(self, api = None, fprefix = 'streamer'):
        self.api = api or API()
        self.counter = 0
        self.fprefix = fprefix
        self.output = open(r'/Users/mathias/Desktop/Streaming_24082020/%s_%s.json' % (self.fprefix, time.strftime('%Y%m%d-%H%M%S')), 'w')


    def on_data(self, data):
        if  'in_reply_to_status' in data:
            self.on_status(data)
        elif 'delete' in data:
            delete = json.loads(data)['delete']['status']
            if self.on_delete(delete['id'], delete['user_id']) is False:
                return False
        elif 'limit' in data:
            if self.on_limit(json.loads(data)['limit']['track']) is False:
                return False
        elif 'warning' in data:
            warning = json.loads(data)['warnings']
            print("WARNING: %s" % warning['message'])
            return


    def on_status(self, status):
        self.output.write(status)
        self.counter += 1
        if self.counter >= 20000:
            self.output.close()
            self.output = open(r'/Users/mathias/Desktop/Streaming_24082020/%s_%s.json' % (self.fprefix, time.strftime('%Y%m%d-%H%M%S')), 'w')
            self.counter = 0
        return


    def on_delete(self, status_id, user_id):
        print("Delete notice: User %s has deleted the tweet %s" % (user_id, status_id))
        return


    def on_limit(self, track):
        print("WARNING: Limitation notice received, tweets missed: %d" % track)
        return

    def on_error(self, status_code):
        print (sys.stderr, 'Encountered error with status code:', status_code)
        return True #Don't kill the stream
        print("Stream restarted")


    def on_timeout(self):
        print(sys.stderr, 'Timeout...')
        return True #Don't kill the stream
        print("Stream restarted")

from tweepy import Stream

#liste des mots à tracker faite par fabien.
keywords_to_track = [
                     "Water crisis",
                     "World Water Council",
                     "WWC",
                     "Conseil mondial de l'eau",
                     "World Water Forum",
                     "Dakar 2021",
                     "blended Finance",
                     "enabling Environment"
                     "private sector finance"
                     "public finance","water projects"
                     "innovative financing",
                     "water governance",
                     "water legislative"
                     "water regulation",
                     "integrity"
                     "transparency",
                     "accountability"
                     "science and technology",
                     "education",
                     "right to water",
                     "sanitation",
                     "water quality",
                     "health",
                     "water ecosystems",
                     "aquatic biodiveristy",
                     "natural disasters",
                     "water waste management",
                     "adaptation",
                     "mitigation",
                     "resilience",
                     "sustainale development",
                     "Agenda2030",
                     "SDG13",
                     "#ClimateIsWater",
                     "climate finance",
                     "climate action",
                     "climate change",
                     "water infrastructures",
                     "rural urban devide",
                     "sustainable agriculture",
                     "water productivity",
                     "water efficiency",
                     "water polution reduction",
                     "food loss reduction"
                    ]

## Debut du stream
def start_stream():
    print('Début du stream, checkez le dossier que vous avez spécifié comme chemin pour le stream', '\n')
    while True:
        try:
        # Instantiate the SListener object
            listen = SListener(api)
        # Instantiate the Stream object
            stream = Stream(auth, listen)
        # Begin collecting data
        #Ici on stream tous les followers du conseil,
        #pour passer à un stream sur tout twitter et filter avec un hastag donné
        #remplacer l'argument de stream.filter par stream.filter(track= ['premier mot','second mot'])
        #ou bien stream.filter(track = keywords_to_track)
            stream.filter(follow = liste_selected_followers )
        except:
            print('Pause de 15 min avant le prochain essai de streaming')
            time.sleep(900)
            continue


start_stream()