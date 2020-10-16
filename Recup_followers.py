## Importation des bibliothèques
#Tweepy est la librairy de fonctions pemettant d'utiliser l'api de twitter depuis python.
from tweepy.streaming import StreamListener

#json est une librairie python qui permet de lire des fichiers json sur python
import json

# Setting up tweepy authentication
# import the module

import os
import csv
import numpy as np

import tweepy
#Code pour l'installation du module tweepy (plus rarement déjà téléchargé)
#import pip
#package_name='tweepy'
#pip.main(['install',package_name])

##Authentification
#Les clefs sont celles du compte associée à mon compte twitter.
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

## Fonction de recherche des followers
#Ici donner le chemin du dossier  où l'on veut enregistrer la liste des followers.
dossier = '/Users/mathias/Desktop/testoun/'
print("Le dossier dans lequel sera téléchargé les id des followers est:", dossier)
def save_followers_status(filename,foloowersid):
    path=dossier+filename
    if not (os.path.isfile(path+'_followers_status.csv')):
      with open(path+'_followers_status.csv', 'wb') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')


    if len(foloowersid)>0:
        print("save followers status of ", filename)
        file = path + '_followers_status.csv'
        # https: // stackoverflow.com / questions / 3348460 / csv - file - written -with-python - has - blank - lines - between - each - row
        with open(file, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for row in foloowersid:
                writer.writerow(np.array(row))
            csv_file.closed

def get_followers_id(person):
    foloowersid = []
    count=0

    influencer=api.get_user( screen_name=person)
    influencer_id=influencer.id
    number_of_followers=influencer.followers_count
    print("number of followers count : ",number_of_followers,'\n','user id : ',influencer_id)
    status = tweepy.Cursor(api.followers_ids, screen_name=person, tweet_mode="extended").items()
    for i in range(0,number_of_followers):
        try:
            user=next(status)
            foloowersid.append([user])
            count += 1
        except tweepy.TweepError:
            print('error limite of twiter sleep for 15 min')
            timestamp = time.strftime("%d.%m.%Y %H:%M:%S", time.localtime())
            print(timestamp)
            if len(foloowersid)>0 :
                print('the number get until this time :', count,'all folloers count is : ',number_of_followers)
                foloowersid = np.array(str(foloowersid))
                save_followers_status(person, foloowersid)
                foloowersid = []
            time.sleep(15*60)
            next(status)
        except :
            print('end of foloowers ', count, 'all followers count is : ', number_of_followers)
            foloowersid = np.array(str(foloowersid))
            save_followers_status(person, foloowersid)
            foloowersid = []
    save_followers_status(person, foloowersid)
    foloowersid = np.array(map(str,foloowersid))
    return(foloowersid)

## Application de la fonction de recherche des followers pour un compte particulier
#Entrer le screen_name de la personne dont on veut connaitre les followers.
screen_name = 'wwatercouncil'
print("Le compte auquel on s'intéresse est %s .Si ce n'est pas le compte souhaité, completer le code avec le screen_name de l'utilisateur dont on veut connaitre des informations sur les followwers"%screen_name)
get_followers_id(screen_name)

#On récupère ici les donées brutes (juste numéro des identifiants qui follow le conseil).

## Mise à jour du tableau des followers (rajout du nombre de followers  pour chacun des followers du conseil)
# followers pour lequels on a deja regardé
import pandas as pd
# Mettre en argument le chemin où est enregistré le tableau csv  des followers dont les informations sont déja connues.
data_path = '/Users/mathias/Desktop/Testfollowers.csv'
print("Le chemin où on est allé chercher la liste des followers déjà connus est :",  data_path)
table_4_know_followers = pd.read_csv(data_path)
known_followers_id =list(table_4_know_followers['follower_id'])
#on transforme les id en entier car la lecture de nombre depuis un fichier csv renvoit un flotant alors que l'on veut des entiers.
for i in range(len(known_followers_id)):
    known_followers_id[i] =int(known_followers_id[i])

# ajout des screen names
followers_screen_names = list(table_4_know_followers['follower_screen_name'])

# ajout des noms
followers_names = list(table_4_know_followers['follower_name'])

# ajout du nombre de followers pour chaque follower
followers_number_of_followers = list(table_4_know_followers['follower_number_of_followers'])

#on reprend les followers enregistrées sur le fichier csv qui contient tous les followers aujourd'hui
table_identites_followers = pd.read_csv(dossier+screen_name+'_followers_status.csv')
followers = list(table_identites_followers[table_identites_followers.columns.values[0]])

#On sort le nombre de nouveaux followers du conseil
liste=[]
for i in range(len(followers)):
    if followers[i] not in known_followers_id :
        liste.append(followers[i])
print('There are',  len(liste) , 'new followers of the', screen_name,'.')
print('La mise à jour des données Twitter est en cours veuillez bien attendre.')

#On met à jour les listes avec les informations sur les nouveaux followers.
for i in range(len(followers)):
    if followers[i] not in known_followers_id :
        liste.append(followers[i])
        follower_info = api.get_user(followers[i])
        known_followers_id.append(follower_info.id)
        followers_screen_names.append(follower_info.screen_name)
        followers_names.append(follower_info.name)
        followers_number_of_followers.append(follower_info.followers_count)

table_followers_complete = pd.DataFrame(known_followers_id, columns = ['follower_id'])
table_followers_complete['follower_screen_name']= followers_screen_names
table_followers_complete['follower_number_of_followers']=followers_number_of_followers
table_followers_complete['follower_name']=followers_names

table_followers_complete=table_followers_complete.sort_values('follower_number_of_followers',ascending=False)

# Aperçu avant enregistrement du tableau ordonné des followers
print("Aperçu des 20 premiers followers trié par nombre décroissant de leur propre followers \n")
print(table_followers_complete.head())

## Sauvegarde finale du nouveau tableau avec les nouveau followers sous format csv
# Mettre en argument un chemin et un nom de fichier pour stocker laliste des followers mise à jour.
path_table_followers_complete =  '/Users/mathias/Desktop/Streaming_24082020/followersinfofinal.csv'
print("La table actualisée des followers va être enregistré à :",path_table_followers_complete)
table_followers_complete.to_csv(path_table_followers_complete)















