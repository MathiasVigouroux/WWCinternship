## Converting all files from a stream folder to csv tab

## A tweet flattening function
import json
def flatten_tweets(path):
    """ Flattens out tweet dictionaries so relevant JSON
        is in a top-level dictionary."""
    tweets_list = []
    with open(path, 'r') as fh:
        tweets_json = fh.read().split("\n")

    # Iterate through each tweet
    for tweet in tweets_json:
        if len(tweet) > 0:
            tweet_obj = json.loads(tweet)

        # Store the user screen name in 'user-screen_name'
            tweet_obj['user-screen_name'] = tweet_obj['user']['screen_name']

            # Check if this is a 140+ character tweet
            if 'extended_tweet' in tweet_obj:
                # Store the extended tweet text in 'extended_tweet-full_text'
                tweet_obj['extended_tweet-full_text'] = tweet_obj['extended_tweet']['full_text']

            if 'retweeted_status' in tweet_obj:
                # Store the retweet user screen name in 'retweeted_status-user-screen_name'
                tweet_obj['retweeted_status-user-screen_name'] = tweet_obj['retweeted_status']['user']['screen_name']

                # Store the retweet text in 'retweeted_status-text'
                tweet_obj['retweeted_status-text'] = tweet_obj['retweeted_status']['text']

                if 'extended_tweet' in tweet_obj['retweeted_status']:
                    tweet_obj['retweeted_status-extended_tweet-full_text'] = tweet_obj['retweeted_status']['extended_tweet']['full_text']

            if 'quoted_status' in tweet_obj:
                tweet_obj['quoted_status-text'] = tweet_obj['quoted_status']['text']

                if 'extended_tweet' in tweet_obj['quoted_status']:
                    tweet_obj['quoted_status-extended_tweet-full_text']=tweet_obj['quoted_status']['extended_tweet']['full_text']

        tweets_list.append(tweet_obj)
    return tweets_list


## Converting all files from a stream folder to csv tab
# Import pandas and glob
import pandas as pd
import glob

# Give the path of the folder containing all the streaming json files

folder_path = '/Users/mathias/Desktop/Streaming_24082020'
print('The folder path were all the streaming are is :', folder_path)

print('\n','----------------------------------------------','\n')
print('The file conversion started, this operation can take time \n')

# Flatten each streaming file into a python object (list) and upgrade the initial list with all the other dictionnary.
path_list=glob.glob(folder_path+'/*.json')
tweets= flatten_tweets(path_list[0])
n=len(path_list)
if n>1 :#there is more than one streaming file in the folder

    for i in range(1,len(path_list)):
        tweets_i = flatten_tweets(path_list[i])
        tweets+= tweets_i

# Create a DataFrame from `tweets`
ds_tweets = pd.DataFrame(tweets)

# Drop all the text duplicate
ds_tweets = ds_tweets.drop_duplicates(subset = ['text'])

# Print out the first 5 tweets text from this dataset
print('The first 5 tweets text from this dataset are \n')
print(ds_tweets['text'].values[0:5])
print('\n','----------------------------------------------','\n')
## Passing the index to datetime object

# Print created_at to see the original format of datetime in Twitter data
#print('We print created_at to see the original format of datetime in Twitter data')
#print(ds_tweets['created_at'].head())
#print('\n')
#Convert the created_at column to np.datetime object
ds_tweets['created_at'] = pd.to_datetime(ds_tweets['created_at'])

# Print created_at to see new format
#print('We print created_at to see new format')
#print(ds_tweets['created_at'].head())
#print('\n')
# Set the index of ds_tweets to created_at
ds_tweets = ds_tweets.set_index('created_at')
print('We print the head of data frame')
print(ds_tweets.head(-2)) #show the first 5 and last 5 rows of the dataframe

## Selecting the streaming date
import datetime as dt
import numpy as np
import pytz

# Enter the start date and end date (year, month, day, hour, minute,seconds, microseconds) (you have to specify always the sec and micsec)
start_date = dt.datetime(2020, 8, 24, 15, 0, 0, tzinfo=dt.timezone.utc)
end_date = dt.datetime(2020, 8, 24, 22, 0, 0, tzinfo=dt.timezone.utc)
print('La période de stream séléctionnée est entre le %s et le %s'%(start_date,end_date))

#greater than the start date and smaller than the end date
mask = (ds_tweets.index > start_date) & (ds_tweets.index <= end_date)
ds_tweets = ds_tweets.loc[mask] # We replace the complete dataframe with a dataframe containing only the slected dates

print('\n','----------------------------------------------','\n')
print('On passe à la recherche des mots les plus fréquents')
print('\n','----------------------------------------------','\n')
## Finding the most frequent words
ds_tweets_text = ds_tweets[['text',
                            'extended_tweet-full_text',
                            'quoted_status-text',
                            'quoted_status-extended_tweet-full_text',
                            'retweeted_status-text',
                            'retweeted_status-extended_tweet-full_text'
                           ]]

text = ""
nb_row, nb_col = ds_tweets_text.shape
for i in range(nb_row):
    for j in range(nb_col):
        text+=str(ds_tweets_text.iloc[i,j])
import nltk
print('On vérifie le téléchargement de certaines bibiliothèques et on les télécharge sinon.')
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter


tokens = [w for w in word_tokenize(text.lower())if w.isalpha()]
#words to suppress
stop_words = stopwords.words('english') + stopwords.words('spanish')+stopwords.words('french')+['https', 'amp']
no_stops = [t for t in tokens if t not in stop_words]

word_count_processed_text = Counter(no_stops)
word_frequency_table_processed_text = pd.DataFrame(word_count_processed_text.most_common())
word_frequency_table_processed_text.columns = ['Word','Frequency']
print('Voici les 10 mots les plus fréquents')
print(word_frequency_table_processed_text[:10])

print('\n','----------------------------------------------','\n')
print("On passe à l'analyse des sentiments")
print('\n','----------------------------------------------','\n')
## Sentiment Analysis
'''Sentiment analysis provides us a small glimpse of the meaning of texts with a rather directly interpretable method. While it has its limitations, it's a good place to begin working with textual data. There's a number of out-of-the-box tools in Python we can use for sentiment analysis.'''
import nltk
nltk.download('vader_lexicon')

# Load SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Instantiate new SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Generate sentiment scores
sentiment_scores = ds_tweets['text'].apply(sid.polarity_scores)
ds_tweets['sentiment_global']= sentiment_scores
print('\n')
column_name = ds_tweets.columns.values
print('Le nom des colonnes est', column_name)
print('\n')
print('\n')

sentiment_global_index =np.where(column_name == 'sentiment_global')
sentiment_global_index = int(sentiment_global_index[0])
ds_tweets['sentiment_compound']=pd.DataFrame([0 for i in range(len(ds_tweets))])
for i in range(len(ds_tweets)):
    if 'compound' in ds_tweets.iloc[i,sentiment_global_index]:
        dico_sentiment = ds_tweets.iloc[i,sentiment_global_index]
        ds_tweets.iloc[i,sentiment_global_index + 1]=dico_sentiment['compound']
print('Voici la table sélectionnée avec le sentiment associé à chaque tweet en dernière colonne')
print(ds_tweets.head())
print('\n','----------------------------------------------','\n')
print("On passe au regroupement des tweets par mots pour avoir un sentiment pour chaque mot de la liste")
print('\n','----------------------------------------------','\n')

## Regroupement des tweets par mots d'intéret.
'''A rough measure of sentiment towards a particular hashtag is to measure average sentiment for tweets mentioning a particular hashtag. It's also possible that other things are happening in that tweet, so it's important to inspect both text as well as metrics generated by automated text methods.'''


sentiment_compound = ds_tweets['sentiment_compound']

# Print out the text of a positive tweet
print("Text of a positive tweet")
print(ds_tweets[sentiment_compound > 0.6]['text'].values[[0,1,2,3,4,5]])

print('\n','----------------------------------------------','\n')

# Print out the text of a negative tweet
print("Text of a negative tweet")
print(ds_tweets[sentiment_compound <= -0.6]['text'].values[[0,1,2,3,4,5]])

# La fonction suivante est une fonction qui permet de filter les twitt contenant un certain mot parmis les tweets en général.
#elles permettent d'avoir les sentiments moyenné sur la journée des deux hastags mot 1 et mot 2.

#Fonction à utilisé en génral
def check_word_in_tweet(word, data):
    """Checks if a word is in a Twitter dataset's text.
    Checks text and extended tweet (140+ character tweets) for tweets,
    retweets and quoted tweets.
    Returns a logical pandas Series.
    """
    contains_column = data['text'].str.contains(word, case = False)
    contains_column |= data['extended_tweet-full_text'].str.contains(word, case = False)
    contains_column |= data['quoted_status-text'].str.contains(word, case = False)
    contains_column |= data['quoted_status-extended_tweet-full_text'].str.contains(word, case = False)
    contains_column |= data['retweeted_status-text'].str.contains(word, False)
    contains_column |= data['retweeted_status-extended_tweet-full_text'].str.contains(word, False)
    return contains_column

# Un exemple
print('\n','----------------------------------------------','\n')
# Generate average sentiment scores for a selected word
selected_word_for_sentiment_analysis = 'water'
print('The word you selected for sentiment analysis is :',selected_word_for_sentiment_analysis)
# Le sentiment d'un mot est calculé comme la moyenne des sentiments des tweets comprennant ce mot
#Attention c'est valable que pour la langue anglaise malheuresement.
sentiment_selected_word = sentiment_compound[check_word_in_tweet(selected_word_for_sentiment_analysis, ds_tweets)].mean()

print('Le sentiment pour le mot "%s" est de %s '% (selected_word_for_sentiment_analysis,sentiment_selected_word))

print('\n','----------------------------------------------','\n')
print('On passe au regroupement sur toute la table')
words_with_sentiment = []
frequency_words_with_sentiment = []
all_words_sentiment = []
#On refait un tableau plus petit qui contient que les 2000 premiers mots
print("\n Le calcul risque d'être long veuillez bien attendre")
for i in range(2000):
    word=word_frequency_table_processed_text['Word'][i]
    words_with_sentiment.append(word)
    frequency_words_with_sentiment.append(word_frequency_table_processed_text['Frequency'][i])
    sentiment_of_word = sentiment_compound[check_word_in_tweet(word, ds_tweets)].mean()
    all_words_sentiment.append(sentiment_of_word)

table_raccourci_avec_sentiment=pd.DataFrame(words_with_sentiment,columns= ['words'])
frequencies = 'words occurence (%s to %s)'%(start_date,end_date)
table_raccourci_avec_sentiment[frequencies] = frequency_words_with_sentiment
table_raccourci_avec_sentiment['sentiment (between -1 (extremly negative) and +1 (extremely positive))'] = all_words_sentiment

print('\n')

print(table_raccourci_avec_sentiment.head(20))

#sauvegarde des mots avec les sentiments (liste courte)
path_table_avec_sentiment = '/Users/mathias/Desktop/WWC/Frequent_words_with_sentiment.csv'
print('Le tableau comprenant les mots les plus utilisés et le sentiment qui leur est associé est enregistré sous:', path_table_avec_sentiment,'changer dans le code si nécessaire')
table_raccourci_avec_sentiment.to_csv(r'/Users/mathias/Desktop/WWC/Frequent_words_with_sentiment.csv',index=False)

print('\n','----------------------------------------------','\n')
print("On passe à l'analyse sur carte")
print('\n','----------------------------------------------','\n')

## Putting Twitter data on the map
# Creating Basemap map
'''Basemap allows you to create maps in Python. The library builds projections for latitude and longitude coordinates and then passes the plotting work on to matplotlib. This means you can build extra features based on the power of matplotlib.

If basemap isn't download please go to : https://matplotlib.org/basemap/users/installing.html and follow the steps.

Other way : go to anaconda and conda prompt and install trhough the conda command https://anaconda.org/anaconda/basemap'''

import os
import conda

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib

from mpl_toolkits.basemap import Basemap

# Import Basemap
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

# Set up the World bounding box
world_boundingbox = [-180, -70, 180, 80]

# Set up the Basemap object
m = Basemap(llcrnrlon = world_boundingbox[0],
           llcrnrlat = world_boundingbox[1],
            urcrnrlon = world_boundingbox[2],
            urcrnrlat = world_boundingbox[3],
            projection='merc')

# Plotting centroid coordinates
'''Because we can't plot whole bounding boxes, we summarize the bounding box location into a single point called a centroid. Plotting these on a Basemap map is straightforward. Once we calculate the centroids, we separate the longitudes and latitudes, then pass to the .scatter() method.'''


def calculateCentroid(place):
    """ Calculates the centroid from a bounding box."""
    # Obtain the coordinates from the bounding box.
    coordinates = place['bounding_box']['coordinates'][0]

    longs = np.unique( [x[0] for x in coordinates] )
    lats  = np.unique( [x[1] for x in coordinates] )

    if len(longs) == 1 and len(lats) == 1:
        # return a single coordinate
        return (longs[0], lats[0])
    elif len(longs) == 2 and len(lats) == 2:
        # If we have two longs and lats, we have a box.
        central_long = np.sum(longs) / 2
        central_lat  = np.sum(lats) / 2
    else:
        raise ValueError("Non-rectangular polygon not supported.")

    return (central_long, central_lat)

ds_tweets_place= ds_tweets['place'].dropna()
print("The number total of tweet is ", len(ds_tweets))
# On a prit ici le lieu du tweet on pourrait prendre la localisation des twittos.
print("The number of tweets with allowed location is",len(ds_tweets_place))

# Calculate the centroids for the dataset
# and isolate longitudue and latitudes

centroids = ds_tweets_place.apply(calculateCentroid)
lon = [x[0] for x in centroids]
lat = [x[1] for x in centroids]

# Draw continents, coastlines, countries, and states
m.fillcontinents(color='white', zorder = 0)
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')

# Draw the points and show the plot
m.scatter(lon, lat, latlon = True, alpha = 0.7)
print("Représentation de tous les tweets entre le %s et le %s ." %(start_date, end_date))

plt.title("Représentation de tous les tweets entre le %s et le %s ." %(start_date, end_date))
plt.show()


## Coloring by sentiment
print('\n','----------------------------------------------','\n')
print("On passe à la coloration des points \n")
print("Un point rouge correspond à un tweet négatif, un point gris à un tweet neutre, une point bleu à un tweet positif")
# Si on veut voir sur la carte tous les tweets comprtant un certain mot.
# Commandes à passer :
selected_word_for_sentiment_analysis = 'the'
print('The word selected for map sentiment analysis is :',selected_word_for_sentiment_analysis)
ds_tweets_filtre_mot = ds_tweets.loc[check_word_in_tweet(selected_word_for_sentiment_analysis, ds_tweets)]
#Si on veut tous les tweets, il suffit de remplacer la prochaine commande par
#ds_tweets_sentiment_place = ds_tweets[['sentiment_compound','place']]
ds_tweets_sentiment_place = ds_tweets_filtre_mot[['sentiment_compound','place']]
ds_tweets_sentiment_place = ds_tweets_sentiment_place.dropna()

ds_tweets_place_filtered= ds_tweets_filtre_mot['place'].dropna()
print("The number of tweets comporting the word '%s' with allowed location is %s." %(selected_word_for_sentiment_analysis,len(ds_tweets_place)))
# Calculate the centroids for the dataset
# and isolate longitudue and latitudes

centroids = ds_tweets_place_filtered.apply(calculateCentroid)
lon_filtered = [x[0] for x in centroids]
lat_filtered = [x[1] for x in centroids]

sentiment_compound_place = ds_tweets_sentiment_place['sentiment_compound']

# Draw the points
# Si on a cherche à afficher tous les tweets sur la carte,
# il faut remplacer lon_filtered par lon et lat_filtered par lat dans l'arguement de la méthode scatter
sentiment_compound_place = ds_tweets_sentiment_place['sentiment_compound']
m.scatter(lon_filtered, lat_filtered, latlon = True,
           c = sentiment_compound_place,
           cmap= 'coolwarm', alpha = 10000000)
m.fillcontinents(color='white', zorder = 0)
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')


# Show the plot
print("Représentation des tweets comportant le mot '%s' entre le %s et le %s coloriés par sentiments. \nRouge = négatif, gris = neutre, bleu = positif" %(selected_word_for_sentiment_analysis,start_date, end_date))
plt.title("Représentation des tweets comportant le mot '%s' entre le %s et le %s coloriés par sentiments. \nRouge = négatif, gris = neutre, bleu = positif" %(selected_word_for_sentiment_analysis,start_date, end_date))
plt.show()






