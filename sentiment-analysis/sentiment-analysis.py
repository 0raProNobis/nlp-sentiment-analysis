import csv
import random
import datetime
import spacy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from mlxtend.plotting import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt

nlp = spacy.load('en_core_web_sm')

sentiment2int = {'negative': 0, 'neutral': 1, 'positive': 2}
int2sentiment = list(sentiment2int.keys())
airline2int = {'Virgin America': 0, 'United': 1, 'Southwest': 2,
               'Delta': 3, 'US Airways': 4, 'American': 5}
int2airline = list(airline2int.keys())

tweets_by_airlines = [[[], [], []]] * len(int2airline)

emojis = [
    [':)'],
    [':(']
]

rem_list = [
    "tweet_id", "negativereason", "negativereason_confidence", "airline_sentiment_gold", "negativereason_gold",
    "tweet_coord", "tweet_location", "user_timezone", 'name', 'airline_sentiment_confidence'
]

def expand_features(tweet):
    [tweet.pop(key) for key in rem_list]
    tweet['airline'] = airline2int[tweet['airline']]
    tweet['airline_sentiment'] = sentiment2int[tweet['airline_sentiment']]
    tweet['retweet_count'] = int(tweet['retweet_count'])
    tweet['tweet_created'] = datetime.datetime.strptime(tweet['tweet_created'], "%Y-%m-%d %H:%M:%S %z").timestamp()
    words = tweet['text'].split()
    cleaned_words = []
    for word in words:
        if word[0] != '@':
            cleaned_words.append(word)
    text = " ".join(cleaned_words)
    doc = nlp(text)
    tweet['lemma'] = []
    for token in doc:
        if token.is_punct or token.like_url or not token.is_oov:
            continue
        tweet['lemma'].append(token.lemma_)
    return tweet

with open("Tweets.csv") as csvfile:
    tweets = csv.DictReader(csvfile)
    for row in tweets:
        tweet = expand_features(row)
        tweets_by_airlines[tweet['airline']][tweet['airline_sentiment']].append(row)

# First set is training (50%), second is validation (25%), third is testing (25%)
data = [[], []]
cleaned_words = [[], []]
for airline in tweets_by_airlines:
    for tweet_set in airline:
        i = 0
        while tweet_set:
            ind = random.randint(0, len(tweet_set)-1)
            tweet = tweet_set.pop(ind)
            cleaned_words[i % len(cleaned_words)].append(' '.join(tweet['lemma']))
            data[i % len(data)].append(tweet)
            i = (i + 1) % (len(data) + 1)

v = CountVectorizer(analyzer = 'word')
train_features = v.fit_transform(cleaned_words[0])
test_features = v.transform(cleaned_words[1])

post_rem_list = ['text', 'lemma', 'tweet_created']
final_train_features = []
train_target = []
for i in range(len(data[0])):
    [data[0][i].pop(key) for key in post_rem_list]
    train_target.append(data[0][i].pop('airline_sentiment'))
    temp = []
    temp.extend(data[0][i].values())
    temp.extend(train_features[i].toarray()[0])
    final_train_features.append(temp)

final_test_features = []
test_target = []
for i in range(len(data[1])):
    [data[1][i].pop(key) for key in post_rem_list]
    test_target.append(data[1][i].pop('airline_sentiment'))
    temp = []
    temp.extend(data[1][i].values())
    temp.extend(test_features[i].toarray()[0])
    final_test_features.append(temp)


Classifiers = {
    #DecisionTreeClassifier(),
    #RandomForestClassifier(n_estimators=200)
     }

Accuracy = []
Model = []
for classifier in Classifiers:
    fit = classifier.fit(final_train_features,train_target)
    pred = fit.predict(final_test_features)
    accuracy = accuracy_score(pred,test_target)
    Accuracy.append(accuracy)
    Model.append(classifier.__class__.__name__)
    print('Accuracy of '+classifier.__class__.__name__+' is '+str(accuracy))
    print(classification_report(pred,test_target))
    '''
    cm=confusion_matrix(pred , test_target)
    plt.figure()
    plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Reds)
    plt.xticks([0, 1, 2], ['Negative', 'Neutral', 'Positive'], fontsize=16,color='black')
    plt.yticks([0, 1, 2], ['Negative', 'Neutral', 'Positive'], fontsize=16)
    plt.show()
    '''
