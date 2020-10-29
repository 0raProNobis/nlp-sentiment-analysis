import csv
import random
import spacy
import pickle
import numpy as np
from mlxtend.plotting import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
from Data import DataSet, DataVersion, DataMatrix
from sklearn.feature_extraction.text import CountVectorizer

sentiment2int = {'negative': 0, 'neutral': 1, 'positive': 2}
int2sentiment = list(sentiment2int.keys())
airline2int = {'Virgin America': 0, 'United': 1, 'Southwest': 2,
               'Delta': 3, 'US Airways': 4, 'American': 5}
int2airline = list(airline2int.keys())

emojis = ['😩', '🌞', '🇸', '🎀', '😀', '🙏', '👏', '💺', '😖', '😊', '😻', '✈', '👍', '🍸', '😎', '🎲', '✌', '💯',
          '😘', '❄', '😁', '🙅', '💗', '🚪', '☀', '🎉', '😵', '°', '😃', '🇬', '💔', '✨', '☺', '🇧', '😆', '…', '😍',
          '😫', '👌', '❤', '😢', '🌴', '💙', '👉', '💕', '🙌', '😱', '🌟', '☕', '📲', '😉', '👀', '😭', '👸', '😜',
          '😂', ':)', ':(', ';)', ':-)', ':-(', ';-)', ':-/', ':/']
###
###
###
nlp = spacy.load('en_core_web_sm')

sentiment2int = {'negative': 0, 'neutral': 1, 'positive': 2}
int2sentiment = list(sentiment2int.keys())
airline2int = {'Virgin America': 0, 'United': 1, 'Southwest': 2,
               'Delta': 3, 'US Airways': 4, 'American': 5}
int2airline = list(airline2int.keys())

tweets_by_airlines = [[[], [], []]] * len(int2airline)

rem_list = [
    "tweet_id", "negativereason", "negativereason_confidence", "airline_sentiment_gold", "negativereason_gold",
    "tweet_coord", "tweet_location", "user_timezone", 'name', 'airline_sentiment_confidence', 'text', 'lemma',
    'airline', 'dep_bigram', 'airline_sentiment', 'isnegative', 'ispositive'
]

def expand_features(tweet):
    words = tweet['text'].split()
    tweet['countvector'] = None
    tweet['dep_bigram'] = []
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
        #tweet['dep_bigram'].append(f'{token.lemma_}{token.head.lemma_}')
    '''
    for i in range(len(emojis)):
        emoji = emojis[i]
        if emoji in tweet['text']:
            tweet['lemma'].append(f'-EMOJI{i}-')
    return tweet
    '''

train_filepath = 'datasets/v1/training-all.csv'
train_cleantext = []
train_target = []
train_features = []

with open(train_filepath, encoding='utf8') as csvfile:
    training = csv.DictReader(csvfile)
    for tweet in training:
        training = expand_features(tweet)
        train_cleantext.append(' '.join(tweet['lemma']) + ' ' + ' '.join(tweet['dep_bigram']))
        train_target.append(tweet['isnegative'])
        train_features.append(tweet)

validate_filepath = 'datasets/v1/validation-all.csv'
valdidate_cleantext = []
validate_target = []
validate_features = []
with open(validate_filepath, encoding='utf8') as csvfile:
    validate = csv.DictReader(csvfile)
    for tweet in validate:
        validate = expand_features(tweet)
        valdidate_cleantext.append(' '.join(tweet['lemma']) + ' ' + ' '.join(tweet['dep_bigram']))
        validate_target.append(tweet['isnegative'])
        validate_features.append(tweet)

# Stop words don't have much of an effect
# max_df=0.75 (99.7, 70.0) (99.7, 76.0)
# max_df=0.5 (99.7, 70.5) (99.7, 76.0)
# max_df=0.3 (99.7, 70.4) (99.7, 76.3)
# max_df=0.1 (99.66, 70.6) (99.67, 77.7)
# max_df=0.07 (99.3, 66.2) (99.4, 73.8)
# max_df=0.05 (99.2, 64.9) (99.3, 73.0)
# max_df=0.01 (95.9, 58.6) (95.9, 63.3)
v = CountVectorizer(analyzer = 'word', max_df=0.1)
train_vectors = v.fit_transform(train_cleantext)
validate_vectors = v.transform(valdidate_cleantext)

for i in range(len(train_features)):
    '''
    if data[i][j]['isnegative']:
        continue
    '''
    train_features[i] = train_vectors[i].toarray()[0].tolist()
    #[train_features[i].pop(key) for key in rem_list]


for i in range(len(validate_features)):
    '''
    if data[i][j]['isnegative']:
        continue
    '''
    validate_features[i] = validate_vectors[i].toarray()[0].tolist()
    #[validate_features[i].pop(key) for key in rem_list]

###
###
###

Classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=200)
]

Accuracy = []
Model = []
training_predictions = []
validation_predictions = []
for classifier in Classifiers:
    fit = classifier.fit(train_features,train_target)
    pred_train = fit.predict(train_features)
    training_predictions.append(pred_train)
    accuracy_train = accuracy_score(pred_train,train_target)
    pred_val = fit.predict(validate_features)
    validation_predictions.append(pred_val)
    accuracy_val = accuracy_score(pred_val,validate_target)
    Accuracy.append(accuracy_val)
    Model.append(classifier.__class__.__name__)
    print('Accuracy training of '+classifier.__class__.__name__+' is '+str(accuracy_train))
    print('Accuracy validation of '+classifier.__class__.__name__+' is '+str(accuracy_val))
    print(classification_report(pred_val,validate_target))
    '''
    cm=confusion_matrix(pred , test_target)
    plt.figure()
    plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Reds)
    plt.xticks([0, 1, 2], ['Negative', 'Neutral', 'Positive'], fontsize=16,color='black')
    plt.yticks([0, 1, 2], ['Negative', 'Neutral', 'Positive'], fontsize=16)
    plt.show()
    '''

### Voting
num_classifiers = len(validation_predictions)
voted_pred = [0] * num_classifiers
for i in range(num_classifiers):
    for j in range(len(validation_predictions[0])):
        voted_pred += (validation_predictions[i][j] / num_classifiers)

for i in range(len(voted_pred)):
        if voted_pred[i] > 0.5:
            voted[i] = 1
        else:
            voted[i] = 0


accuracy_vote = accuracy_score(voted_pred,validate_target)
Accuracy.append(accuracy_vote)
print('Accuracy on validation data of voting is '+str(accuracy_val))
print(classification_report(voted_pred,validate_target))