import csv
import random
import spacy
import pickle
import numpy as np
from mlxtend.plotting import plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
from Data import DataSet, DataVersion, DataMatrix
from sklearn.feature_extraction.text import CountVectorizer

sentiment2int = {'negative': 0, 'neutral': 1, 'positive': 2}
int2sentiment = list(sentiment2int.keys())
airline2int = {'Virgin America': 0, 'United': 1, 'Southwest': 2,
               'Delta': 3, 'US Airways': 4, 'American': 5}
int2airline = list(airline2int.keys())

neut_emoji = ['🇸', '🎀', '🙏', '👏', '💺', '✈', '🍸', '😎', '🎲', '✌', '❄', '🚪', '☀', '😵', '°','🇬', '✨', '🇧',
               '…', '👌', '🌴', '💙', '👉', '🙌', '😱', '🌟', '☕', '📲',  '👀', '👸', '😜']

pos_emoji = [
    '😘', '😁', '💗', ':)', ';)', ':-)', ';-)', '💕', '😊', '😻', '☺', '😆',
    '😍', '😂', '😉', '😀', '❤', '😃', '👍', '🎉', '💯'
]
neg_emoji = [':(', ':-(', ':-/', ':/', '😭', '😢', '😫', '😩', '🌞', '😖', '💔', '🙅']
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


    for e in neut_emoji:
        if e in tweet['text']:
            tweet['lemma'].append(f'-EMOJINEUT-')

    for e in pos_emoji:
        if e in tweet['text']:
            tweet['lemma'].append(f'-EMOJIPOS-')

    for e in neg_emoji:
        if e in tweet['text']:
            tweet['lemma'].append(f'-EMOJINEG-')

    return tweet



train_filepath = 'datasets/v1/training-all.csv'
train_cleantext = []
train_target = []
train_features = []

with open(train_filepath, encoding='utf8') as csvfile:
    training = csv.DictReader(csvfile)
    for tweet in training:
        tweet = expand_features(tweet)
        train_cleantext.append(' '.join(tweet['lemma']))
        train_target.append(tweet['isnegative'])
        train_features.append(tweet)

validate_filepath = 'datasets/v1/validation-all.csv'
valdidate_cleantext = []
validate_target = []
validate_features = []
with open(validate_filepath, encoding='utf8') as csvfile:
    validate = csv.DictReader(csvfile)
    for tweet in validate:
        tweet = expand_features(tweet)
        valdidate_cleantext.append(' '.join(tweet['lemma']))
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

# Base max_df=0.1: (99.7, 70.2) (99.7, 77.5)
# min_df=1  (99.7, 69.9) (99.7, 77.2)
# min_df=2  (99.5, 70.0) (99.5, 77.3)
# min_df=3  (99.5, 70.0) (99.5, 77.1)
# min_df=4  (99.4, 70.0) (99.4, 77.0)
# min_df=5  (99.4, 69.0) (99.4, 77.2)
# min_df=6  (99.3, 69.0) (99.3, 77.2)
# min_df=7  (99.3, 69.4) (99.3, 76.8)
# min_df=10 (99.1, 69.2) (99.1, 76.2)

# max_df=650 (99.2, 69.9) (99.2, 77.3)
# max_df=600 (99.2, 69.9) (99.2, 77.2)
# max_df=550 (99.2, 69.6) (99.2, 77.0)
# max_df=500 (99.2, 70.1) (99.2, 76.9)
# max_df=450 (99.1, 69.2) (99.1, 77.0)
# max_df=400 (99.1, 68.2) (99.1, 77.0)
# max_df=350 (99.1, 69.1) (99.1, 76.6)
# max_df=300 (99.0, 67.9) (99.0, 76.0)
# max_df=250 (98.6, 67.0) (98.6, 75.2)
# max_df=200 (98.5, 67.8) (98.5, 74.4)
# max_df=150 (98.2, 67.2) (98.2, 73.1)
# max_df=100 (97.7, 65.9) (97.7, 72.1)
# max_df=50  (96.1, 62.9) (96.1, 68.3)
v = CountVectorizer(analyzer = 'word', max_df=0.1, min_df=10, ngram_range=(1, 2))
train_vectors = v.fit_transform(train_cleantext)
validate_vectors = v.transform(valdidate_cleantext)
postrain_features = []
posvalidate_features = []
postrain_target = []
posvalidate_target = []

final_train_target = []
final_validate_target = []

for i in range(len(train_features)):
    features = train_vectors[i].toarray()[0].tolist()
    if train_features[i]['isnegative'] == '0':
        postrain_target.append(train_features[i]['ispositive'])
        postrain_features.append(features)
    final_train_target.append(train_features[i]['airline_sentiment'])
    train_features[i] = features
    #[train_features[i].pop(key) for key in rem_list]


for i in range(len(validate_features)):
    features = validate_vectors[i].toarray()[0].tolist()
    if validate_features[i]['isnegative'] == '0':
        posvalidate_target.append(validate_features[i]['ispositive'])
        posvalidate_features.append(features)
    final_validate_target.append(validate_features[i]['airline_sentiment'])
    validate_features[i] = features
    #[validate_features[i].pop(key) for key in rem_list]

###
###
###

neg_Classifiers = [
    RandomForestClassifier(n_estimators=200),
    MLPClassifier(hidden_layer_sizes=(20,15,10,5), random_state=1, max_iter=7),
    SVC(C=2)
]

neg_models = []

pos_Classifiers = [
    RandomForestClassifier(n_estimators=200),
    MLPClassifier(hidden_layer_sizes=(20,15,10,5), random_state=1, max_iter=7),
    SVC(C=2)
]

pos_models = []

Accuracy = []
Model = []

for classifier in neg_Classifiers:
    fit = classifier.fit(train_features,train_target)
    neg_models.append(fit)
    pred_train = fit.predict(train_features)
    accuracy_train = accuracy_score(pred_train,train_target)
    pred_val = fit.predict(validate_features)
    accuracy_val = accuracy_score(pred_val,validate_target)
    Accuracy.append(accuracy_val)
    Model.append(classifier.__class__.__name__)
    print('Accuracy training of '+classifier.__class__.__name__+' is '+str(accuracy_train))
    print('Accuracy validation of '+classifier.__class__.__name__+' is '+str(accuracy_val))
    print(classification_report(pred_val,validate_target))

for classifier in pos_Classifiers:
    fit = classifier.fit(postrain_features, postrain_target)
    pos_models.append(fit)
    pred_train = fit.predict(postrain_features)
    accuracy_train = accuracy_score(pred_train, postrain_target)
    pred_val = fit.predict(posvalidate_features)
    accuracy_val = accuracy_score(pred_val, posvalidate_target)
    Accuracy.append(accuracy_val)
    Model.append(classifier.__class__.__name__)
    print('Accuracy training of ' + classifier.__class__.__name__ + ' is ' + str(accuracy_train))
    print('Accuracy validation of ' + classifier.__class__.__name__ + ' is ' + str(accuracy_val))
    print(classification_report(pred_val, posvalidate_target))
    '''
    cm=confusion_matrix(pred , test_target)
    plt.figure()
    plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Reds)
    plt.xticks([0, 1, 2], ['Negative', 'Neutral', 'Positive'], fontsize=16,color='black')
    plt.yticks([0, 1, 2], ['Negative', 'Neutral', 'Positive'], fontsize=16)
    plt.show()
    '''

with open('forestneg.pkl', 'wb') as fle:
    pickle.dump(neg_models[0], fle)
with open('mlpneg.pkl', 'wb') as fle:
    pickle.dump(neg_models[1], fle)
with open('svcneg.pkl', 'wb') as fle:
    pickle.dump(neg_models[2], fle)

with open('forestpos.pkl', 'wb') as fle:
    pickle.dump(pos_models[0], fle)
with open('mlppos.pkl', 'wb') as fle:
    pickle.dump(pos_models[1], fle)
with open('svcpos.pkl', 'wb') as fle:
    pickle.dump(pos_models[2], fle)

with open('counter.pkl', 'wb') as fle:
    pickle.dump(v, fle)

exit(0)
neg_pred = []

# 78.6% base
# 78.7 only max_df=0.1
# 78.3 No emojis
for fit in neg_models:
    pred_val = fit.predict(validate_features)
    neg_pred.append(pred_val)

pos_pred = []

for fit in pos_models:
    pred_val = fit.predict(validate_features)
    pos_pred.append(pred_val)

### Voting

num_classifiers = len(neg_pred)
voted_pred = [0] * len(neg_pred[0])
for i in range(len(neg_pred[0])):
    neg_votes = {'0': 0, '1': 0}
    pos_votes = {'0': 0, '1': 0}
    for j in range(num_classifiers):
        neg_votes[neg_pred[j][i]] += 1
        pos_votes[pos_pred[j][i]] += 1
    if neg_votes['1'] >= 2:
        '''at least two of three neg models predicted negative'''
        voted_pred[i] = '0'
    elif pos_votes['1'] >= 2:
        '''
        at least two of three neg models predicted nonnegative and at least two of three positive models predict
        positive
        '''
        voted_pred[i] = '2'
    else:
        voted_pred[i] = '1'


accuracy_vote = accuracy_score(voted_pred,final_validate_target)
Accuracy.append(accuracy_vote)
print('Accuracy on validation data of voting is '+str(accuracy_vote))
print(classification_report(voted_pred,final_validate_target))

