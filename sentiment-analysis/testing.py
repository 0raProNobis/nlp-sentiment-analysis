import csv
import pickle
import spacy
import matplotlib.pyplot as plt

from mlxtend.plotting import plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from Data import DataSet, DataVersion, DataMatrix
from sklearn.feature_extraction.text import CountVectorizer

neut_emoji = ['🇸', '🎀', '🙏', '👏', '💺', '✈', '🍸', '😎', '🎲', '✌', '❄', '🚪', '☀', '😵', '°','🇬', '✨', '🇧',
               '…', '👌', '🌴', '💙', '👉', '🙌', '😱', '🌟', '☕', '📲',  '👀', '👸', '😜']

pos_emoji = [
    '😘', '😁', '💗', ':)', ';)', ':-)', ';-)', '💕', '😊', '😻', '☺', '😆',
    '😍', '😂', '😉', '😀', '❤', '😃', '👍', '🎉', '💯'
]
neg_emoji = [':(', ':-(', ':-/', ':/', '😭', '😢', '😫', '😩', '🌞', '😖', '💔', '🙅']
nlp = spacy.load('en_core_web_sm')

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


with open('forestneg.pkl', 'rb') as fle:
    fneg = pickle.load(fle)
with open('mlpneg.pkl', 'rb') as fle:
    mneg = pickle.load(fle)
with open('svcneg.pkl', 'rb') as fle:
    sneg = pickle.load(fle)

with open('forestpos.pkl', 'rb') as fle:
    fpos = pickle.load(fle)
with open('mlppos.pkl', 'rb') as fle:
    mpos = pickle.load(fle)
with open('svcpos.pkl', 'rb') as fle:
    spos = pickle.load(fle)

with open('counter.pkl', 'rb') as fle:
    v = pickle.load(fle)

neg_models = [fneg, mneg, sneg]
pos_models = [fpos, mpos, spos]


test_filepath = 'datasets/v1/validation-all.csv'
cleantext = []
target = []
features = []
with open(test_filepath, encoding='utf8') as csvfile:
    tweets = csv.DictReader(csvfile)
    for tweet in tweets:
        tweet = expand_features(tweet)
        target.append(tweet['airline_sentiment'])
        cleantext.append(' '.join(tweet['lemma']))
        features.append(tweet)

fittedtext = v.transform(cleantext)
for i in range(len(features)):
    features[i] = fittedtext[i].toarray()[0].tolist()

neg_pred = []
for fit in neg_models:
    pred_val = fit.predict(features)
    neg_pred.append(pred_val)

pos_pred = []

for fit in pos_models:
    pred_val = fit.predict(features)
    pos_pred.append(pred_val)

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

accuracy_vote = accuracy_score(voted_pred,target)
print('Accuracy on validation data is '+str(accuracy_vote))
print(classification_report(voted_pred,target))
