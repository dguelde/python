from csv import DictReader, DictWriter

import numpy as np
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from nltk import word_tokenize as word_tokenize
from nltk import pos_tag as pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import nltk
import copy

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'
GOOFINGAROUND=1


def generatePOS(text):
    POS=[]
    for item in text:
        item=word_tokenize(item)
        tagged=pos_tag(item)
        sentenceAsPOS=[]
        for word in tagged:
            sentenceAsPOS.append(word[1])
        sentenceAsPOS=" ".join(sentenceAsPOS)
        POS.append(sentenceAsPOS)
    return POS

def generateBigrams(text):
    bigrams=[]
    for item in text:
        item=word_tokenize(item)
        #tokens=item.split()
        
        bigram = nltk.bigrams(item)
        sentenceAsBigrams=[]
        for word in bigram:
            word="_".join(word)
            sentenceAsBigrams.append(word)
        sentenceAsBigrams=" ".join(sentenceAsBigrams)
        bigrams.append(sentenceAsBigrams)
    return bigrams
def generateWordPOS(text):
    POS=[]
    for item in text:
        item=word_tokenize(item)
        tagged=pos_tag(item)
        sentenceAsPOS=[]
        for word in tagged:
            sentenceAsPOS.append(word[0]+word[1])
        sentenceAsPOS=" ".join(sentenceAsPOS)
        POS.append(sentenceAsPOS)
    return POS

def stripStopWords(text):
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
    stripped=[]
    for item in text:
        item=word_tokenize(item)
        words=[i for i in item if i not in stop_words]
        words=" ".join(words)
        stripped.append(words)
    return stripped



class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-10:]
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            print("Pos: %s" % " ".join(feature_names[top10]))
            print("Neg: %s" % " ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":

    
    

    # Cast to list to keep it all in memory
    
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))
    feat = Featurizer()
    labels = []
    text=[]
    trainLabels=[]
    testSet=[]
    for line in train:
        trainLabels.append(line[kTARGET_FIELD])
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    for line in test:
        testSet.append(line[kTEXT_FIELD])
    for line in train:
        text.append(line[kTEXT_FIELD])



    if GOOFINGAROUND==1: #use holdout set to play with features
        validation=copy.deepcopy(text[10000:])
        validationLabels=copy.deepcopy(trainLabels[10000:])
        text=copy.deepcopy(text[:10000])
        textLabels=copy.deepcopy(trainLabels[:10000])
    else: #use full set
        validation=copy.deepcopy(testSet)
        textLabels=trainLabels


    
    
    temp=stripStopWords(text)
    tempv=stripStopWords(validation)
    temp2=generatePOS(temp)
    tempv2=generatePOS(tempv)
    temp3=generateBigrams(temp2)
    tempv3=generateBigrams(tempv2)
    text=temp3
    validation=tempv3
    
    

    print text[0]
    print validation[0]
    
    
   
    
    x_train = feat.train_feature(x for x in text)

    print("Label set: %s" % str(labels))
    #x_train = feat.train_feature(x for x in text)
    x_test = feat.test_feature(x for x in validation)

    y_train = array(list(x for x in textLabels))
    


    print(len(text), len(y_train))
    print(set(y_train))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    feat.show_top10(lr, labels)

    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "spoiler"])
    o.writeheader()
    #for ii, pp in zip([x['id'] for x in test], predictions):
    #for ii, pp in zip([x['id'] for x in test], predictions):
    #    d = {'id': ii, 'spoiler': labels[pp]}
    #    o.writerow(d)
    counter=0
    for index in xrange(len(validation)):
        d={'id':index,'spoiler':predictions[index]}
        o.writerow(d)