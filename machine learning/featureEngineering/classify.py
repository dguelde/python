from csv import DictReader, DictWriter

import numpy as np
from numpy import array
import re, string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from nltk import word_tokenize as word_tokenize
from nltk import pos_tag as pos_tag
import nltk
import copy
from nltk.stem.lancaster import LancasterStemmer

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])
"""
def lazy(samples):
    bigrams=np.empty(len(samples),dtype=object)
    counter=0
    for line in samples:
        item=line[kTEXT_FIELD]
        item = [i.lower() for i in word_tokenize(item) if i.lower() not in stop_words]
        item=" ".join(item)
        item=word_tokenize(item)
        item=pos_tag(item)
        
        shittojoin=[]
        for thingy in item:
            temp="".join(thingy)
            shittojoin.append(temp)
        item=" ".join(shittojoin)
        item=word_tokenize(item)
        
        bigram = nltk.trigrams(item)
        sentenceAsBigrams=[]
        for word in bigram:
            word="_".join(word)
            sentenceAsBigrams.append(word)
        sentenceAsBigrams=" ".join(sentenceAsBigrams)
        bigramSentence=sentenceAsBigrams#+" "+line['page']+" "+line['trope']
        bigrams[counter]=bigramSentence
        counter+=1
  
    return bigrams
"""
def lazy(samples):
    processedResults=[]
    for line in samples:
        trope = line['trope']
        item=line[kTEXT_FIELD]
        item = [i.lower() for i in word_tokenize(item) if i.lower() not in stop_words] #remove stopwords
        #item=word_tokenize(item)
        stemmed=[]
        for word in item:
            stemmed.append(LancasterStemmer().stem(word)) #stem words
        item=" ".join(stemmed)
        item=word_tokenize(item)
        item=pos_tag(item) #tag with part of speech
        toJoin=[]
        for word in item:
            temp="".join(word)
            toJoin.append(temp)
        item=" ".join(toJoin)
        item=word_tokenize(item)
        bigram = nltk.trigrams(item) #make bigrams
        sentenceAsBigrams=[]
        for word in bigram:
            word="_".join(word)
            sentenceAsBigrams.append(word)
        item=" ".join(sentenceAsBigrams)
        item=item + " " + trope
        processedResults.append(item)

    return processedResults


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


class POSTransformer(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, examples):
        # return self and nothing else 
        return self
    
    def transform(self, samples):
        POS=np.empty(len(samples),dtype=object)
        counter=0
        for line in samples:

            item=line[kTEXT_FIELD]
            item=item.lower()
            item=item.translate(None,string.punctuation)

            #item=word_tokenize(item)
            #item = [word for word in item if word not in stopwords.words('english')]
            #item = [i.lower() for i in word_tokenize(item) if i.lower() not in stop_words]
            tagged=pos_tag(item)
            sentenceAsPOS=[]
            for word in tagged:
                sentenceAsPOS.append(word[1])

            sentenceAsPOS=" ".join(sentenceAsPOS)
            POS[counter]=sentenceAsPOS
            counter+=1
        
        return POS

class BigramTransformer(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, examples):
        # return self and nothing else 
        return self
    
    def transform(self, samples):
        bigrams=np.empty(len(samples),dtype=object)
        counter=0
        for line in samples:
            item=line[kTEXT_FIELD]
            #item=word_tokenize(item)
            item = [i.lower() for i in word_tokenize(item) if i.lower() not in stop_words]
            bigram = nltk.bigrams(item)
            sentenceAsBigrams=[]
            for word in bigram:
                word="_".join(word)
                sentenceAsBigrams.append(word)
            sentenceAsBigrams=" ".join(sentenceAsBigrams)
            bigrams[counter]=sentenceAsBigrams
            counter+=1
      
        return bigrams




if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))
    feat = Featurizer()
     # remove it if you need punctuation 
    
    

    labels = []
    for line in train:   #train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])
    y_train = array(list(labels.index(x[kTARGET_FIELD])
                         for x in train))   
    
    


    #engineer some features
    """
    allmyfeatures = FeatureUnion([
        #("bigrams", BigramTransformer()),
        ("POS", POSTransformer())
        ])
    
    engineeredTrain = allmyfeatures.fit_transform(train)
    engineeredTest=allmyfeatures.fit_transform(test)
    """
    engineeredTrain=lazy(train)
    engineeredTest=lazy(test)
    print engineeredTrain[0]

    #train on engineered features
    print("Label set: %s" % str(labels))
    x_train = feat.train_feature(x for x in engineeredTrain)
    x_test = feat.test_feature(x for x in engineeredTest)
    
    
    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    feat.show_top10(lr, labels)

    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "spoiler"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'spoiler': labels[pp]}
        o.writerow(d)