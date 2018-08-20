# Machine Learning Project

from random import random
import csv
from os import listdir
from os.path import isfile, join
import numpy as np
from nltk.corpus import stopwords
import re
from gensim.models import word2vec
import gensim
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from nltk.corpus import gutenberg
from nltk.corpus import reuters
from nltk.corpus import brown
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

USEBIGRAMS=1
VECTORLENGTH=200

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



def GetSamples(files):
	# 80% train, 20% validate
	completeX=[]
	X = []
	y = []
	validateX=[]
	validateY=[]
	for f in files:			
		with open(f, 'r') as csvfile:
			reader = csv.reader(csvfile)
			for row in reader:
				p=random()
				if p<.8:
					X.append(row[1])
					y.append(row[0])
				else:
					validateX.append(row[1])
					validateY.append(row[0])
				completeX.append(row[1])
	
	return (np.array(X), np.array(y), np.array(validateX), np.array(validateY),np.array(completeX))

def RemoveStopWords(stop_words, word_list):
	filtered_text = [
	word for word in word_list if word.lower() not in stop_words]
	return filtered_text

def KeepWordsSpaces(text):
  regex = re.compile(r'[^a-zA-Z ]')
  
  text = regex.sub(' ', text)

  return text

def PreprocessSamples(X):
  stop_words = stopwords.words('english')
  #stemmer = SnowballStemmer('english')
  X_modified = []
  # Preprocess the training examples: remove stop words, stem the words, etc.
  for line in X:
  	line = KeepWordsSpaces(line)
  	line=line.lower()
    # Tokenize the line into a list of words.
	line = line.split()
	line = RemoveStopWords(stop_words, line)
	#line = StemText(stemmer, line)
	# Rejoin the list of tokens in to a single string.
	line = ' '.join(line)
	X_modified.append(line)
  return np.array(X_modified)

def MultinomialNaiveBayes(X, y):
	clf = MultinomialNB()
	print 'Multinomial Naive Bayes score', cross_val_score(clf, X, y).mean()

def DecisionTreeStump(X, y):
	clf = DecisionTreeClassifier(max_depth=2)
	print 'Decision Tree Stump score', cross_val_score(clf, X, y).mean()

def AdaBoostDecisionTree(X, y):
	clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2))
	print 'AdaBoosted Decision Tree score', cross_val_score(clf, X, y).mean()

def PerceptronPrediction(X, y):
	clf = Perceptron()
	print 'Perceptron score', cross_val_score(clf, X, y).mean()

def AdaboostPerceptron(X, y):
	clf = AdaBoostClassifier(Perceptron(), algorithm='SAMME')
	print 'AdaBoosted Perceptron score', cross_val_score(clf, X, y).mean()

def word2vecClassifier(model,x_train,y_train,x_test,y_test):
	"""
	sentences = []
	for entry in X:
		sentence=[]
		for word in entry.split():
			sentence.append(word)
		sentences.append(sentence)
	X=sentences[:]
	#docs_en = [reuters.words(i) for i in reuters.fileids()]
	#texts_en = docs_en # because we loaded tokenized documents in step 1

	if (USEBIGRAMS):
		bigram_transformer = gensim.models.Phrases(X)
		model = word2vec.Word2Vec(bigram_transformer[X],size=VECTORLENGTH, window=5, min_count=1, workers=1,sg=1)
	else:
		model = word2vec.Word2Vec(X,size=VECTORLENGTH, window=5, min_count=1, workers=1,sg=1)
	"""
	forest = RandomForestClassifier( n_estimators = 100 )
	vectorizedTrainingSample=[]
	for line in x_train:
		temp=np.zeros(VECTORLENGTH)
		counter=0.
		for word in line:
			try:
				temp=np.add(temp,model[word])
				counter+=1.
			except:
				pass
		minimum=np.amin(temp)
		temp=np.subtract(temp,minimum)
		vectorizedTrainingSample.append(temp)
	vectorizedTestSet=[]
	for line in x_train:
		temp=np.zeros(VECTORLENGTH)
		counter=0.
		for word in line:
			try:
				temp=np.add(temp,model[word])
				counter+=1.
			except:
				pass
		minimum=np.amin(temp)
		temp=np.subtract(temp,minimum)
		vectorizedTestSet.append(temp)
	forest = forest.fit( vectorizedTrainingSample, y_train)
	predictions = forest.predict( vectorizedTestSet )





def getWord2VecModel(X):
	
	sentences = []
	for entry in X:
		sentence=[]
		for word in entry.split():
			sentence.append(word)
		sentences.append(sentence)
	X=sentences[:]

	if (USEBIGRAMS):
		bigram_transformer = gensim.models.Phrases(X)
		model = word2vec.Word2Vec(bigram_transformer[X],size=VECTORLENGTH, window=5, min_count=1, workers=1,sg=1)
	else:
		model = word2vec.Word2Vec(X,size=VECTORLENGTH, window=5, min_count=1, workers=1,sg=1)
	LCgutenberg=[]
	LCbrown=[]
	
	for line in gutenberg.sents():
		line=RemoveStopWords(stopwords.words('english'),line)
		sentence=[]
		if (len(line)>0):
			for item in line:
				item=item.lower()
				sentence.append(item)
			
			LCgutenberg.append(sentence)
	for line in brown.sents():
		line=RemoveStopWords(stopwords.words('english'),line)
		if (len(line)>0):
			sentence=[]

			for item in line:
				item=item.lower()
				sentence.append(item)
			
			LCbrown.append(sentence)

	#model.train(gutenberg.sents())
	model.train(LCgutenberg)
	model.train(LCbrown)
	"""
	word1=""
	while(word1!="quit"):
		word1=raw_input("word 1:")
		try:
			print "most similar words:",model.similar_by_word(word1,topn=10)
		except:
			pass

	return model
	"""
	"""
	print "word2Vec"
	try:
		MultinomialNaiveBayes(vectorized, y)
	except:
		pass
	try:
		DecisionTreeStump(vectorized, y)
	except:
		pass
	try:
		AdaBoostDecisionTree(vectorized, y)
	except:
		pass
	try:
		PerceptronPrediction(vectorized, y)
	except:
		pass
	try:
		AdaboostPerceptron(vectorized, y)
	except:
		pass
	print "\n"
	"""





if __name__ == "__main__":
	feat = Featurizer()
	samples = []
	for f in listdir('csv_data/'):
		f = join('csv_data/', f)
		if isfile(f):
			samples.append(f)
	unprocessedX, y, unprocessedValidateX, validateY, completeX = GetSamples(samples)
	completeProcessedSamples=PreprocessSamples(completeX)
	model=getWord2VecModel(completeProcessedSamples)

	X=PreprocessSamples(unprocessedX)
	processedValidateX=PreprocessSamples(unprocessedValidateX)

	XValidate=PreprocessSamples(unprocessedValidateX)
	x_train = feat.train_feature(x for x in X)
	x_test = feat.test_feature(x for x in XValidate)
	#x_test_vectorized=word2vecClassifier(XValidate,validateY)
	
	predictions=np.chararray((6,len(validateY)),itemsize=15)
	numberOfClassifiers=0


	NBclassifier=MultinomialNB()
	NBclassifier.fit(x_train,y)
	predictions[numberOfClassifiers] = NBclassifier.predict(x_test)
	numberOfClassifiers+=1

	DTclassifier=DecisionTreeClassifier(max_depth=2)
	DTclassifier.fit(x_train,y)
	predictions[numberOfClassifiers] = DTclassifier.predict(x_test)
	numberOfClassifiers+=1

	ABclassifier=AdaBoostClassifier(DecisionTreeClassifier(max_depth=2))
	ABclassifier.fit(x_train,y)
	predictions[numberOfClassifiers] = ABclassifier.predict(x_test)
	numberOfClassifiers+=1

	Pclassifier=Perceptron()
	Pclassifier.fit(x_train,y)
	predictions[numberOfClassifiers] = Pclassifier.predict(x_test)
	numberOfClassifiers+=1

	APclassifier=AdaBoostClassifier(Perceptron(), algorithm='SAMME')
	APclassifier.fit(x_train,y)
	predictions[numberOfClassifiers] = APclassifier.predict(x_test)
	numberOfClassifiers+=1

	
	predictions[numberOfClassifiers] = word2vecClassifier(model,X,y,processedValidateX,validateY)
	numberOfClassifiers+=1
	
	
	majorityVote=np.chararray(len(validateY))
	for index2 in range(0,len(validateY)):
		#print index2
		voteTally={}
		for item in predictions[:,index2]:
			#print item
			try:
				voteTally[item]+=1
			except:
				voteTally[item]=1
		v=list(voteTally.values())
		k=list(voteTally.keys())
		majority=k[v.index(max(v))]
		majorityVote[index2]=majority
	score=0.
	for index in range(len(majorityVote)):
		#print majorityVote[index],validateY[index]
		if(majorityVote[index]==validateY[index][0]):
			score+=1
	print "score: ",score
	print "accuracy: ",score/len(majorityVote)


	


	

	 



