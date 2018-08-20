def warn(*args, **kwargs): #hide sklearn warnings, http://stackoverflow.com/questions/32612180/eliminating-warnings-from-scikit-learn
    pass
import warnings
warnings.warn = warn



import argparse
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.base import clone 
import matplotlib.pyplot as plt

np.random.seed(1234)

class FoursAndNines:
    """
    Class to store MNIST data
    """

    def __init__(self, location):

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')

        # Split the data set 
        train_set, valid_set, test_set = cPickle.load(f)

        # Extract only 4's and 9's for training set 
        self.x_train = train_set[0][np.where(np.logical_or( train_set[1]==4, train_set[1] == 9))[0],:]
        self.y_train = train_set[1][np.where(np.logical_or( train_set[1]==4, train_set[1] == 9))[0]]
        self.y_train = np.array([1 if y == 9 else -1 for y in self.y_train])
        
        # Shuffle the training data 
        shuff = np.arange(self.x_train.shape[0])
        np.random.shuffle(shuff)
        self.x_train = self.x_train[shuff,:]
        self.y_train = self.y_train[shuff]

        # Extract only 4's and 9's for validation set 
        self.x_valid = valid_set[0][np.where(np.logical_or( valid_set[1]==4, valid_set[1] == 9))[0],:]
        self.y_valid = valid_set[1][np.where(np.logical_or( valid_set[1]==4, valid_set[1] == 9))[0]]
        self.y_valid = np.array([1 if y == 9 else -1 for y in self.y_valid])
        
        # Extract only 4's and 9's for test set 
        self.x_test  = test_set[0][np.where(np.logical_or( test_set[1]==4, test_set[1] == 9))[0],:]
        self.y_test  = test_set[1][np.where(np.logical_or( test_set[1]==4, test_set[1] == 9))[0]]
        self.y_test = np.array([1 if y == 9 else -1 for y in self.y_test])
        
        f.close()

class AdaBoost:
    def __init__(self, n_learners=20, base=DecisionTreeClassifier(max_depth=1)):
        """
        Create a new adaboost classifier.
        
        Args:
            n_learners (int, optional): Number of weak learners in classifier.
            base (BaseEstimator, optional): Your general weak learner 

        Attributes:
            base (estimator): Your general weak learner 
            n_learners (int): Number of weak learners in classifier.
            alpha (ndarray): Coefficients on weak learners. 
            learners (list): List of weak learner instances. 
        """
        
        self.n_learners = n_learners 
        self.base = base
        self.alpha = np.zeros(self.n_learners)
        self.learners = []
        
    def fit(self, X_train, y_train):
        """
        Train AdaBoost classifier on data. Sets alphas and learners. 
        
        Args:
            X_train (ndarray): [n_samples x n_features] ndarray of training data   
            y_train (ndarray): [n_samples] ndarray of data 
        """

        # TODO 
        # Hint: You can create and train a new instantiation 
        # of your sklearn weak learner as follows 

        w = np.ones(len(y_train))
        w = np.divide(w,len(w)) #initialize data weights
        for index in range(0,self.n_learners): #find weights, alphas 
            """if(index>0):
                h=clone(self.learners[index-1])
                self.learners.append(h)
            """
            h = clone(self.base) #new instance of base learner
            self.learners.append(h) 
            self.learners[index].fit(X_train, y_train, sample_weight=w)
            predictions=np.zeros((len(y_train)))
            for index2 in range(0,len(y_train)):
                predictions[index2]=self.learners[index].predict(X_train[index2])
            prediction=0
            Z=0.0
            I=0
            wTemp=0.0
            error=0.0
            for index2 in range (0,len(y_train)):
                if predictions[index2] != y_train[index2]:
                    error+=w[index2]
            error=error/np.sum(w)
            Z = np.sum(w)
          
            self.alpha[index]=.5*np.log((1-error)/error)
            for index2 in range(0,len(y_train)):
                w[index2] = (w[index2]/Z)*np.exp(-1*self.alpha[index]*y_train[index2]*predictions[index2])
                  
    def predict(self, X):
        """
        Adaboost prediction for new data X.
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            
        Returns: 
            [n_samples] ndarray of predicted labels {-1,1}
        """

        # TODO 
        predictions = np.zeros(X.shape[0]) #predictions for all samples in X
        for index in range(0,len(X)): #for each sample
            prediction=0 #prediction of individual sample
            for index2 in range(0,self.n_learners): #iterate through learners
                prediction+=(self.alpha[index2]*self.learners[index2].predict(X[index]))
            if prediction>0:
                predictions[index]=1
            else:
                predictions[index]=-1
        return predictions
    
    def score(self, X, y):
        """
        Computes prediction accuracy of classifier.  
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            y (ndarray): [n_samples] ndarray of true labels  
            
        Returns: 
            Prediction accuracy (between 0.0 and 1.0).
        """
        predictions = self.predict(X)
        sum=0
        for index in range(0,len(y)):
            if predictions[index]==y[index]:
                sum+=1.0
        accuracy=sum/len(y)       
        # TODO 

        return accuracy
    
    def staged_score(self, X_train, y_train):
        """
        Computes the ensemble score after each iteration of boosting 
        for monitoring purposes, such as to determine the score on a 
        test set after each boost.
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            y (ndarray): [n_samples] ndarray of true labels  
            
        Returns: 
            [n_learners] ndarray of scores 
        """

        # TODO
        stagedScore=np.zeros(self.n_learners)
        iterations = self.n_learners
        h = clone(self.base)
        w = np.ones(len(y_train))
        w = np.divide(w,len(w)) #initialize data weights
        self.learners=[]
        self.n_learners=0
        w = np.ones(len(y_train))
        w = np.divide(w,len(w)) #initialize data weights
        for index in range(0,iterations): #find weights, alphas 
            if(index>0):
                h=clone(self.learners[index-1])
            self.learners.append(h)
            self.n_learners=len(self.learners)
            
            self.learners[index].fit(X_train, y_train, sample_weight=w)
            predictions=np.zeros((len(y_train)))
            for index2 in range(0,len(y_train)):
                predictions[index2]=self.learners[index].predict(X_train[index2])
            prediction=0
            Z=0.0
            I=0
            wTemp=0.0
            error=0.0
            for index2 in range (0,len(y_train)):
                if predictions[index2] != y_train[index2]:
                    error+=w[index2]
            error=error/np.sum(w)
            Z = np.sum(w)
            print "prob=",index
            self.alpha[index]=.5*np.log((1-error)/error)
            for index2 in range(0,len(y_train)):
                w[index2] = (w[index2]/Z)*np.exp(-1*self.alpha[index]*y_train[index2]*predictions[index2])
            stagedScore[index]=self.score(X_train,y_train)
        return stagedScore


def mnist_digit_show(flatimage, outname=None):

	import matplotlib.pyplot as plt

	image = np.reshape(flatimage, (-1,28))

	plt.matshow(image, cmap=plt.cm.binary)
	plt.xticks([])
	plt.yticks([])
	if outname: 
	    plt.savefig(outname)
	else:
	    plt.show()

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='AdaBoost classifier options')
	parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
	parser.add_argument('--n_learners', type=int, default=50,
                        help="Number of weak learners to use in boosting")
	args = parser.parse_args()

	data = FoursAndNines("../data/mnist.pkl.gz")

    # An example of how your classifier might be called 
	clf = AdaBoost(n_learners=50, base=DecisionTreeClassifier(max_depth=1, criterion="entropy"))
	clf.fit(data.x_train, data.y_train)


