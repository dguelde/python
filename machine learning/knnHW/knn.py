#Modified by Donovan Guelde
#original file: knn HW
#Collaborators: None
#References: online documentation for sklearn (Balltree), Collections(counter), cPickle, and numpy
#CSCI 5622 Machine Learning Fall '16

import argparse
from collections import Counter, defaultdict

import random
import numpy
from numpy import median
from sklearn.neighbors import BallTree
from math import floor #used to determine if majority was found (in majority function)

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()
       


class Knearest:
    """
    kNN classifier
    """

    def __init__(self, x, y, k=5):
        """
        Creates a kNN instance
        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """

        # You can modify the constructor, but you shouldn't need to.
        # Do not use another data structure from anywhere else to
        # complete the assignment.

        self._kdtree = BallTree(x)
        self._y = y
        self._k = k

    def majority(self, item_indices):
        """
        Given the indices of training examples, return the majority label.  If
        there's a tie, return the median of the majority labels (as implemented 
        in numpy).
        :param item_indices: The indices of the k nearest neighbors
        """
        # Finish this function to return the most common y label for
        # the given indices.  The current return value is a placeholder 
        # and definitely needs to be changed. 
        #
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.median.html
        majority=0
        assert len(item_indices) == self._k, "Did not get k inputs"
        resultsArray=[-1]*len(item_indices) #create empty array of the same size as item_indices to store real result, rather than index
        for index in range (0,len(item_indices)):
            resultsArray[index]=self._y[item_indices[index][0]]#translate index to corresponding result
        commonIndex = Counter(resultsArray) #gives most common number
        for number,count in commonIndex.iteritems():
            if count >= (floor(self._k/2))+1: #if there is a clear majority, return that number
                majority = number
            else:
                majority = int(numpy.median(resultsArray)) #if no majority, return median (as int), as per instructions

        return majority

    def classify(self, example):
        """
        Given an example, classify the example.
        :param example: A representation of an example in the same
        format as training data
        """

        # Finish this function to find the k closest points, query the
        # majority function, and return the predicted label.
        # Again, the current return value is a placeholder 
        # and definitely needs to be changed. 
        dist, ind  = self._kdtree.query(example,self._k) #gives k nearest neighbors and distances(distances not used, but its part of the return value...)
        ind = ind.reshape(self._k,1)#change array geometry to match expected in majority function
        classification = self.majority(ind)
        return classification

    def confusion_matrix(self, test_x, test_y):
        """
        Given a matrix of test examples and labels, compute the confusion
        matrix for the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.
        :param test_x: Test data representation
        :param test_y: Test data answers
        """


        # Finish this function to build a dictionary with the
        # mislabeled examples.  You'll need to call the classify
        # function for each example.
        d = defaultdict(dict)
        for index in range (0,10): #pre-fill dictionary so it is of appropriate size/number of key/value pairs
            d[index]={}
            for index2 in range (0,10):
                d[index][index2]=0
            
        data_index = 0
        for xx, yy in zip(test_x, test_y):
            bestGuess = self.classify(test_x[data_index].reshape(1,-1)) #bestGuess is the majority from classification process using knn
            rightAnswer = int(test_y[data_index]) #correct 'real' classification
            d[rightAnswer][bestGuess]+=1 #increment d[real][guess] by 1, correct or incorrect
            data_index += 1
            if data_index % 100 == 0:
                print("%i/%i for confusion matrix" % (data_index, len(test_x)))
        return d

    @staticmethod
    def accuracy(confusion_matrix):
        """
        Given a confusion matrix, compute the accuracy of the underlying classifier.
        """

        # You do not need to modify this function

        total = 0
        correct = 0
        for ii in confusion_matrix:
            total += sum(confusion_matrix[ii].values())
            correct += confusion_matrix[ii].get(ii, 0)

        if total:
            return float(correct) / float(total)
        else:
            return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")

    # You should not have to modify any of this code

    if args.limit > 0:
        print("Data limit: %i" % args.limit)
        knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit],
                       args.k)
    else:
        knn = Knearest(data.train_x, data.train_y, args.k)
    print("Done loading data")

    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    print("\t" + "\t".join(str(x) for x in xrange(10)))
    print("".join(["-"] * 90))
    for ii in xrange(10):
        print("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0))
                                       for x in xrange(10)))
    print("Accuracy: %f" % knn.accuracy(confusion))
    
