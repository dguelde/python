# Donovan Guelde
# CSCI 5622
# Fall 16
# references: online documentation for numpy, python
# collaborators: None

import random
import argparse

from numpy import zeros, sign, amax, amin, argmax, argmin, delete, where
from math import exp, log
from collections import defaultdict


kSEED = 1735
kBIAS = "BIAS_CONSTANT"

random.seed(kSEED)


def sigmoid(score, threshold=20.0):
    """
    Prevent overflow of exp by capping activation at 20.

    :param score: A real valued number to convert into a number between 0 and 1
    """

    if abs(score) > threshold:
        score = threshold * sign(score)

    return 1.0 / (1.0 + exp(-score))


class Example:
    """
    Class to represent a logistic regression example
    """
    def __init__(self, label, words, vocab, df):
        """
        Create a new example

        :param label: The label (0 / 1) of the example
        :param words: The words in a list of "word:count" format
        :param vocab: The vocabulary to use as features (list)
        """
        self.nonzero = {}
        self.y = label
        self.x = zeros(len(vocab))
        for word, count in [x.split(":") for x in words]:
            if word in vocab:
                assert word != kBIAS, "Bias can't actually appear in document"
                self.x[vocab.index(word)] += float(count)
                self.nonzero[vocab.index(word)] = word

        self.x[0] = 1


class LogReg:
    def __init__(self, num_features, lam, eta=lambda x: 0.1):
        """
        Create a logistic regression classifier

        :param num_features: The number of features (including bias)
        :param lam: Regularization parameter
        :param eta: A function that takes the iteration as an argument (the default is a constant value)
        """
        
        self.w = zeros(num_features)
        self.lam = lam
        self.eta = eta
        self.last_update = defaultdict(int)

        assert self.lam>= 0, "Regularization parameter must be non-negative"

    def progress(self, examples):
        """
        Given a set of examples, compute the probability and accuracy

        :param examples: The dataset to score
        :return: A tuple of (log probability, accuracy)
        """

        logprob = 0.0
        num_right = 0
        for ex in examples:
            p = sigmoid(self.w.dot(ex.x))
            if ex.y == 1:
                logprob += log(p)
            else:
                logprob += log(1.0 - p)

            # Get accuracy
            if abs(ex.y - p) < 0.5:
                num_right += 1

        return logprob, float(num_right) / float(len(examples))

    def sg_update(self, train_example, iteration, use_tfidf=False):
        """
        Compute a stochastic gradient update to improve the log likelihood.

        :param train_example: The example to take the gradient with respect to
        :param iteration: The current iteration (an integer)
        :param use_tfidf: A boolean to switch between the raw data and the tfidf representation
        :return: Return the new value of the regression coefficients
        """
        
        # TODO: Implement updates in this function
        """
        if (self.lam == 0): #unreg
            sum = 0
            index=0
            for item in self.w:
                sum+=self.w[index]*train_example.x[index]
                index+=1
            sigm = 1/(1+exp(-sum))
            gradient = train_example.y-sigm
            index = 0
            for item in (train_example.x):
                self.w[index] = self.w[index] + self.eta(1.0)*gradient*item
                index+=1
                
        else: #reg
        """
        shrinkage = (1-2*self.lam*self.eta(iteration))
        if(len(self.last_update)==0): #initialize on first iteration
            sum = 0
            index=0
            for item in self.w:
                sum+=self.w[index]*train_example.x[index]
                index+=1
            #sigm = 1/(1+exp(-sum))
            sigm = sigmoid(sum)
            gradient = train_example.y-sigm
            index = 0
            for item in (train_example.x):
                self.w[index] = self.w[index] + self.eta(iteration)*gradient*item #update w
                if(train_example.x[index]==0 and index>0): #update last_update but not bias term
                    self.last_update[index]+=1
                index+=1
            index=0 #apply shrinkage
            for item in self.w:
                if(self.w[index]!=0 and index>0):
                    self.w[index]=self.w[index]*shrinkage
                index+=1
        else: #subsequent iterations
            sum = 0
            index=0
            for item in self.w:
                sum+=self.w[index]*train_example.x[index]
                index+=1
            #sigm = 1/(1+exp(-sum))
            sigm = sigmoid(sum)
            gradient = train_example.y-sigm

            index=0
            for item in train_example.x: #update w on nonzero x's
                if(train_example.x[index]!=0):
                    self.w[index] = self.w[index] + self.eta(iteration)*gradient*item #update w
                    #self.w[index] = self.w[index] + (gradient*train_example.x[index])
                else:
                    self.last_update[index]+=1
                index+=1
            #regularization:
            index=0
            for item in train_example.x:
                if (train_example.x[index] !=0 and index>0):
                    self.w[index] = self.w[index] * pow(shrinkage,self.last_update[index]+1)
                    self.last_update[index]=0
                index+=1      

                


            

                

            
        return self.w

def read_dataset(positive, negative, vocab, test_proportion=0.1):
    """
    Reads in a text dataset with a given vocabulary

    :param positive: Positive examples
    :param negative: Negative examples
    :param vocab: A list of vocabulary words
    :param test_proprotion: How much of the data should be reserved for test
    """

    df = [float(x.split("\t")[1]) for x in open(vocab, 'r') if '\t' in x]
    vocab = [x.split("\t")[0] for x in open(vocab, 'r') if '\t' in x]
    assert vocab[0] == kBIAS, \
        "First vocab word must be bias term (was %s)" % vocab[0]

    train = []
    test = []
    for label, input in [(1, positive), (0, negative)]:
        for line in open(input):
            ex = Example(label, line.split(), vocab, df)
            if random.random() <= test_proportion:
                test.append(ex)
            else:
                train.append(ex)

    # Shuffle the data 
    random.shuffle(train)
    random.shuffle(test)

    return train, test, vocab

def eta_schedule(iteration):
    # TODO (extra credit): Update this function to provide an
    # EFFECTIVE iteration dependent learning rate size.  
    return 1.0 

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--lam", help="Weight of L2 regression",
                           type=float, default=0.0, required=False)
    argparser.add_argument("--eta", help="Initial SG learning rate",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--positive", help="Positive class",
                           type=str, default="../data/hockey_baseball/positive", required=False)
    argparser.add_argument("--negative", help="Negative class",
                           type=str, default="../data/hockey_baseball/negative", required=False)
    argparser.add_argument("--vocab", help="Vocabulary that can be features",
                           type=str, default="../data/hockey_baseball/vocab", required=False)
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=1, required=False)

    args = argparser.parse_args()
    train, test, vocab = read_dataset(args.positive, args.negative, args.vocab)

    print("Read in %i train and %i test" % (len(train), len(test)))

    # Initialize model
    lr = LogReg(len(vocab), args.lam, lambda x: args.eta)

    # Iterations
    iteration = 0
    for pp in xrange(args.passes):
        random.shuffle(train)
        for ex in train:
            lr.sg_update(ex, iteration)
            if iteration % 5 == 1:
                train_lp, train_acc = lr.progress(train)
                ho_lp, ho_acc = lr.progress(test)
                print("Update %i\tTP %f\tHP %f\tTA %f\tHA %f" %
                      (iteration, train_lp, ho_lp, train_acc, ho_acc))
            iteration += 1
    """
    print "Positive indicators:"
    for index in range(0,5):
        print vocab[argmax(lr.w)],"&",amax(lr.w),"&",exp(amax(lr.w)),"\\\\"
        vocab=delete(vocab,argmax(lr.w))
        lr.w=delete(lr.w,argmax(lr.w))
    print "Negative Indicators:"
    for index in range(0,5):
        print vocab[argmin(lr.w)],"&",amin(lr.w),"&",exp(amin(lr.w)),"\\\\"
        vocab=delete(vocab,argmin(lr.w))
        lr.w=delete(lr.w,argmin(lr.w))
    """
        #print where(lr.w==0)]," zero"
        #lr.w=delete(lr.w,lr.w[where(lr.w==0)])
    #a = where(lr.w==0).tolist()
    print "worthless indicators:"
    a = where(lr.w==0)
    #b = a.tolist()
    for item in a[0]:
        print vocab[item]#,"&",lr.w[item],"&",exp(lr.w[item]),"\\\\"
   