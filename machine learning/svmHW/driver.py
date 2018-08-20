# Donovan Guelde
# CSCI 5622 Fall '16
# SVM HW
# script finds a range of values for C that give good results (train with
# data.x_train, validate with data.x_validate), then uses
# the sklearn svm library with this C value on the data.x_test set
# its slow, but does a lot of stuff (varying C values for 3 different kernels)




import argparse
import numpy as np 
import os


def warn(*args, **kwargs): #http://stackoverflow.com/questions/32612180/eliminating-warnings-from-scikit-learn
    pass
import warnings
warnings.warn = warn

from svm import weight_vector, find_support, find_slack
from sklearn.svm import SVC

class ThreesAndEights:
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

		self.x_train = train_set[0][np.where(np.logical_or( train_set[1]==3, train_set[1] == 8))[0],:]
		self.y_train = train_set[1][np.where(np.logical_or( train_set[1]==3, train_set[1] == 8))[0]]

		shuff = np.arange(self.x_train.shape[0])
		np.random.shuffle(shuff)
		self.x_train = self.x_train[shuff,:]
		self.y_train = self.y_train[shuff]

		self.x_valid = valid_set[0][np.where(np.logical_or( valid_set[1]==3, valid_set[1] == 8))[0],:]
		self.y_valid = valid_set[1][np.where(np.logical_or( valid_set[1]==3, valid_set[1] == 8))[0]]
		
		self.x_test  = test_set[0][np.where(np.logical_or( test_set[1]==3, test_set[1] == 8))[0],:]
		self.y_test  = test_set[1][np.where(np.logical_or( test_set[1]==3, test_set[1] == 8))[0]]

		f.close()

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

#this function finds C values that give good results, 
#and validates the training model while doing so.
#using different kernels and varying C values, returns a range of C values to
#the main function, which are applied to the test set in an
#effort to find the highest accuracy on data.x_test
def find_C_values(kernel,data): #a function to find a range of C values that seem to give good results
	if kernel == "linear":
		C_Values = [.01,.1,1,10,100] #good range for linear, found by accident...
	else:
		lastAccuracy = 0.0
		accuracy=0.1 #increase C until lastAccuracy is better than current iteration(because accuracy=.5 at c=1, it only gets better, until it doesn't...)
		C_temp = 8.0 #start here, all kernels except linear have terrible results below, so go up from here
		old_C=1.0
		while(accuracy >= lastAccuracy): #keep updating C until accuracy peaks and starts to drop 
			print "accuracy =",accuracy
			print "last accuracy =",lastAccuracy
			old_C = C_temp
			lastAccuracy=accuracy
			C_temp=C_temp*2
			print "C_temp =",C_temp
			clf = SVC(C=C_temp, kernel=kernel, degree=3) #degree 3 sounds good?
			clf.fit(data.x_train,data.y_train) #training
			counter=0.0
			counter2=0.0
			for index in range (0,len(data.x_valid)): #validate trained SVM, change C if results suck...
				prediction = clf.predict(data.x_valid[index])
				trueValue = data.y_valid[index]
				counter+=1
				if (prediction!= trueValue):
					counter2+=1
			accuracy = (counter-counter2)/counter
			
		print "after loop:"
		print "accuracy =",accuracy
		print "last accuracy =",lastAccuracy
		old_C=old_C/2 #ensures the 'peak accuracy' is somewhere in the range we will return (unless we found a local maximum...)
		interval = (C_temp - old_C) / 4
		C_Values = [0]*5
		for index in range (0,5):
			C_Values[index]=(old_C+(interval*index))
		print C_Values

					

	return C_Values #this function is gonna waste a lot of time, I think.......

if __name__ == "__main__":
	
	kernels = ["rbf","poly","linear"]
	parser = argparse.ArgumentParser(description='SVM classifier options')
	parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
	args = parser.parse_args()
	data = ThreesAndEights("../data/mnist.pkl.gz")
	
	for item in kernels:
		#C_Values = find_C_values(item,data)
		C_Values = [.001,.1,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192]
		newPath = ("./results/Kernel_{0}".format(item))
		for item2 in C_Values:
			#newPath = "./results/Kernel_{0}/C_Value_{1}/supportVectors".format(item,item2)
			#os.makedirs(newPath)
			#newPath = "./results/Kernel_{0}/C_Value_{1}/misclassified_threes".format(item,item2)
			#os.makedirs(newPath)
			#newPath = "./results/Kernel_{0}/C_Value_{1}/misclassified_eights".format(item,item2)
			#os.makedirs(newPath)
			print "make clf, kernel={}, C={}:".format(item,item2)
			clf = SVC(C=item2, kernel=item, degree=3)
			clf.fit(data.x_train,data.y_train) #train
			supportVectors = clf.support_
			outputFile = open("./results/results2.txt","a+") #store results here(text)
			outputFile.write("kernel = {}, C = {}\n".format(item,item2))
			class1_StartingIndex = clf.n_support_[0]
			class0Support = [0]*10
			class1Support = [0]*10
			counter=0
			for index in range(0,10):#save 10 support vectors from each class as examples
					#mnist_digit_show(data.x_train[supportVectors[index]], ("./results/Kernel_{0}/C_Value_{1}/supportVectors/{2}.png".format(item,item2,index)))
					class0Support[index]=supportVectors[index]
					#mnist_digit_show(data.x_train[supportVectors[index + class1_StartingIndex]], ("./results/Kernel_{0}/C_Value_{1}/supportVectors/{2}.png".format(item,item2,index+class1_StartingIndex)))
					class1Support[index]=supportVectors[index+class1_StartingIndex]
			outputFile.write("\t Support Vectors class 0 (by x_train index): {}\n".format(class0Support))
			outputFile.write("\t Support Vectors class 1 (by x_train index): {}\n\n".format(class1Support))
			outputFile.close()
			counter=0.0
			counter2=0.0
			misidentified_threes=0.0
			misidentified_eights=0.0
			total_threes=0.0
			total_eights=0.0
			for index in range(0,len(data.y_test)): #test
				if (data.y_test[index] == 3):
					total_threes+=1
				else:
					total_eights +=1
			misidentified_threes_list = []
			misidentified_eights_list = []
			for index in range (0,len(data.x_test)):
				prediction = clf.predict(data.x_test[index])
				trueValue = data.y_test[index]
				counter+=1
				
				if (prediction!= trueValue):
					counter2+=1
					if (trueValue == 3):
						misidentified_threes+=1
						misidentified_threes_list.append(index)
						#if (misidentified_threes <= 10):
							#mnist_digit_show(data.x_test[index], ("./results/Kernel_{0}/C_Value_{1}/misclassified_threes/{2}.png".format(item,item2,index)))
					else:
						misidentified_eights+=1
						misidentified_eights_list.append(index)
						#if (misidentified_eights <= 10):
							#mnist_digit_show(data.x_test[index], ("./results/Kernel_{0}/C_Value_{1}/misclassified_eights/{2}.png".format(item,item2,index)))
			accuracy = (counter-counter2)/counter
			accuracy_of_threes = (total_threes - misidentified_threes)/total_threes
			accuracy_of_eights = (total_eights - misidentified_eights)/total_eights
			results_string = "\taccuracy = {0}\n\tmisclassified threes = {1}, misclassified eights = {2}\n\taccuracy of threes = {3}, accuracy of eights = {4}\n\ttotal threes = {5}, total eights = {6}\n\n".format(accuracy,misidentified_threes,misidentified_eights,accuracy_of_threes,accuracy_of_eights,total_threes,total_eights,item,item2)
			outputFile = open("./results/results2.txt","a+")
			outputFile.write(results_string)
			outputFile.write("\tmisclassified 3's: {}\n".format(misidentified_threes_list))
			outputFile.write("\tmisclassified 8's: {}\n\n\n\n".format(misidentified_eights_list))
			outputFile.close()
	
	


	# -----------------------------------
	# Plotting Examples 
	# -----------------------------------

	# Display in on screen  
	#mnist_digit_show(data.x_train[ 0,:])

	# Plot image to file 
	#mnist_digit_show(data.x_train[1,:], "mnistfig.png")









