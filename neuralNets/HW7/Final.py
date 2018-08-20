from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def readFile(iteration,batch_size,time_steps):
	xArray=np.zeros(shape=(batch_size*time_steps,8))
	yArray=[]
	down=0
	up=0
	same=0
	with open("BTCData.txt") as f:
		f.readline() #skip header
		data=f.readlines()
		counter=0
		counter2=0
		counter3=1
		iteration=iteration-1
		oldPrice = 0
		currentPrice = 0
		for line in data:
			dataPoints = line.split(',')
			currentPrice = float(dataPoints[2])
			if(currentPrice > oldPrice):
				trend = [0,0,1] #price goes up
				up+=1
			if(currentPrice == oldPrice):
				trend = [0,1,0] #price flat
				same+=1
			if(currentPrice < oldPrice):
				trend = [1,0,0] #price drop
				down+=1
			oldPrice = currentPrice
			if(counter >= (iteration*(batch_size*time_steps)) and counter < ((iteration*batch_size*time_steps)+batch_size*time_steps)):
				#print("counter = ",counter,"counter2 = ",counter2)
				xArray[counter2][0]=float(dataPoints[1])
				xArray[counter2][1]=float(dataPoints[2])
				xArray[counter2][2]=float(dataPoints[3])
				xArray[counter2][3]=float(dataPoints[4])
				xArray[counter2][4]=float(dataPoints[5])
				xArray[counter2][5]=float(dataPoints[6])
				xArray[counter2][6]=float(dataPoints[7])
				xArray[counter2][7]=float(dataPoints[8])
				if((counter2+1)%time_steps==0):
					yArray.append(trend)
				counter2+=1
			counter+=1
	f.close()
	#print("up= ",up)
	#print("down = ",down)
	#print("same = ",same)
	return np.array(xArray),np.array(yArray)


def normalizeArray(array):
	minmax=preprocessing.MinMaxScaler()
	normalizedArray = minmax.fit_transform(array)
	return normalizedArray

#define constants
#unrolled through N time steps
time_steps=5
#hidden LSTM units
num_units=10
num_units2 = 10
#rows of 8 items
n_input=8
#learning rate for adam
learning_rate=0.001
#3 classes
n_classes=3
#size of batch
batch_size=5
modByThis=10
numIterations=1000

#weights and biases
out_weights=tf.Variable(tf.random_normal([num_units2,n_classes]))
out_bias=tf.Variable(tf.random_normal([n_classes]))

#defining placeholders
#input image placeholder
x=tf.placeholder("float",[None,time_steps,n_input])

#input label placeholder
y=tf.placeholder("float",[None,n_classes])

input=tf.unstack(x ,time_steps,1)

#defining the network
lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
lstm_layer2 = rnn.BasicLSTMCell(num_units2,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer2,input,dtype="float32")


#converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
prediction=(tf.matmul(outputs[-1],out_weights)+out_bias)


#loss_function
#loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
loss=tf.losses.mean_squared_error(prediction,y)
#optimization
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#model evaluation
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

validAcc=[]
validErr=[]
trainAcc=[]
trainErr=[]
testAcc=0.0
testErr=0.0
iterationCount=0
testCount=0
#initialize variables
tempQ=0;
init=tf.global_variables_initializer()

A = [1,2,3,4,5,6,7,8,9,0]
B = [2,3,4,5,6,7,8,9,0,1]
C = [3,4,5,6,7,8,9,0,1,2]
bigTestAcc=0
bigTestErr=0
bigTrainAcc=np.zeros(numIterations)
bigTrainErr=np.zeros(numIterations)
bigValidationAcc=np.zeros(numIterations)
bigValidationErr=np.zeros(numIterations)


#for bigCounter in range(0,10):

with tf.Session() as sess:
	bestAccuracySoFar=9999
	bigTestAcc=0
	converged=False
	count=0
	testAcc=0.0
	testErr=0.0
	trainAcc=[]
	trainErr=[]
	validAcc=[]
	validErr=[]
	sess.run(init)
	for iteration in range(numIterations):
	#while(converged==False):
		
		tempQ+=1
		iter=1
		modValue=(iter+tempQ)%modByThis
		iterationCount+=1
		
		validCount=0
		trainCount=0
		epochValidErr=0
		epochValidAcc=0
		epochTrainErr=0
		epochTrainAcc=0
		while iter<=(1690/(batch_size*time_steps)):
			if(iter%modByThis!=3 and iter%modByThis!=6 and iter%modByThis!= 9):
				batch_x,batch_y=readFile(iter,batch_size,time_steps)
				tempX = batch_x.T
				tempX = normalizeArray(tempX)
				batch_x = tempX.T
				batch_x=batch_x.reshape((batch_size,time_steps,n_input))
				sess.run(opt, feed_dict={x: batch_x, y: batch_y})
				acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
				los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
				trainCount+=1
				epochTrainAcc+=acc
				epochTrainErr+=los
				
			if(iter%modByThis==3 or iter%modByThis==9): #validate
				batch_x,batch_y=readFile(iter,batch_size,time_steps)
				batch_x,batch_y=readFile(iter,batch_size,time_steps)
				tempX = batch_x.T
				tempX = normalizeArray(tempX)
				batch_x = tempX.T
				batch_x=batch_x.reshape((batch_size,time_steps,n_input))
				sess.run(opt, feed_dict={x: batch_x, y: batch_y})
				acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
				los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
				validCount+=1
				epochValidErr+=los
				epochValidAcc+=acc
				
			iter=iter+1
		validationError=epochValidErr/validCount
		validationAccuracy=epochValidAcc/validCount

		trainingAccuracy = epochTrainAcc/trainCount
		trainingError=epochTrainErr/trainCount

		trainAcc.append(trainingAccuracy)
		trainErr.append(trainingError)

		validAcc.append(validationAccuracy)
		validErr.append(validationError)
	#print(validAcc)
	#print(trainAcc)
	'''
	temp=np.zeros(numIterations)
	temp=np.sum([bigTrainErr,trainErr],axis=0)
	bigTrainErr=temp
	temp=np.zeros(numIterations)
	temp=np.sum([bigTrainAcc,trainAcc],axis=0)
	bigTrainAcc = temp
	temp=np.zeros(numIterations)
	temp=np.sum([bigValidationErr,validErr],axis=0)
	bigValidationErr=temp
	temp=np.zeros(numIterations)
	temp=np.sum([bigValidationAcc,validAcc],axis=0)
	bigValidationAcc=temp
	'''
	# test
	iter=1
	testErr=0
	testAcc=0
	testCount=0
	
	while iter<=(1690/(batch_size*time_steps)):
		
		
		if(iter%modByThis==6):
			batch_x,batch_y=readFile(iter,batch_size,time_steps)
			tempX = batch_x.T
			tempX = normalizeArray(tempX)
			batch_x = tempX.T
			batch_x=batch_x.reshape((batch_size,time_steps,n_input))
			acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
			los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
			testCount+=1
			testAcc+=acc
			testErr+=los
		iter+=1
	testAcc=testAcc / testCount
	print("testAcc = ",testAcc)
	bigTestAcc += (testAcc / testCount)
	print("bigtestacc = ",bigTestAcc)
	bigTestErr += (testErr / testCount)
print(bigTestErr)
print(bigTestAcc)
bigTestErr=bigTestErr/10
bigTestAcc=bigTestAcc/10
bigTrainAcc=np.divide(bigTrainAcc,10)
bigTrainErr=np.divide(bigTrainErr,10)
bigValidationAcc=np.divide(bigValidationAcc,10)
bigValidationErr=np.divide(bigValidationErr,10)
y_av = movingaverage(validAcc, modByThis)
plt.plot(trainAcc,label="Accuracy on Training Set")
plt.plot(validAcc,label="Accuracy on Validation Set")
plt.axhline(y=testAcc,label="Final Accuracy on Test Set")
#plt.plot(y_av,label="Accuracy on Validation Set moving average")
plt.axhline(y=0.520118, color='r', linestyle='-',label="Baseline")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy on Training/Validation Set")
plt.legend(loc=8)
plt.savefig("acc.png")
plt.show()

plt.plot(bigTrainErr,label="Loss on Training Set")
plt.plot(bigValidationErr,label="Loss on Validation Set")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.axhline(y=bigTestErr,label="Loss on Final Test Set")
plt.title("Loss on Training/Validation Set")
plt.legend(loc=8)
plt.savefig("loss.png")
plt.show()






