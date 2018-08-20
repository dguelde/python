

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

def readFile():
	xArray=[]
	yArray=[]
	with open("BTCData.txt") as f:
		f.readline() #skip header
		data=f.readlines()
		counter=0
		oldPrice = 0
		currentPrice = 0
		for line in data:
			dataPoints = line.split(',')
			currentPrice = float(dataPoints[2])
			if(currentPrice > oldPrice):
				trend = [0,0,1] #price goes up
			if(currentPrice == oldPrice):
				trend = [0,1,0] #price flat
			if(currentPrice < oldPrice):
				trend = [1,0,0] #price drop
			xArray.append([float(dataPoints[1]),float(dataPoints[2]),float(dataPoints[3]),float(dataPoints[4]),	\
				float(dataPoints[5]),float(dataPoints[6]),float(dataPoints[7]),float(dataPoints[8])])
			yArray.append(trend)
	f.close()

	return np.array(xArray),np.array(yArray)





def RNN(x, weights, biases,num_hidden,timesteps,useLSTM):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 0)
    print("timesteps = ",timesteps)
    print("x shape = ",tf.shape(x))
    print(x)
    # basic RNN or LSTM cell
    if(useLSTM==False):
        rnn_cell = tf.contrib.rnn.BasicRNNCell(num_hidden)
    else:
        rnn_cell = tf.contrib.rnn.LSTMCell(num_hidden)
    # Get lstm cell output
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


time_steps = 10
num_units = 128
n_input = 8
learning_rate = 0.001
n_classes = 3
batch_size = 65

#defining placeholders
#input vector placeholder
x=tf.placeholder("float",[None,time_steps,n_input])
#input label placeholder
y=tf.placeholder("float",[None,n_classes])

# Define weights
out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
out_bias=tf.Variable(tf.random_normal([n_classes]))


y = tf.nn.tanh(RNN(X,weights,biases,H,timesteps,useLSTM))

loss_op = tf.losses.mean_squared_error(Y,y)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model 
correct_pred = tf.equal(tf.argmax(y), tf.argmax(Y)) 
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

totalAccuracy=[]
totalStandardError=[]
replicationAccuracy = []
for replication in range(10): #train
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        epoch = 1
        bestAccuracySoFar=0
        converged=False
        count = 0
        batch_x, batch_y = readFile()

        testX,testY = readFile()
        sess.run(init)
        #for epoch in range(numEpochs):
        #for step in range(1, training_steps+1):
        while(converged == False):
            randomize = np.arange(len(batch_x))
            np.random.shuffle(randomize)
            batch_x=batch_x[randomize] #arrays have been 'synchronously' shuffled
            batch_y=batch_y[randomize]
            
            if epoch % display_step == 0 or epoch == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,Y: batch_y})
                print("Step " + str(epoch) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
                if(acc>bestAccuracySoFar):
                    bestAccuracySoFar=acc
                    count = 0
                else:
                    count = count +1
                if (count >= 25): 
                    converged = True
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            epoch+=1
        #test on test set after convergance 
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: testX,Y: testY})
        replicationAccuracy.append(acc)
        print(acc)
print(replicationAccuracy)
print("H = ",H)
print("N = ",N)
print("mean accuracy: ",np.mean(replicationAccuracy))
print((np.std(replicationAccuracy,ddof=1)))
print("SEM: ",(np.std(replicationAccuracy,ddof=1)/(10**(.5))))  
