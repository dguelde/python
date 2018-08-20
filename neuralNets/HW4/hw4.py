'''
Tyler Scott
CSCI 5922
Homework 4

Process CIFAR-10 data.
Reference: https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/cifar10.py
'''

import tensorflow as tf
import os
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data_path = 'cifar-10-batches-py/'
img_size = 32
num_channels = 3
num_classes = 10

num_training_files = 5
training_images_per_file = 10000
num_training_images = num_training_files * training_images_per_file

def load_data(filename):
	file_path = os.path.join(data_path, filename)
	print('Loading data:', file_path)

	with open(file_path, 'rb') as f:
		data = pickle.load(f, encoding='bytes')

	raw_images = data[b'data']
	class_num = np.array(data[b'labels'])

	raw_float_images = np.array(raw_images, dtype=float) / 255.0
	images = raw_float_images.reshape([-1, num_channels, img_size, img_size])
	images = images.transpose([0, 2, 3, 1])

	return images, class_num

def training_data():
	images = np.zeros(shape=[num_training_images, img_size, img_size, num_channels], dtype=float)
	class_num = np.zeros(shape=[num_training_images], dtype=int)

	begin = 0

	for i in range(num_training_files):
		images_batch, class_batch = load_data(filename='data_batch_' + str(i + 1))

		num_images = len(images_batch)
		images[begin:begin + num_images, :] = images_batch
		class_num[begin:begin + num_images] = class_batch

		begin += num_images

	return images, class_num, np.eye(num_classes, dtype=float)[class_num]

def test_data():
	images, class_num = load_data(filename="test_batch")
	return images, class_num, np.eye(num_classes, dtype=float)[class_num]

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

if __name__ == "__main__":
	batchSize=500
	numEpochs = 10;
	input_size = 32*32
	classes_num = 10
	i = 0; 
	trainData = training_data();
	testData = test_data();
	train_X = trainData[0]
	train_Y = trainData[2]
	test_X = testData[0]
	test_Y = testData[2]
	validationAccuracy=[]
	trainAccuracy = []


	x = tf.placeholder(tf.float32,shape=[None,32,32,3])
	y = tf.placeholder(tf.int64, shape = [None, classes_num])
	
	W_conv1 = weight_variable([3, 3, 3, 32])
	b_conv1 = bias_variable([32])

	x_image = tf.reshape(x, [-1, 32, 32, 3])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

	W_conv2 = weight_variable([3, 3, 32, 64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

	W_conv3 = weight_variable([3, 3, 64, 64])
	b_conv3 = bias_variable([64])
	h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)

	W_conv4 = weight_variable([3, 3, 64, 64])
	b_conv4 = bias_variable([64])
	h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

	W_conv5 = weight_variable([3, 3, 64, 64])
	b_conv5 = bias_variable([64])
	h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)


	h_pool1 = max_pool_2x2(h_conv5)

	

	
	h_pool2 = max_pool_2x2(h_pool1)

	



	W_fc1 = weight_variable([8 * 8 * 64, 64])
	b_fc1 = bias_variable([64])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*4*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


	'''
	W_fc2 = weight_variable([16, 64])
	b_fc2 = bias_variable([64])
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
	'''
	W_fc2 = weight_variable([64,10])
	b_fc2 = bias_variable([10])

	y_conv = tf.matmul(h_fc1,W_fc2) + b_fc2
	epoch=0
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y,1))
	correct_test_prediction = tf.equal(tf.argmax(y_conv,1),trainData[1][0:batchSize])

	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	converged=False
	bestAccuracy=0
	start=time.clock()
	timeout = start+10
	epochsUntilQuit = 10
	with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			while(converged==False):
				randomize = np.arange(len(train_X))
				np.random.shuffle(randomize)
				train_x=train_X[randomize] #arrays have been 'synchronously' shuffled
				train_y=train_Y[randomize]
				validate_x = train_X[40000:]
				validate_y = train_Y[40000:]
				train_x = train_X[:40000]
				train_y = train_Y[:40000]
				train_accuracy_sum = 0
				for i in range(int(40000/batchSize)):
					train_step.run(feed_dict={x: train_x[i*batchSize:(i+1)*batchSize],y:train_y[i*batchSize:(i+1)*batchSize]})
					if((i+1)%10==0):
						train_accuracy = accuracy.eval(feed_dict={x: train_x[i*batchSize:(i+1)*batchSize],y:train_y[i*batchSize:(i+1)*batchSize]})
						train_accuracy_sum+=train_accuracy
						print("epoch %d step %d, training accuracy %g"%(epoch,i+1,train_accuracy))
					
						

				trainAccuracy.append(train_accuracy_sum/8)
				validateAcc=0
				for j in range(int(10000/batchSize)):
					validateAcc += accuracy.eval(feed_dict={x: validate_x[j*batchSize:(j+1)*batchSize],y:validate_y[j*batchSize:(j+1)*batchSize]})
				validateAcc=validateAcc/20
				if(validateAcc > bestAccuracy):
					bestAccuracy=validateAcc
					epochsUntilQuit = 10

				validationAccuracy.append(validateAcc)
				print("epoch %d, validate accuracy %g, epochs until quit %d"%(epoch,validateAcc,epochsUntilQuit))
				epoch+=1
				epochsUntilQuit-=1
				if(epochsUntilQuit==0):
					converged = True
					print("converged epoch %d, training accuracy %g"%(epoch,train_accuracy))
			testAcc=0
			for i in range(int(10000/batchSize)):
				testAcc += accuracy.eval(feed_dict={x: test_X[i*batchSize:(i+1)*batchSize],y:test_Y[i*batchSize:(i+1)*batchSize]})
			testAcc = testAcc / 20
			print("test accuracy = ",testAcc)
			print("best validate accuracy = ",bestAccuracy)
	plt.plot(trainAccuracy,label="Accuracy on Training Set")
	plt.plot(validationAccuracy,label="Accuracy on Validation Set")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.title("Accuracy on Training/Validation Set")
	plt.legend(loc=8)
	plt.savefig("c32c64c64c64mp2mp2fc64.png")
	plt.show()




