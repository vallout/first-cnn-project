import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

from utils import get_images_minibatch, get_labels_minibatch

def create_placeholders(n_H0, n_W0, n_C0, n_y):

	"""
	Creates the placeholders for the tensorflow session.

	Arguments:
	n_H0 -- scalar, height of an input image
	n_W0 -- scalar, width of an input image
	n_C0 -- scalar, number of channels of the input
	n_y -- scalar, number of classes

	Returns:
	X -- placeholder for the data input of shape [None, n_H0, n_W0, n_C0]
	Y -- placeholder for the input labels of shape [None, n_y] and dtype 'float'
	"""

	X = tf.placeholder(
	 	dtype=tf.float32,
	 	shape=(None, n_H0, n_W0, n_C0)
		)
	Y = tf.placeholder(
		dtype=tf.float32,
		shape=(None, n_y)
		)

	return X, Y

def initialize_parameters():
	"""
	Initializes weight parameters to build a neural network with tensorflow. 

	Returns:
	parameters -- a dictionary of tensors containing W1, W2
	"""

	W1 = tf.get_variable("W1", [128, 128, 3, 16], initializer=tf.contrib.layers.xavier_initializer())
	W2 = tf.get_variable("W2", [64, 64, 16, 32], initializer=tf.contrib.layers.xavier_initializer())
	W3 = tf.get_variable("W3", [32, 32, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
	W4 = tf.get_variable("W4", [16, 16, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
	W5 = tf.get_variable("W5", [8, 8, 128, 256], initializer=tf.contrib.layers.xavier_initializer())

	parameters = {"W1": W1,
				  "W2": W2,
				  "W3": W3,
				  "W4": W4,
				  "W5": W5}

	return parameters

def forward_propagation(X, parameters):
	"""
	Implements the forward propagation for the model:
	CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

	Arguments:
	X -- input dataset placeholder, of shape (input size, number of examples)
	parameters -- python dictionary containing your parameters "W1", "W2"
				  the shapes are given in initialize_parameters

	Returns:
	Z3 -- the output of the last LINEAR unit
	"""
	W1 = parameters["W1"]
	W2 = parameters["W2"]
	W3 = parameters["W3"]
	W4 = parameters["W4"]
	W5 = parameters["W5"]

	Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding="SAME")
	A1 = tf.nn.relu(Z1)
	P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

	Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding="SAME")
	A2 = tf.nn.relu(Z2)
	P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

	Z3 = tf.nn.conv2d(P2, W3, strides=[1,1,1,1], padding="SAME")
	A3 = tf.nn.relu(Z3)
	P3 = tf.nn.max_pool(A3, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

	Z4 = tf.nn.conv2d(P3, W4, strides=[1,1,1,1], padding="SAME")
	A4 = tf.nn.relu(Z4)
	P4 = tf.nn.max_pool(A4, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

	P4 = tf.contrib.layers.flatten(P4)
	Z5 = tf.contrib.layers.fully_connected(P4, 30, activation_fn=None)

	return Z5

def compute_cost(Z5, Y):
	"""
	Compute the cost

	Arguments:
	Z5 -- output of forward propagation (output of the last LINEAR unit), of shape (30, number of examples)
	Y -- "true" labels vector placeholder, same shape as Z3

	Returns:
	cost - Tensor of cost function
	"""

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z5, labels=Y))

	return cost 

def _mini_batches(X, Y, minibatch_size):
	"""
	Create minibaches of training data sets

	Arguments:
	X -- training data of shape (m, n_H, n_W, n_C)
	Y -- labels of training data of shape (m, n_y)
	minibatch_size -- integer of size of minibatch

	Returns:
	minibatches - List of Tensors of minibatches
	"""
	minibatches = []

	(m, n_H, n_W, n_C) = X.shape
	(_, n_y) = Y.shape
	number_of_minibatches = int(m/minibatch_size)

	for i in range(number_of_minibatches):
		minibatch_X = np.zeros(shape=(minibatch_size, n_H, n_W, n_C))
		minibatch_Y = np.zeros(shape=(minibatch_size, n_y))
		for j in range(minibatch_size):
			minibatch_X[j,:,:,:] = X[j+i*minibatch_size,:,:,:]
			minibatch_Y[j,:] = Y[j+i*minibatch_size,:]
		minibatches.append((minibatch_X, minibatch_Y))

	return minibatches



def model(X_train, Y_train, learning_rate = 0.005,
		  num_epochs = 100, minibatch_size = 50, print_cost = True):
	"""
	Implements a three-layer ConvNet in Tensorflow:
	CONV2d -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

	Arguments: 
	X_train -- training set, of shape (None, 64, 64, 3)
	y_train -- training set, of shape (None, n_y = 6)
	X_test -- test set, of shape (None, 64, 64, 3)
	Y_test -- test set, of shape (None, n_y = 6)
	learning_rate -- learning rate of the optimization
	num_epochs -- number of epochs of the optimiuation loop
	minibatch_size -- size of a minibatch
	print_cost -- True to print the cost every 100 epochs

	Returns:
	train_accuracy -- real number, accuracy on the train set (X_train)
	test_accuracy -- real number, testing accuracy on the test set (X_test)
	parameters -- parameters learnt by the model. They can then be used to predict.
	"""

	(m, n_H0, n_W0, n_C0) = X_train.shape
	n_y = Y_train.shape[1]
	costs = []

	X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

	parameters = initialize_parameters()

	Z5 = forward_propagation(X, parameters)

	cost = compute_cost(Z5, Y)

	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

	init = tf.global_variables_initializer()

	with tf.Session() as sess:

		sess.run(init)

		for epoch in range(num_epochs):

			minibatch_cost = 0.
			num_minibatches = int(m / minibatch_size)
			minibatches = _mini_batches(X_train, Y_train, minibatch_size)

			for minibatch in minibatches:

				minibatch_X, minibatch_Y = minibatch

				_, temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})

				minibatch_cost += temp_cost / num_minibatches

			if print_cost == True and epoch % 5 == 0:
				print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
			if print_cost == True and epoch % 1 == 0:
				costs.append(minibatch_cost)

	plt.plot(np.squeeze(cost))
	plt.ylabel('cost')
	plt.xlabel('iterations (per tens)')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()

	predict_op = tf.argmax(Z5, 1)
	correct_prediction = tf.equal(predict_op, tf.argmax(Y,1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print(accuracy)
	train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
	print("Train Accuracy:", train_accuracy)

	return train_accuracy, parameters