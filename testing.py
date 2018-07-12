from convNet import create_placeholders, initialize_parameters, forward_propagation, compute_cost, model
from utils import get_images_as_tensor, get_labels_as_tensor, shuffle_Y_and_X_synchronous
import tensorflow as tf
import numpy as np

def test_create_placeholders():

	X, Y = create_placeholders(128, 128, 3, 30)
	print ("X = " + str(X))
	print ("Y = " + str(Y))

#test_create_placeholders()

def test_initialize_parameters():

	with tf.Session() as sess_test:
		parameters = initialize_parameters()
		init = tf.global_variables_initializer()
		sess_test.run(init)
		print("W1 = " + str(parameters["W1"].eval()[1,1,1]))
		print("W2 = " + str(parameters["W2"].eval()[1,1,1]))

#test_initialize_parameters()

def test_forward_propagation():
	with tf.Session() as sess_test:
		X, Y = create_placeholders(128, 128, 3, 30)
		parameters = initialize_parameters()
		Z5 = forward_propagation(X, parameters)
		init = tf.global_variables_initializer()
		sess_test.run(init)
		a = sess_test.run(Z3, {X: np.random.randn(5,128,128,3), Y: np.random.rand(5,30)})
		print("Z3 = " + str(a))

#test_forward_propagation()

def test_compute_cost():

	with tf.Session() as sess_test:
		X, Y = create_placeholders(128, 128, 3, 30)
		parameters = initialize_parameters()
		Z3 = forward_propagation(X, parameters)
		cost = compute_cost(Z3, Y)
		init = tf.global_variables_initializer()
		sess_test.run(init)
		a = sess_test.run(cost, {X: np.random.randn(4,128,128,3), Y: np.random.randn(4,30)})
		print("cost = " + str(a))

#test_compute_cost()

def test_shuffling():
	X = np.arange(100)
	Y = np.arange(0,50,2)

	X = np.reshape(X, (5,5,2,2))
	Y = np.reshape(Y, (5,5))

	new_X, new_Y = shuffle_Y_and_X_synchronous(X,Y,5)

	print(new_X[2], new_Y[2])

# test_shuffling()

y_train=get_labels_as_tensor(13000)
x_train=get_images_as_tensor(13000)

model(x_train, y_train)
