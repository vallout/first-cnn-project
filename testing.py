from convNet import create_placeholders, initialize_parameters, forward_propagation, compute_cost, model
from utils import get_images_minibatch, get_labels_minibatch
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
		print("W3 = " + str(parameters["W3"].eval()[1,1,1]))
		print("W4 = " + str(parameters["W4"].eval()[1,1,1]))
		print("W5 = " + str(parameters["W5"].eval()[1,1,1]))

#test_initialize_parameters()

def test_forward_propagation():
	with tf.Session() as sess_test:
		X, Y = create_placeholders(128, 128, 3, 30)
		parameters = initialize_parameters()
		Z5 = forward_propagation(X, parameters)
		init = tf.global_variables_initializer()
		sess_test.run(init)
		a = sess_test.run(Z5, {X: np.random.randn(5,128,128,3), Y: np.random.rand(5,30)})
		print("Z5 = " + str(a))

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

x_train=get_images_minibatch(13000, 0)
y_train=get_labels_minibatch(13000, 0)

model(x_train, y_train)