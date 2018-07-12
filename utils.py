import tensorflow as tf
import numpy as np
import csv
import cv2

csv_train = './DL# Beginner/meta-data/train.csv'
csv_test = './DL# Beginner/meta-data/test.csv'

def create_dict():
	animal_dict={
		'antelope': 1,
		'bat': 2,
		'beaver': 3,
		'bobcat': 4,
		'buffalo': 5,
		'chihuahua': 6,
		'chimpanzee': 7,
		'collie': 8,
		'dalmatian': 9,
		'german+shepherd': 10,
		'grizzly+bear': 11,
		'hippopotamus': 12,
		'horse': 13,
		'killer+whale': 14,
		'mole': 15,
		'moose': 16,
		'mouse': 17,
		'otter': 18,
		'ox': 19,
		'persian+cat': 20,
		'raccoon': 21,
		'rat': 22,
		'rhinoceros': 23,
		'seal': 24,
		'siamese+cat': 25,
		'spider+monkey': 26,
		'squirrel': 27,
		'walrus': 28,
		'weasel': 29,
		'wolf': 30,
	}
	return animal_dict

def _read_csv(path):
	"""
	Read labels from csvfile

	Arguments:
	path -- the path as string where the csvfile is located

	Returns:
	labels -- List of labels as string
	"""

	labels = []

	with open(path, newline='') as csvfile:
	    labelsreader = csv.reader(csvfile)
	    for row in labelsreader:
	        labels.append(row[1])

	return labels

def _convert_labels_to_numbers(path):
	"""
	convert the labels to numbers for classification

	Arguments:
	path -- the path as string where the csvfile is located

	Returns:
	labels_num -- List of labels as integers according to dictionary
	"""

	labels = _read_csv(path)
	labels_num = []
	animal_dict = create_dict()
	for label in labels:
		if not (label == 'Animal'):
			labels_num.append(animal_dict.get(label))

	return labels_num


def _resize_image_and_to_tensor(counter, image_height, image_width):
	"""
	resize one image and convert to "tensor"

	Arguments:
	counter -- the counter of the image in the file directory
	image_height -- scalar, the height of the output image
	image_width -- scalar, the width of the output image

	Returns:
	resized_image -- resized image as tensor
	"""

	im = cv2.imread('./DL# Beginner/train/Img-' + str(counter+1) + '.jpg')
	resized_image = cv2.resize(im, (image_height, image_width))

	return resized_image


def get_images_as_tensor(number_of_images, image_height=64, image_width=64):
	"""
	convert all the images to tensors of the same size for the convolutional net

	Arguments:
	number_of_images -- the number of the images
	image_height -- scalar, the height of the output image
	image_width -- scalar, the width of the output image

	Returns:
	images_tensor -- all images resized as tensor
	"""

	images_tensor = np.zeros([number_of_images, image_height, image_width, 3])
	for i in range(number_of_images):
		images_tensor[i,:,:,:] = _resize_image_and_to_tensor(i, image_height, image_width)
		if i % 1000 == 0:
			print("images minibatches " + str(i))

	return images_tensor

def get_labels_as_tensor(number_of_images, number_of_classes=30):
	"""
	convert the labels from csvfile to tensor

	Arguments:
	number_of_images -- the number of the images
	number_of_classes -- number of classes in the dataset

	Returns:
	labels_tensor -- all labels as tensor
	"""

	labels_tensor = np.zeros([number_of_images, number_of_classes])
	labels = _convert_labels_to_numbers(csv_train)
	for i in range(number_of_images):
		labels_tensor[i,labels[i]-1] = 1
		if i % 1000 == 0:
			print("labels minibatches " + str(i))

	return labels_tensor

def shuffle_Y_and_X_synchronous(X, Y, length):
	"""
	shuffles X and Y in same order

	Arguments:
	X -- training data of shape (m, n_H, n_W, n_C)
	Y -- labels of training data of shape (m, n_y)
	length -- scalar, equivalent to m

	Returns:
	X_shuffled -- tensor X shuffled
	Y_shuffled -- tensor Y shuffled
	"""

	rand_array = np.arange(length)
	np.random.shuffle(rand_array)
	X_shuffled = X[rand_array]
	Y_shuffled = Y[rand_array]

	return X_shuffled, Y_shuffled

def mini_batches(X, Y, minibatch_size):
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

	X_shuffled, Y_shuffled = shuffle_Y_and_X_synchronous(X,Y,m)

	for i in range(number_of_minibatches):
		minibatch_X = np.zeros(shape=(minibatch_size, n_H, n_W, n_C))
		minibatch_Y = np.zeros(shape=(minibatch_size, n_y))
		for j in range(minibatch_size):
			minibatch_X[j,:,:,:] = X_shuffled[j+i*minibatch_size,:,:,:]
			minibatch_Y[j,:] = Y_shuffled[j+i*minibatch_size,:]
		minibatches.append((minibatch_X, minibatch_Y))

	return minibatches
