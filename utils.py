import tensorflow as tf 
import numpy as np 
import csv
import cv2

csv_train = './DL# Beginner/meta-data/train.csv'
csv_test = './DL# Beginner/meta-data/test.csv'

def create_dict():
	animal_dict={
		'antalope': 1,
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

# read in training-images; path=string
def _read_csv(path, testset=False):
	filenames = []
	labels = []

	with open(path, newline='') as csvfile:
	    spamreader = csv.reader(csvfile)
	    for row in spamreader:
	        filenames.append(row[0])
	        if not testset:
	        	labels.append(row[1])

	return labels, filenames

def _convert_labels_to_numbers(path):
	labels,_ = _read_csv(path)
	labels_num = []
	animal_dict = create_dict()
	for label in labels:
		labels_num.append(animal_dict.get(label))

	return labels_num

# testing it
# filenames,_ = read_csv(csv_train)
# print(filenames)

def _resize_image_and_to_tensor(counter, image_height, image_width):

	im = cv2.imread('./DL# Beginner/train/Img-' + str(counter+1) + '.jpg')
	resized_image = cv2.resize(im, (image_height, image_width))

	return resized_image


def get_images_minibatch(number_of_images, offset, image_height=128, image_width=128):

	minibatch_tensor = np.zeros([number_of_images, image_height, image_width, 3])
	for i in range(number_of_images):
		minibatch_tensor[i,:,:,:] = _resize_image_and_to_tensor(i+offset, image_height, image_width)

	return minibatch_tensor

def get_labels_minibatch(number_of_images, offset, number_of_classes=30):
	label_tensor = np.zeros([number_of_images, number_of_classes])
	labels = _convert_labels_to_numbers(csv_train)
	for i in range(number_of_images):
		label_tensor[i,:] = labels[i+offset]

	return label_tensor