# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 14:24:14 2017

@author: Yanchen
@version: python3.5
"""

import os
import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# part1: data preprocess
# constant parameters
# image size
image_height = 640;
image_width = 480;
pixel_depth = 255.0

# function: load photo from specific file path
def loadPhoto(filepath, source):
	num_images = 0;
	filelist = os.listdir(os.path.join(filepath, source));
	dataset = np.ndarray(shape = (len(filelist), image_width, image_height), dtype=np.float32);
	label = np.ndarray(shape = (len(filelist)), dtype=np.int);

	for image in filelist:
		image_path = os.path.join(os.path.join(filepath, source), image);
		try:
			img_data = (ndimage.imread(image_path).astype(float) - pixel_depth/2)/pixel_depth;
			if (img_data.shape != (image_width, image_height)):
				raise Exception('Unexpected image shape: %s' % str(img_data.shape));
			dataset[num_images, :, :] = img_data;
			label[num_images] = source[len(source)-1];
			num_images = num_images + 1;
		except Exception as e:
			print('Could not read:', image, ':', e, '- it\'s ok, skipping.');
	
	dataset = dataset[0:num_images, :, :];
	return dataset, label;

def randomize(dataset, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:,:]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels

filepath = os.path.join(os.getcwd(), 'photo');
filepath = os.path.join(os.getcwd(), 'dataset.pickle');
# initialize variables
dataset = None;
labels = None;

if (os.path.exists(filepath)):
	try:
		with open(filepath, 'rb') as f:
			save = pickle.load(f);
			dataset = save['dataset'];
			labels = save['labels'];
			del save  # hint to help gc free up memory
			print('Dataset', dataset.shape, labels.shape);
	except Exception as e:
		print('Could not read file: ' + e);
else:
	# file path
	filepath = os.path.join(os.getcwd(), 'photo');
	# load data from filepath
	dataset, label = loadPhoto(filepath, 'label_2');
	print('dataset label files size: %d' % dataset_2.shape[0]);
	# randomize dataset
	dataset, labels = randomize(dataset, labels);
	# save dataset
	filepath = os.path.join(filepath, 'dataset.pickle');
	try:
		f = open(filepath, 'wb')
		save = {
		'dataset': dataset, 
		'labels': labels,
		};
		pickle.dump(save, f, pickle.HIGHEST_PROTOCOL);
		print('Successful save data to', filepath);
	except Exception as e:
		print('Unable to save data to', filepath, ':', e);

def sklearn_model(train, y_train):
	model = LogisticRegression(solver='sag');
	model.fit(x_train, y_train);
	print('Accuray of linear model: %d' % model.score(x_train, y_train));
	return model;

status = input('Continue for sklearning logistic regression: y or n\n');
# sklearn logistic regression
if (status == 'y'):
	# convert dataset and labels
	dataset = dataset.reshape(dataset.shape[0], image_width*image_height);
	# cross validation
	x_train, x_test, y_train, y_test = train_test_split(dataset, labels);
	model = sklearn_model(x_train, y_train);
	predict = model.predict(x_test);
	accuracy = sum((predict == y_test) == True)/len(predict)*100.0;
	print('Test accuracy: %f%%' % accuracy);
elif (status == 'n'):
	pass;
else:
	print('Invalid input\n');
