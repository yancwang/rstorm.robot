# -*- coding: utf-8 -*-
"""
Created on Thu Jun 01 11:55:49 2017

@author: yancw
"""

import os
import numpy as np
from six.moves import cPickle as pickle
from scipy import ndimage
from skimage import feature
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# part1: data preprocess
def loadPhoto(filepath, source):
	num_images = 0;
	filelist = os.listdir(os.path.join(filepath, source));
	dataset = np.ndarray(shape=(len(filelist), image_width, image_height), dtype=np.float32);
	label = np.ndarray(shape=(len(filelist)), dtype=np.int);

	for image in filelist:
		image_path = os.path.join(os.path.join(filepath, source), image);
		try:
			img_data = ndimage.imread(image_path).astype(float);
			img_data = feature.canny(img_data, sigma=2)
			img_data = img_data * 1;
			# img_data = (img_data - pixel_depth/2)/pixel_depth;
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

filepath = os.path.join(os.getcwd(), 'dataset.pickle');
# initialize variables
dataset = None;
labels = None;
# image size
image_height = 640;
image_width = 480;
pixel_depth = 255.0

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
	dataset_1, label_1 = loadPhoto(filepath, 'label_1'); 
	print('dataset label 1 files size: %d' % dataset_1.shape[0]);
	dataset_2, label_2 = loadPhoto(filepath, 'label_2');
	print('dataset label 2 files size: %d' % dataset_2.shape[0]);
	# merge train dataset
	dataset = np.concatenate((dataset_1, dataset_2), axis = 0);
	labels = np.concatenate((label_1, label_2), axis = 0);
	# randomize dataset
	dataset, labels = randomize(dataset, labels);
	# save dataset
	filepath = os.path.join(os.getcwd(), 'dataset.pickle');
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

def split_train_test(dataset, labels):
	x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=0);
	return x_train, x_test, y_train, y_test;

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
	x_train, x_test, y_train, y_test = split_train_test(dataset, labels);
	model = sklearn_model(x_train, y_train);
	predict = model.predict(x_test);
	accuracy = sum((predict == y_test) == True)/len(predict)*100.0;
	filepath = os.path.join(os.getcwd(), 'linearModel.pkl');
	joblib.dump(model, filepath);
	print('Test accuracy: %f%%' % accuracy);
elif (status == 'n'):
	pass;
else:
	print('Invalid input\n');