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
		del save  # hint to help gc free up memory
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
	print('Test accuracy: %f%%' % accuracy);
elif (status == 'n'):
	pass;
else:
	print('Invalid input\n');

status = input('Continue for tensorflow logistic regression: y or n\n');
# tensorflow logistic regression 
num_labels = 2;

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_height*image_width)).astype(np.float32)
	# Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return dataset, labels

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

if (status == 'y'):
	train_dataset, train_labels = reformat(train_dataset, train_labels);
	test_dataset, test_labels = reformat(test_dataset, test_labels);
	print('Training set', train_dataset.shape, train_labels.shape);
	print('Testing set', test_dataset.shape, test_labels.shape);

	graph = tf.Graph();
	with graph.as_default():
		tf_train_dataset = tf.constant(train_dataset);
		tf_train_labels = tf.constant(train_labels);
		tf_test_dataset = tf.constant(test_dataset);

		weights = tf.Variable(tf.truncated_normal([image_height*image_width, num_labels]));
		bias = tf.Variable(tf.zeros([num_labels]));

		logits = tf.matmul(tf_train_dataset, weights) + bias
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits));

		optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss);

		train_prediction = tf.nn.softmax(logits);
		test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + bias);

	num_steps = 200;

	with tf.Session(graph=graph) as session:
		tf.global_variables_initializer().run();
		print('Initialized');

		for step in range(0, num_steps):
			_, l, predictions = session.run([optimizer, loss, train_prediction]);

		print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels));
elif (status == 'n'):
	pass;
else:
	print('Invalid input\n');

status = input('Continue for tensorflow neural network regression: y or n\n');
#tensorflow neural network regression

num_hidden_nodes = 16;

if (status == 'y'):
	train_dataset, train_labels = reformat(train_dataset, train_labels);
	test_dataset, test_labels = reformat(test_dataset, test_labels);
	print('Training set', train_dataset.shape, train_labels.shape);
	print('Testing set', test_dataset.shape, test_labels.shape);

	graph = tf.Graph()
	with graph.as_default():
		tf_train_dataset = tf.constant(train_dataset);
		tf_train_labels = tf.constant(train_labels);
		tf_test_dataset = tf.constant(test_dataset);
		#variables
		weights1 = tf.Variable(tf.truncated_normal([image_height*image_width, num_hidden_nodes]));
		biases1 = tf.Variable(tf.zeros([num_hidden_nodes]));
		weights2 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_labels]));
		biases2 = tf.Variable(tf.zeros([num_labels]));
		#training computation
		tf_train_layer = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1);
		logits = tf.matmul(tf_train_layer, weights2) + biases2;
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels));
		#optimizer
		optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss);
		#predictions for the training, validation, and test data
		train_prediction = tf.nn.softmax(logits);
		tf_test_layer = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1);
		test_prediction = tf.nn.softmax(tf.matmul(tf_test_layer, weights2) + biases2);

	num_steps = 1001

	with tf.Session(graph=graph) as session:
		tf.global_variables_initializer().run()
		print("Initialized")
		for step in range(num_steps):
			_, l, predictions = session.run([optimizer, loss, train_prediction])
			if (step % 200 == 0):
				print("Training loss at step %d: %f" % (step, l))
				print("Training accuracy: %.1f%%" % accuracy(predictions, train_labels))
				print("Testing accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

elif (status == 'n'):
	pass;
else:
	print('Invalid input\n');
