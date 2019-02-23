from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import operator

def calc_prior(train_data, l):
	prior = {}
	for key, value in train_data.items():
		prior[key] = len(value) / l
	return prior

''' Separating data by classes '''
def separate_classes(train_data):
	classes = {}
	l = len(train_data)
	for i in range(l):
		if train_data[i][-1] not in classes.keys():
			classes[train_data[i][-1]] = [train_data[i][:-1]]
		else:
			classes[train_data[i][-1]].append(train_data[i][:-1])
	return classes, l

''' Calculate mean of each attribute in data '''
def mean(data):
	return np.mean(data,axis=0)

''' Calculate std of each attribute in data '''
def std(data):
	return np.std(data,axis=0)

''' Formula for mutivariate normal distribution (pdf)'''
def calc_norm(x, mean, std):
	pi = 3.1428
	var = np.power(std,2)
	mean = np.array(mean, dtype="float64")
	x = np.array(x, dtype="float64")
	exponent = np.exp(-(np.power(x - mean, 2) / (2 * var)))
	prod = np.prod((1 / (np.sqrt(2 * np.pi * var))) * exponent)
	return prod

''' Calculating mean and covariance for training_data '''
def nbc(train_data):
	mean_cov = {}
	for key, value in train_data.items():
		mean_cov[key] = (mean(train_data[key]),std(train_data[key]))
	return mean_cov

''' Calculating probaility of each data point for every class '''
def calc_probabilities(mean_cov, x, prior):
	probabilities = {}
	for key, value in mean_cov.items():
		mean = value[0]
		std = value[1]
		probabilities[key] = prior[key]*calc_norm(x, mean, std)
	return probabilities

def predict(mean_cov, x, prior):
	probs = calc_probabilities(mean_cov, x, prior)
	return max(probs.items(), key=operator.itemgetter(1))[0]
	# return bestLabel

def separate_test_set(data):
	return np.delete(data, -1, axis=1), data[:,-1]

''' Predicting class labels for test data '''
def test(mean_cov, ts_data, prior):
	return [predict(mean_cov, ts_data[i], prior) for i in range(len(ts_data))]

''' Calculating accuracy of the model '''
def accuracy(predictions, ts_labels):
	diff = np.array(predictions) - np.array(ts_labels)
	return 1 - (np.count_nonzero(diff)/len(diff))

if __name__ == "__main__":

	data = np.loadtxt("data_banknote_authentication.txt", delimiter=",")

	acc = []

	kf = KFold(n_splits=3, shuffle=True)
	data = np.array(data)
	for train_index, test_index in kf.split(data):
		train_data, test_data = data[train_index], data[test_index]
		train_data = np.array(train_data)
		test_data = np.array(test_data)
		train_data, l = separate_classes(train_data)
		prior = calc_prior(train_data, l)

		# Naive Bayes Classifier
		mean_cov = nbc(train_data)
		ts_data, ts_labels = separate_test_set(test_data)
		predictions = test(mean_cov, ts_data, prior)
		acc.append(accuracy(predictions, ts_labels))

	print(acc)
	print(sum(acc)/kf.get_n_splits(data))