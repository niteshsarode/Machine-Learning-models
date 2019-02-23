from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import operator

def separate_labels(data):
	t_labels = data[:,-1]
	t_data = np.delete(data, -1, axis=1)
	return t_data, t_labels

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def update_weights_bias(train_data, train_labels, weights, bias, lr):
	
	N = len(train_data)
	predictions = predict(train_data, weights, bias)
	dw = np.dot(train_data.T,  train_labels - predictions)
	db = np.sum(train_labels - predictions)
	weights = weights + (dw*lr) / N
	bias = bias + (db*lr) / N

	return weights, bias

def train(train_data, train_labels):
	cost_history = []
	iterations = 100
	lr = 1.0

	weights = np.random.randn(train_data.shape[1])
	bias = np.random.randn()

	for i in range(iterations):
		weights, bias = update_weights_bias(train_data, train_labels, weights, bias, lr)

	return weights, bias

def predict(x, weights, bias):
	z = []
	for i in range(x.shape[0]):
		y = x[i] * weights
		z.append(np.sum(y) + bias)
	z = np.array(z)
	return sigmoid(z)

def cost_function(features, labels, weights):
	pass

def test(test_data, weights, bias):
	predicted_probs = predict(test_data, weights, bias)
	predicted_probs = predicted_probs+0.5
	predicted_labels = np.floor(predicted_probs)
	return predicted_labels

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

		train_data, train_labels = separate_labels(train_data)
		test_data, test_labels = separate_labels(test_data)

		# Logistic Regression
		weights, bias = train(train_data, train_labels)
		predicted_labels = test(test_data, weights, bias)
		acc.append(accuracy(predicted_labels, test_labels))

	print(acc)
	print(sum(acc)/kf.get_n_splits(data))
