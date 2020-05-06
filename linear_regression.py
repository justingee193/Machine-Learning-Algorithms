import numpy as np
import pandas as pd
import random as rand

class linear_regression(object):
	def __init__(self, coef=None, intercept=None):
		self.coef = coef
		self.intercept = intercept

	def fit(self, X, y):
		A = np.column_stack([[1]*X.shape[0], X])
		self.AtA = A.transpose().dot(A)
		self.AtB = A.transpose().dot(y)
		self.result = np.linalg.solve(self.AtA, self.AtB)

		self.intercept = self.result[0]
		self.coef = self.result[1:]
		return self

	def predict(self, X):
		if X.ndim == 1:
			return np.concatenate([[[1]*X.shape[0]], [X]]).transpose().dot(self.result)
		else:
			return np.column_stack([[1]*X.shape[0], X]).dot(self.result)

class stochastic_gradient_descent(object):
	def __init__(self, epochs=10000, learning_rate=0.001, coef_=None, intercept_=None, penalty=0.01, 
				 verbose=False):
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.coef = coef_
		self.intercept = intercept_
		self.penalty = penalty
		self.verbose = verbose

	def fit(self, X, y):
		self.coef = np.zeros(X.shape[1]+1)

		def update(X, y, coef, penalty, learning_rate):

			X = np.concatenate([[1], X])

			y_hat = coef.dot(X)
			loss = (y_hat-y)**2
			partial = 2*(y_hat - y)
			coef = coef - learning_rate*(partial*X + 2*penalty*coef)
			return coef, loss

		for epoch in range(self.epochs):
			for batch in range(round(X.shape[0]/3)):
				sample = int(rand.uniform(0, X.shape[0]))
				self.coef, loss = update(X=X.iloc[sample], y=y.iloc[sample], coef=self.coef, 
										 learning_rate=self.learning_rate, penalty=self.penalty)

				if self.verbose == True:
					print("Epoch: {}/{} Batch: {}/{} Loss: {}".format(epoch+1, self.epochs, 
						  											  batch+1, self.batches, loss))
		
		self.intercept = self.coef[0]
		self.coef = self.coef[1:]
		return self

	def predict(self, X):
		if X.ndim == 1:
			X = np.concatenate([[[1]*X.shape[0]], X])
			return X.dot(np.append(self.intercept, self.coef))
		else:
			X = np.column_stack([[1]*X.shape[0], X])
			return X.dot(np.append(self.intercept, self.coef))
