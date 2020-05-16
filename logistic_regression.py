import numpy as np
import random as rand

class logistic_regression():
	def __init__(self, n_iter=1000, eta=0.001, batch_size=50, random_state=1):
		self.n_iter = n_iter
		self.eta = eta
		self.batch_size = batch_size
		self.random_state = random_state

	def sigmoid(self, X):
		return np.array(1/(1+np.exp(-np.dot(X, self.w))))

	def update(self, X, y):
		y_hat = self.sigmoid(X)
		gradient = np.dot(X.transpose(), y_hat-y) # gradient of sigmoid function
		avg = gradient/len(X)
		self.w -= self.eta*gradient
		return self.w

	def bias(self, X):
		return np.column_stack([np.ones((X.shape[0],1)), X])

	def fit(self, X, y):
		X = self.bias(X)
		y = np.array(y)

		seed = rand.seed(self.random_state)
		self.w = [rand.normalvariate(mu=0.0, sigma=0.01) for i in range(X.shape[1])]

		for i in range(self.n_iter):
			sample = rand.sample(range(len(X)), self.batch_size)	
			self.w = self.update(X=X[sample,:], y=y[sample])
		return self

	def predict(self, X):
		X = self.bias(X)
		return np.where(self.sigmoid(X) >= 0.5, 1, 0)

def main():
	import numpy as np
	import pandas as pd
	from sklearn.datasets import load_breast_cancer
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score, roc_auc_score
	cancer = load_breast_cancer()
	df = pd.DataFrame(np.c_[cancer['data'], cancer['target']], 
		columns = np.append(cancer['feature_names'], ['target']))

	X = df[cancer.feature_names]
	y = df['target']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=444)
	
	lr = logistic_regression()
	fit = lr.fit(X_train, y_train)
	y_pred = lr.predict(X=X_test)

	print(accuracy_score(y_true=y_test, y_pred=y_pred))
#	print(roc_auc_score(y_true=y_test, y_pred=y_pred))

if __name__ == '__main__':
	main()