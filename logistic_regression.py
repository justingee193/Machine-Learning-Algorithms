import numpy as np
import random as rand

class logistic_regression():
	def __init__(self, n_iter=1000, eta=0.01, batch_size=1, random_state=1):
		self.n_iter = n_iter
		self.eta = eta
		self.batch_size = batch_size
		self.random_state = random_state

	def sigmoid(self, X):
		return np.array(1/(1+np.exp(-np.dot(X, self.w))))

	def update(self, X, y):
		y_hat = self.sigmoid(X)
		gradient = np.dot(X.transpose(), y_hat-y)
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
	from sklearn.datasets import load_iris
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score, roc_auc_score

	iris = load_iris()

	X = iris.data[0:100,[0,2]]
	y = iris.target[0:100]
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)
	
	lr = logistic_regression()
	fit = lr.fit(X_train, y_train)
	y_pred = lr.predict(X=X_test)

	print(accuracy_score(y_true=y_test, y_pred=y_pred))
	
if __name__ == '__main__':
	main()