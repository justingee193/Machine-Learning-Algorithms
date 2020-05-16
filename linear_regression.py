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

class gradient_descent(object):
	def __init__(self, epochs=1000, learning_rate=0.001, batch_size=1, 
					penalty=0.01, verbose=False, random_state=1):
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.penalty = penalty
		self.verbose = verbose
		self.random_state = random_state

	def SGD(self, X, y, coef):
		y_hat = self.solve(X=X)
		gradient = 2*(y_hat-y)
		coef -= self.learning_rate*(gradient*X + 2*self.penalty*self.coef)[0]
		return coef

	def BGD(self, X, y, coef):
		size = len(y)
		y_hat = self.solve(X=X)
		coef = coef - self.learning_rate*(1.0/size)*np.dot(X.T, y_hat-y)
		return coef

	def solve(self, X):
		return np.dot(X, self.coef)

	def bias(self, X):
		return np.column_stack([[1]*X.shape[0], X])

	def loss(self, X, y, coef):
		size = len(y)
		y_hat = self.solve(X=X)
		loss = (y_hat - y)**2
		return loss.sum() / size

	def fit(self, X, y):
		X = self.bias(X=X)
		y = np.array(y)
		self.coef = np.zeros(X.shape[1])
		
		seed = rand.seed(self.random_state)
		for epoch in range(self.epochs):
			sample = rand.sample(range(len(X)), self.batch_size)

			if self.batch_size == 1:
				self.coef = self.SGD(X=X[sample,:], y=y[sample], coef=self.coef)
			else:
				self.coef = self.BGD(X=X[sample,:], y=y[sample], coef=self.coef)

			loss = self.loss(X=X, y=y, coef=self.coef)

			if self.verbose == True:
				print("Epoch: {}/{} Loss: {}".format(epoch+1, self.epochs, round(loss, 5)))

		self.intercept = self.coef[0]
		self.coef = self.coef[1:]
		return self

	def predict(self, X):
		X = self.bias(X)
		return np.dot(X, np.append(self.intercept, self.coef))

def main():
	import numpy as np
	import pandas
	from sklearn.datasets import load_iris
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import mean_squared_error

	iris = load_iris()
	df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

	X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)']]
	y = df['petal width (cm)']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=444)
	
	lr = linear_regression()
	fit = lr.fit(X=X_train, y=y_train)
	y_pred = lr.predict(X=X_test)

	print(mean_squared_error(y_true=y_test, y_pred=y_pred))
	
	stoch = gradient_descent()
	fit = stoch.fit(X=X_train, y=y_train)
	y_pred = stoch.predict(X=X_test)
	
	print(mean_squared_error(y_true=y_test, y_pred=y_pred))
	

if __name__ == '__main__':
	main()