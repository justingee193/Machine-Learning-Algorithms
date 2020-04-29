import numpy as np
import pandas as pd

class linear_regression_la(object):
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
		return self.result

	def predict(self, X):
		if X.ndim == 1:
			return np.insert(X, 0, 1).dot(self.result)
		else:
			return np.column_stack([[1]*X.shape[0], X]).dot(self.result)

def main():
	from sklearn.datasets import load_iris
	import loss_functions

	iris = load_iris()
	df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

	linear = linear_regression_la()
	fit = linear.fit(X=df[['sepal width (cm)', 'petal length (cm)', 'petal width (cm)']], y=df['sepal length (cm)'])
	predict = linear.predict(X=df[['sepal width (cm)', 'petal length (cm)', 'petal width (cm)']])
	
	print(loss_functions.mse(y_true=df['sepal length (cm)'], y_hat=predict))
	print(loss_functions.mae(y_true=df['sepal length (cm)'], y_hat=predict))

if __name__ == '__main__':
	main()