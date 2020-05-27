import numpy as np
import operator

class knn(object):
	def __init__(self, n_neighbors=1):
		self.n_neighbors = n_neighbors
	def euclidean(self, X_train, X_test):
		distances = []
		for x in X_train:
			inner = X_test - x
			sums = np.sum((inner)**2)
			distances.append(np.sqrt(sums))
		return distances
	def k_neighbors(self, distances):
		sorted_dist = sorted(range(len(distances)), key=lambda k: distances[k])
		return sorted_dist[:self.n_neighbors]
	def fit(self, X, y):
		self.X = X
		self.y = y
		return self
	def predict(self, X):
		results = list()
		for x in X:
			distance = self.euclidean(X_train=self.X, X_test=x)
			neighbors = self.k_neighbors(distances=distance)
			target = self.y[neighbors]
			cls, counts = np.unique(target, return_counts=True)
			index, _ = max(enumerate(counts), key=operator.itemgetter(1))
			results.append(cls[index])
		return np.array(results)

def main():
	import operator
	from sklearn.datasets import load_iris
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score

	iris = load_iris()
	X = iris.data[0:100,[2,3]]
	y = iris.target[0:100]
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
	
	k_nn = knn()
	fit = k_nn.fit(X=X_train, y=y_train)

	y_pred = k_nn.predict(X=X_test)
	print(y_pred)
	print(y_test)
	print(accuracy_score(y_true=y_test, y_pred=y_pred))

	
	import matplotlib.pyplot as plt
	
	plt.scatter(X_train[:,0], X_train[:,1], c='blue')
	plt.scatter(X_test[:,0], X_test[:,1], c='red')
	plt.show()
	

if __name__ == '__main__':
	main()