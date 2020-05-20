import numpy as np

class naive_bayes(object):
	def __init__(self):
		pass

	def fit(self, X, y):
		self.target = np.unique(y)
		_, count = np.unique(y, return_counts=True)
		self.prior = count / len(y)
		self.means = [np.mean(X[y==cls], axis=0) for cls in self.target]
		self.var = [np.var(X[y==cls], axis=0) for cls in self.target]
		return self

	def predict(self, X):
		return [self.predict_(x) for x in X]

	def predict_(self, X):
		likelihood = list()
		for idx, cls in enumerate(self.target):
			likelihood.append(np.prod(self.pdf(idx, X)))

		product = likelihood * self.prior
		evidense = np.inner(likelihood, self.prior)
		return np.argmax(product / evidense)

	def pdf(self, cls_idx, X):
		mean = self.means[cls_idx]
		var = self.var[cls_idx]
		numerator = np.exp(-(X-mean)**2 / (2*var))
		denominator = np.sqrt(2*np.pi*var)
		return numerator / denominator

def main():
	from sklearn.datasets import load_iris
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score

	iris = load_iris()

	X = iris.data[:, [2,3]]
	y = iris.target
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1, stratify=y)

	bayes = naive_bayes()
	fit = bayes.fit(X=X_train, y=y_train)
	y_pred = bayes.predict(X=X_test)

	print(accuracy_score(y_true=y_test, y_pred=y_pred))
	

if __name__ == '__main__':
	main()