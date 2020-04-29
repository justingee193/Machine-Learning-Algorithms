import numpy as np
import math

""" Regression Loss Functions """
def mse(y_true, y_hat):
	return ((y_true - y_hat)**2).mean()

def mae(y_true, y_hat):
	return np.absolute(y_true - y_hat).mean()

def mbe(y_true, y_hat):
	return (y_true-y_hat).mean()

""" Classification Loss Functions """
def cross_entropy_loss(y_true, y_hat):
	return -np.sum(y_true*math.log(y_hat), (1-y_true)*math.log(1-y_hat))