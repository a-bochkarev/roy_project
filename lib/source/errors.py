import numpy as np

def RMSE(X, Y):
	"""
	Root Mean Squared Error of 2 arrays
	Input:
	X, Y : array_like with the same shape
	Output:
	Value of RMSE
	"""
	return np.sqrt(1./np.prod(X.shape) * np.sum((X-Y)**2))

def MAE(X, Y):
	"""
	Mean Average Error of 2 arrays
	Input:
	X, Y : array_like with the same shape
	Output:
	Value of MAE
	"""
	return 1./np.prod(X.shape) * np.sum(np.abs(X-Y))