import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	weights = np.array(initial_weights)
	bias = np.array(initial_bias, dtype ="float")
	features = np.array(features)
	labels = np.array(labels)
	mse_values = []

	for _ in range(epochs):
		z = np.dot(features, weights) + bias
		predictions = f(z)

		mse = np.mean((predictions - labels)**2)
		mse_values.append(round(mse,4))

		# Gradient calculations 
		errors = predictions - labels
		weight_gradients = (2/len(labels)) * np.dot(features.T, errors * predictions * (1-predictions))
		bias_gradients = (2/len(labels)) * np.sum(errors*predictions*(1-predictions))

		weights -= learning_rate * weight_gradients
		bias -= learning_rate * bias_gradients

		updated_weights = np.round(weights, 4)
		updated_bias = np.round(bias, 4)

	return updated_weights, updated_bias, mse_values

def f(z):
	return 1/(1 + np.exp(-z))
def dfdz(z):
	return f(z) * (1 - f(z))
