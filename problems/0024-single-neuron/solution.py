import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	rows = len(features)
	cols = len(features[0])
	weighted = [[0 for _ in range(cols)] for _ in range(rows)] 
	probabilities = []
	weighted_row = []
	for row in range(rows):
		for col in range(cols):
			weighted[row][col] = features[row][col] * weights[col]
		weighted_row.append(sum(weighted[row]) + bias)
	sigma = []
	sigma = [1/(1 + math.exp(-z)) for z in weighted_row]
	sigma = [round(elem, 4) for elem in sigma]
	means = []
	means = [(x - y)**2 for x, y in zip(sigma, labels)]
	mse = round(1/len(means) * sum(means), 4)
	return sigma, mse