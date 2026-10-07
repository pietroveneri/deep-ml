import numpy as np

def softmax(values):
	max_score = max(values)
	exp_scores = [np.exp(z - max_score) for z in values]
	total = sum(exp_scores)
	return [e / total for e in exp_scores]

def pattern_weaver(n, crystal_values, dimension):

	crystal_values = np.array(crystal_values)
	attention_scores = [np.dot(crystal, crystal_values.T)  / np.sqrt(dimension) for crystal in crystal_values]
	attention_prob = [softmax(attention_score) for attention_score in attention_scores]
	attention_prob = np.array(attention_prob)
	x = attention_prob @ crystal_values
	
	return np.round(x,3)