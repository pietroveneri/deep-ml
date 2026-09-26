import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	scores = np.array(scores)
	log_soft = []
	for score in scores:
		log_soft.append(score - np.log(np.sum(np.exp(scores))))
	return np.array(log_soft)