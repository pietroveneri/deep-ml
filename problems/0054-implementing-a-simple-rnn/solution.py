import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	input_sequence = np.array(input_sequence, dtype = "float")
	initial_hidden_state = np.array(initial_hidden_state, dtype = "float")
	Wx = np.array(Wx, dtype = "float")
	Wh = np.array(Wh, dtype = 'float')


	h = initial_hidden_state
	for i, step in enumerate(input_sequence):
		h = np.tanh(np.dot(Wh, h) + np.dot(Wx, input_sequence[i]) + b)
	final_hidden_state = np.round(h, 4)
	return final_hidden_state