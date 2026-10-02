
import numpy as np
class SimpleRNN:
	def __init__(self, input_size, hidden_size, output_size):
		self.hidden_size = hidden_size
		self.W_xh = np.random.randn(hidden_size, input_size)*0.01
		self.W_hh = np.random.randn(hidden_size, hidden_size)*0.01
		self.W_hy = np.random.randn(output_size, hidden_size)*0.01
		self.b_h = np.zeros((hidden_size, 1))
		self.b_y = np.zeros((output_size, 1))

	def forward(self, x):
		x = np.array(x, dtype=float)
		h = np.zeros((self.hidden_size, 1), dtype=float)
		hidden_states = []
		outputs = []
		for t in range(len(x)):
			x_t = x[t].reshape(-1, 1)
			h = np.tanh(self.W_hh @ h + self.W_xh @ x_t + self.b_h)
			y_t = self.W_hy @ h + self.b_y
			hidden_states.append(h.copy())
			outputs.append(y_t.copy())

		return np.array(outputs)

	def backward(self, x, y, learning_rate):
		x = np.array(x, dtype=float)
		y = np.array(y, dtype=float)

		h_states = [
			np.zeros((self.hidden_size, 1))
		]

		predictions = []

		for t in range(len(x)):
			x_t = x[t].reshape(-1, 1)

			h_t = np.tanh(
				self.W_xh @ x_t + 
				self.W_hh @ h_states[-1] + 
				self.b_h
			)

			h_states.append(h_t)

			prediction_t = self.W_hy @ h_t + self.b_y
			predictions.append(prediction_t)

		dW_hy = np.zeros_like(self.W_hy)
		dW_xh = np.zeros_like(self.W_xh)
		dW_hh = np.zeros_like(self.W_hh)

		db_h = np.zeros_like(self.b_h)
		db_y = np.zeros_like(self.b_y)

		# Gradient flowing backward from future hidden states
		dh_next = np.zeros((self.hidden_size, 1))

		# BPTT
		for t in reversed(range(len(x))):
			y_t = y[t].reshape(-1, 1)
			prediction_t = predictions[t]

			dy = (prediction_t - y_t) 

			dW_hy += dy @ h_states[t + 1].T
			db_y += dy

			dh = self.W_hy.T @ dy + dh_next

			dh_raw = (1 - h_states[t + 1] ** 2) * dh

			x_t = x[t].reshape(-1, 1)

			dW_xh += dh_raw @ x_t.T
			dW_hh += dh_raw @ h_states[t].T
			db_h += dh_raw	

			dh_next = self.W_hh.T @ dh_raw

		self.W_xh -= learning_rate * dW_xh
		self.W_hh -= learning_rate * dW_hh
		self.W_hy -= learning_rate * dW_hy

		self.b_h -= learning_rate * db_h
		self.b_y -= learning_rate * db_y	

		return np.array([p.flatten() for p in predictions])
			

