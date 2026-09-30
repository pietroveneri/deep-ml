import numpy as np

class LSTM:
	def __init__(self, input_size, hidden_size):
		self.input_size = input_size
		self.hidden_size = hidden_size

		# Initialize weights and biases
		self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

		self.bf = np.zeros((hidden_size, 1))
		self.bi = np.zeros((hidden_size, 1))
		self.bc = np.zeros((hidden_size, 1))
		self.bo = np.zeros((hidden_size, 1))

	def forward(self, x, initial_hidden_state, initial_cell_state):
		"""
		Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
		"""
		x = np.array(x, dtype =float)

		h = np.array(initial_hidden_state, dtype = float).reshape(self.hidden_size, 1)
		C = np.array(initial_cell_state, dtype = float).reshape(self.hidden_size, 1)

		hidden_states = []

		def sigmoid(z):
			return 1/(1 + np.exp(-z))
		
		for i in range(len(x)):
			x_i = x[i].reshape(self.input_size, 1)

			combined = np.concatenate((h, x_i), axis=0)

			f = sigmoid(self.Wf @ combined + self.bf)
			i = sigmoid(self.Wi @ combined + self.bi)
			C_ = np.tanh(self.Wc @ combined + self.bc)
			o = sigmoid(self.Wo @ combined + self.bo)

			C = f * C + i * C_
			h = o * np.tanh(C)

			hidden_states.append(h.copy())

		return np.array(hidden_states), h, C
		

