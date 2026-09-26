
import numpy as np
import copy
import math

# DO NOT CHANGE SEED
np.random.seed(42)

# DO NOT CHANGE LAYER CLASS
class Layer(object):

	def set_input_shape(self, shape):
		self.input_shape = shape

	def layer_name(self):
		return self.__class__.__name__

	def parameters(self):
		return 0

	def forward_pass(self, X, training):
		raise NotImplementedError()

	def backward_pass(self, accum_grad):
		raise NotImplementedError()

	def output_shape(self):
		raise NotImplementedError()

# Your task is to implement the Dense class based on the above structure
class Dense(Layer):
	def __init__(self, n_units, input_shape=None):
		self.layer_input = None
		self.input_shape = input_shape
		self.n_units = n_units
		self.trainable = True
		self.W = None
		self.w0 = None
		self.W_opt = None
		self.w0_opt = None

	def initialize(self, optimizer):
		limit = 1/np.sqrt(self.input_shape[0])
		self.W = np.random.uniform(-limit, limit, [self.input_shape[0], self.n_units])
		self.w0 = np.zeros([1, self.n_units])
		self.W_opt = copy.copy(optimizer)
		self.w0_opt = copy.copy(optimizer)

	def parameters(self):
		# For a fully connected layer, the number of parameters is given by the number of input units times the number of output units, plus one bias term for each output unit.
		return np.prod(self.W.shape) + np.prod(self.w0.shape)
	
	def forward_pass(self, X, training=True):
		self.layer_input = X
		return np.dot(X, self.W) + self.w0

	def backward_pass(self, accum_grad):
		if self.trainable == False:
			raise Exception("Layer not trainable")
		W = self.W
		grad_w = np.dot(self.layer_input.T, accum_grad) # dW = X_T * dZ
		grad_w0 = np.sum(accum_grad, keepdims = True, axis = 0) # dw0 = sum(dZ)

		accum_grad = np.dot(accum_grad, W.T) # dX = dZ * W_T

		self.W = self.W_opt.update(self.W, grad_w)
		self.w0 = self.w0_opt.update(self.w0, grad_w0)
		
		return accum_grad

	def output_shape(self):
		x = self.forward_pass
		# Return output shape tuple
		return np.prod(x.shape)
