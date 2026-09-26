class Value:
	def __init__(self, data, _children=(), _op=''):
		self.data = data
		self.grad = 0
		self._backward = lambda: None
		self._prev = set(_children)
		self._op = _op
	def __repr__(self):
		def fmt(x):
			return int(x) if float(x).is_integer() else round(x, 4)
		return f"Value(data={fmt(self.data)}, grad={fmt(self.grad)})"

	def __add__(self, other):
		out = Value(self.data + other.data, 
					_children = (self, other), 
					_op="+")
		
		def _backward():
			self.grad += out.grad
			other.grad += out.grad
		out._backward = _backward
		return out
		
	def __mul__(self, other):
		out = Value(self.data * other.data,
		_children = (self, other), 
		_op = "*"
		)
		def _backward():
			self.grad += other.data * out.grad
			other.grad += self.data * out.grad
		out._backward = _backward
		return out

	def relu(self):
		out = Value(self.data if self.data > 0 else 0, _children=(self,), _op="ReLU")
		def _backward():
			self.grad += (1 if self.data > 0 else 0 ) * out.grad

		out._backward = _backward
		return out

	def backward(self):
		topo = []
		visited = set()

		def build_topo(v):
			if v not in visited:
				visited.add(v)
				for child in v._prev:
					build_topo(child)
				topo.append(v)
		build_topo(self)
		self.grad = 1
		for node in reversed(topo):
			node._backward()