import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	def canBeReshaped(a, new_shape):
		return (new_shape[0] * new_shape[1] == len(a) * len(a[0]))
	if not canBeReshaped(a, new_shape):
		blank = []
		return blank 
	b = []
	for row in a:
		for value in row:
			b.append(value)
	k = 0
	reshaped_matrix = [[0 for _ in range(new_shape[1])] for _ in range(new_shape[0])]
	for i in range(new_shape[0]):
		for j in range(new_shape[1]) :
			reshaped_matrix[i][j] = b[k]
			k += 1	
	
	return reshaped_matrix