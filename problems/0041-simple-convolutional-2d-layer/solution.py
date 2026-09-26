import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape
	b = np.zeros((input_height + 2 * padding, input_width + 2 * padding))
	b_height, b_width = b.shape
	x_offset = padding
	y_offset = padding
	b[x_offset:input_width + x_offset, y_offset:input_height + y_offset] = input_matrix
	output_width = int((b_width - kernel_width) / stride + 1)
	output_height = int((b_height - kernel_height) / stride + 1)	
	output_matrix = np.zeros((output_height, output_width), dtype = "float")
	for i in range(output_height):
		for j in range(output_width):
			row = i * stride
			col = j * stride
			output_matrix[i][j] = np.sum(b[row:row+kernel_height, col:col + kernel_width]*kernel)
	return output_matrix
