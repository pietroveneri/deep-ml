def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	means = []
	rows = len(matrix)
	cols = len(matrix[0])
	if mode == "row":
		for row in range(rows):
			sums = sum(matrix[row])
			means.append(sums / cols)
	elif mode == "column":
		for col in range(cols):
			sums = 0
			for row in range(rows): 
				sums += matrix[row][col]
			means.append(sums / rows)
	
	return means