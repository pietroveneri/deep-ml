def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	eigenvalues = []
	trace = matrix[0][0] + matrix[1][1]
	det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
	eigen1 = (trace + (trace**2 - 4 * det)**0.5) / 2
	eigen2 = (trace - (trace**2 - 4 * det)**0.5) / 2
	eigenvalues.append(eigen1)
	eigenvalues.append(eigen2)
	eigenvalues.sort(reverse = True)
	return eigenvalues