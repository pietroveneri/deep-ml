def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    """
    Calculate the inverse of a 2x2 matrix.
    
    Args:
        matrix: A 2x2 matrix represented as [[a, b], [c, d]]
    
    Returns:
        The inverse matrix as a 2x2 list, or None if the matrix is singular
        (i.e., determinant equals zero)
    """
    a= matrix[0][0] 
    b=matrix[0][1] 
    c=matrix[1][0] 
    d=matrix[1][1] 
    det = a*d - b*c
    if det == 0:
        return None
    A_inv = [[d, -b], [-c, a]]
    def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
        # Your code here
        for i in range(len(matrix)): 
            for j in range(len(matrix[0])):
                matrix[i][j] *= scalar
        return matrix
    A_inv = scalar_multiply(A_inv, 1/det)
    return A_inv