"""
Regression Solver Implementation
Machine Learning HW1 - 2025 Fall

Core algorithms for polynomial regression:
- Matrix class: Custom matrix operations without using numpy's linear algebra
- LSE: Least Squares Estimation with L2 regularization
- Steepest Descent: Gradient descent with L1 regularization
- Newton's Method: Second-order optimization

All matrix operations (especially inverse) are implemented from scratch.

Author: 313553058 Ian Tsai
"""

import numpy as np

class Matrix:
    """
    Custom matrix class implementing essential linear algebra operations.

    This class provides basic matrix operations needed for regression
    without relying on numpy's built-in linear algebra functions.
    All operations, especially matrix inverse, are implemented from scratch.
    """

    def __init__(self, matrix):
        """
        Initialize matrix with given 2D list.

        Args:
            matrix: 2D list representing the matrix
        """
        self.m = len(matrix)        # Number of rows
        self.n = len(matrix[0])     # Number of columns
        self.matrix = [list(row) for row in matrix]  # Deep copy to avoid reference issues

    def __str__(self):
        """
        String representation for debugging purposes.

        Returns:
            Formatted string showing matrix values with 5 decimal places
        """
        return '\n'.join([' '.join([f"{val:.5f}" for val in row]) for row in self.matrix])

    def PrintMatrix(self):
        """Print matrix row by row for debugging."""
        for row in self.matrix:
            print(row)

    def Transpose(self):
        """
        Compute matrix transpose.

        Returns:
            Matrix: Transposed matrix where A[i][j] becomes A^T[j][i]
        """
        return Matrix([[self.matrix[j][i] for j in range(self.m)] for i in range(self.n)])

    def Multiply(self, matrix2):
        """
        Matrix multiplication: self × matrix2.

        Args:
            matrix2: Matrix to multiply with

        Returns:
            Matrix: Result of matrix multiplication
        """
        return Matrix([[sum(i * j for i, j in zip(row, col))
                       for col in matrix2.Transpose().matrix]
                       for row in self.matrix])

    def Add(self, matrix2, n):
        """
        Matrix addition with scalar multiplication: self + n × matrix2.

        Args:
            matrix2: Matrix to add
            n: Scalar multiplier for matrix2

        Returns:
            Matrix: Result of self + n × matrix2
        """
        return Matrix([[a + b * n for (a, b) in zip(self.matrix[i], matrix2.matrix[i])]
                      for i in range(self.m)])

    def LUDecomposition(self):
        """
        Perform LU decomposition without pivoting.

        Decomposes matrix A into L (lower triangular) and U (upper triangular)
        such that A = LU. This is used for computing matrix inverse.

        Returns:
            tuple: (L, U) matrices where L is lower triangular with 1s on diagonal

        Raises:
            ValueError: If matrix is singular (zero on diagonal during elimination)
        """
        size = self.m
        L = [[0] * size for _ in range(size)]  # Initialize L as zero matrix
        U = [list(row) for row in self.matrix]  # U starts as copy of original matrix

        # Gaussian elimination process
        for i in range(size):
            # Check for zero pivot (singular matrix)
            if U[i][i] == 0:
                raise ValueError("LU decomposition failed: Zero detected on diagonal.")

            L[i][i] = 1.0  # Diagonal of L is always 1

            # Eliminate column i below diagonal
            for j in range(i + 1, size):
                scalar = U[j][i] / U[i][i]  # Multiplier for row operation
                L[j][i] = scalar  # Store multiplier in L
                # Update row j of U: U[j] = U[j] - scalar × U[i]
                U[j] = [val_1 - scalar * val_2 for (val_1, val_2) in zip(U[j], U[i])]

        return Matrix(L), Matrix(U)

    def Inverse(self):
        """
        Compute matrix inverse using LU decomposition.

        Solves AX = I by:
        1. Decomposing A = LU
        2. Solving LY = I for Y (forward substitution)
        3. Solving UX = Y for X (backward substitution)

        Returns:
            Matrix: Inverse of the matrix
        """
        L, U = self.LUDecomposition()
        size = self.m
        # Create identity matrix
        identity = Matrix([[1 if i == j else 0 for j in range(size)] for i in range(size)])

        # Solve LY = I, then UX = Y
        y = SolveLy_B(L, identity)
        return SolveUx_y(U, y)

def SolveLy_B(L, B):
    """
    Solve lower triangular system Ly = B using forward substitution.

    For each column of B, solves Ly = b from top to bottom.

    Args:
        L: Lower triangular matrix
        B: Right-hand side matrix

    Returns:
        Matrix: Solution matrix Y
    """
    size = L.m
    Y = [[0] * size for _ in range(size)]

    # Process each column of B
    for j in range(size):
        # Forward substitution for column j
        for i in range(j + 1):
            Y[j][i] = B.matrix[j][i]
            # Subtract contributions from previous solutions
            for k in range(j):
                Y[j][i] -= L.matrix[j][k] * Y[k][i]
            Y[j][i] /= L.matrix[j][j]

    return Matrix(Y)

def SolveUx_y(U, Y):
    """
    Solve upper triangular system Ux = Y using backward substitution.

    For each column of Y, solves Ux = y from bottom to top.

    Args:
        U: Upper triangular matrix
        Y: Right-hand side matrix

    Returns:
        Matrix: Solution matrix X
    """
    size = U.m
    X = [[0] * size for _ in range(size)]

    # Process from bottom row to top
    for j in range(size - 1, -1, -1):
        for i in range(size):
            X[j][i] = Y.matrix[j][i]
            # Subtract contributions from already solved variables
            for k in range(j + 1, size):
                X[j][i] -= U.matrix[j][k] * X[k][i]
            X[j][i] /= U.matrix[j][j]

    return Matrix(X)

def ComputeError(A, b, coef):
    """
    Calculate sum of squared errors: ||Aw - b||²

    Args:
        A: Design matrix (n × m)
        b: Target vector (n × 1)
        coef: Weight vector (m × 1)

    Returns:
        float: Sum of squared residuals
    """
    error_matrix = A.Multiply(coef).Add(b, -1)  # Aw - b
    return sum([error[0] ** 2 for error in error_matrix.matrix])

def LeastSquaresEstimation(A, b, lambda_val):
    """
    Solve regression using closed-form LSE with L2 regularization (Ridge).

    Minimizes: ||Aw - b||² + λ||w||²
    Solution: w = (A^T A + λI)^(-1) A^T b

    The regularization term λI ensures the matrix is invertible
    and prevents overfitting by penalizing large weights.

    Args:
        A: Design matrix (n × m)
        b: Target vector (n × 1)
        lambda_val: L2 regularization parameter

    Returns:
        Matrix: Optimal weight vector w*
    """
    ATA = A.Transpose().Multiply(A)  # Compute A^T A
    ATb = A.Transpose().Multiply(b)  # Compute A^T b

    # Add L2 regularization term λI to A^T A
    # This makes the matrix positive definite and ensures invertibility
    identity = Matrix([[1 if i == j else 0 for j in range(A.n)] for i in range(A.n)])
    ATA_lI = ATA.Add(identity, lambda_val)  # A^T A + λI

    # Solve normal equation: (A^T A + λI)w = A^T b
    return ATA_lI.Inverse().Multiply(ATb)

def SteepestDescent_L1(A, b, lr=0.000002, tol=1e-6, max_iter=800000, lambda_val=0):
    """
    Iterative optimization using gradient descent with L1 regularization (Lasso).

    Minimizes: ||Aw - b||² + λ||w||₁

    L1 regularization promotes sparsity in the solution by using
    the subgradient of the absolute value function (sign function).

    Gradient: 2A^T(Aw - b) + λ·sign(w)
    Update rule: w^(k+1) = w^(k) - η[2A^T(Aw - b) + λ·sign(w)]

    Args:
        A: Design matrix (n × m)
        b: Target vector (n × 1)
        lr: Learning rate (step size)
        tol: Convergence tolerance
        max_iter: Maximum number of iterations
        lambda_val: L1 regularization parameter

    Returns:
        Matrix: Optimized weight vector
    """
    w = Matrix([[0]] * A.n)  # Initialize weights to zero

    for iteration in range(max_iter):
        # Calculate residual: r = Aw - b
        residual = A.Multiply(w).Add(b, -1)

        # Calculate gradient of squared error: ∇f = 2A^T(Aw - b)
        # Note: The factor of 2 is absorbed into the learning rate
        gradient_lse = A.Transpose().Multiply(residual)

        # Calculate L1 regularization gradient: λ·sign(w)
        # sign(w) is the subgradient of |w|
        l1_gradient = Matrix([[lambda_val * np.sign(w.matrix[i][0])] for i in range(A.n)])

        # Total gradient = gradient of squared error + L1 penalty gradient
        gradient = gradient_lse.Add(l1_gradient, 1)

        # Update weights: w^(k+1) = w^(k) - lr × gradient
        w_new = w.Add(gradient, -lr)

        # Check for numerical instability
        if any(np.isnan(w_new.matrix[i][0]) for i in range(len(w_new.matrix))):
            print(f"Iteration {iteration}: Detected NaN values in weights, stopping early.")
            break

        # Check convergence: if max change in weights < tolerance
        if max(abs(w_new.matrix[i][0] - w.matrix[i][0]) for i in range(len(w.matrix))) < tol:
            break

        w = w_new

    return w

def NewtonMethod(A, b):
    """
    Newton's method for quadratic optimization (no regularization).

    For least squares problems, Newton's method converges in one iteration
    because the objective function is quadratic.

    The Hessian (second derivative) is constant: H = 2A^T A
    The gradient (first derivative) is: ∇f = 2A^T(Aw - b)

    At optimum, ∇f = 0, which gives: A^T Aw = A^T b
    Solution: w = (A^T A)^(-1) A^T b

    This is equivalent to LSE with λ = 0 (no regularization).

    Args:
        A: Design matrix (n × m)
        b: Target vector (n × 1)

    Returns:
        Matrix: Optimal weight vector (exact solution for quadratic problems)
    """
    H = A.Transpose().Multiply(A)    # Hessian matrix = A^T A
    grad = A.Transpose().Multiply(b)  # Gradient at w=0 is -A^T b, we want A^T b

    # Newton's update: w = w_0 - H^(-1)∇f(w_0)
    # Since w_0 = 0 and ∇f(0) = -A^T b, we get: w = H^(-1) A^T b
    return H.Inverse().Multiply(grad)