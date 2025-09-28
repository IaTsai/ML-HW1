"""
Polynomial Regression with Regularization
Machine Learning HW1 - 2025 Fall

This program implements three regression methods:
1. Least Squares Estimation (LSE) with L2 regularization
2. Steepest Descent with L1 regularization
3. Newton's Method without regularization

Author: 313553058 Ian Tsai
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from regression_solver_2025 import Matrix, LeastSquaresEstimation, SteepestDescent_L1, NewtonMethod, ComputeError

def format_equation(coef):
    """
    Convert coefficient matrix to readable polynomial equation string.

    Args:
        coef: Matrix object containing polynomial coefficients
              from highest to lowest degree

    Returns:
        String representation of polynomial equation
        e.g., "3.0238 x^2 + 4.9061 x^1 - 0.2314"
    """
    n = len(coef.matrix)
    terms = []
    for i in range(n):
        power = n - 1 - i
        coeff_val = coef.matrix[i][0]
        if power == 0:
            terms.append(f"{coeff_val:.10f}")
        else:
            terms.append(f"{coeff_val:.10f} x^{power}")

    equation = ""
    for i, term in enumerate(terms):
        if i == 0:
            equation = term
        else:
            if float(term.split()[0]) >= 0:
                equation += f" + {term}"
            else:
                equation += f" {term}"

    return equation

def plot_regression(coef, data_points, title, subplot_index, total_plots):
    """
    Create visualization of regression results with data points and fitted curve.

    Args:
        coef: Matrix object containing polynomial coefficients
        data_points: List of (x, y) tuples representing training data
        title: String title for the subplot
        subplot_index: Position of current subplot (1-based)
        total_plots: Total number of subplots to create

    Returns:
        None (displays plot)
    """
    x_vals = np.linspace(-6, 6, 100)
    y_vals = sum(coef.matrix[i][0] * x_vals**(len(coef.matrix) - i - 1) for i in range(len(coef.matrix)))

    plt.subplot(total_plots, 1, subplot_index)
    plt.plot(x_vals, y_vals, label=title, color="blue", linewidth=2)
    data = np.array(data_points)
    plt.scatter(data[:, 0], data[:, 1], c='red', s=20, edgecolors='black', alpha=0.7)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-6, 6)

if __name__ == '__main__':
    # Check command line arguments
    if len(sys.argv) < 2:
        print('Usage: python main_regression_2025.py <testfile>')
        sys.exit(1)

    # Load training data from input file
    data_points = []
    with open(sys.argv[1], 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                x, y = map(float, line.split(','))
                data_points.append((x, y))
            except ValueError:
                print(f"Warning: Invalid format, skipping line → {line}")

    # Get polynomial degree and regularization parameter from user
    n = int(input('Enter the number of polynomial bases n: '))
    lambda_val = float(input('Enter lambda (for LSE and Steepest descent): '))

    # Build design matrix A (Vandermonde matrix) and target vector b
    # A[i][j] = x_i^(n-1-j), where x_i is the i-th data point
    A = Matrix([[x[0] ** (n - 1 - i) for i in range(n)] for x in data_points])
    b = Matrix([[x[1]] for x in data_points])

    print("\n" + "="*60)
    print("POLYNOMIAL REGRESSION RESULTS")
    print("="*60)
    print(f"Number of bases: {n}")
    print(f"Lambda value: {lambda_val}")
    print(f"Data points: {len(data_points)}")
    print("="*60 + "\n")

    # Method 1: Least Squares Estimation with L2 regularization (Ridge regression)
    # Closed-form solution: w = (A^T A + λI)^(-1) A^T b
    print("METHOD 1: Closed-form LSE (with L2 Regularization)")
    print("-" * 50)
    coef_LSE = LeastSquaresEstimation(A, b, lambda_val)
    lse_error = ComputeError(A, b, coef_LSE)
    print(f"Fitting line: {format_equation(coef_LSE)}")
    print(f"Total error: {lse_error:.10f}\n")

    # Method 2: Gradient Descent with L1 regularization (Lasso regression)
    # Iterative optimization using subgradient of L1 norm
    print("METHOD 2: Steepest Descent (with L1 Regularization)")
    print("-" * 50)
    coef_SD = SteepestDescent_L1(A, b, lr=0.00001, max_iter=250000, lambda_val=lambda_val)
    sd_error = ComputeError(A, b, coef_SD)
    print(f"Fitting line: {format_equation(coef_SD)}")
    print(f"Total error: {sd_error:.10f}\n")

    # Method 3: Newton's Method for quadratic optimization
    # Second-order method, converges in one step for quadratic functions
    print("METHOD 3: Newton's Method (no regularization)")
    print("-" * 50)
    coef_Newton = NewtonMethod(A, b)
    newton_error = ComputeError(A, b, coef_Newton)
    print(f"Fitting line: {format_equation(coef_Newton)}")
    print(f"Total error: {newton_error:.10f}\n")

    # Generate visualization: compare all three methods
    plt.figure(figsize=(10, 12))
    plt.suptitle(f'Polynomial Regression Results (n={n}, λ={lambda_val})', fontsize=14, fontweight='bold')

    plot_regression(coef_LSE, data_points, f'LSE (L2 Reg, λ={lambda_val})', 1, 3)
    plot_regression(coef_SD, data_points, f'Steepest Descent (L1 Reg, λ={lambda_val})', 2, 3)
    plot_regression(coef_Newton, data_points, "Newton's Method (No Reg)", 3, 3)

    plt.tight_layout()
    plt.show()