# Machine Learning HW1 - Polynomial Regression with Regularization

**Course**: Machine Learning 2025 Fall
**Student ID**: 313553058
**Name**: Ian Tsai

## Overview

Implementation of polynomial regression using three different optimization methods with regularization techniques. This project demonstrates the mathematical foundations and practical implementation of fundamental machine learning algorithms.

## Features

- **Least Squares Estimation (LSE)** with L2 Regularization
- **Steepest Descent Method** with L1 Regularization
- **Newton's Method** without regularization
- Custom matrix operations (LU decomposition for matrix inversion)
- Visualization of regression results

## Project Structure

```
Deliver/
├── main_regression_2025.py         # Main program entry point
├── regression_solver_2025.py       # Core algorithm implementations
├── test_data.txt                   # Test dataset
├── ML_HW1_313553058mathDervation.pdf  # Mathematical derivation PDF
└── README.md                       # This file
```

## Quick Start

### Prerequisites

- Python 3.7+
- NumPy
- Matplotlib

### Installation

```bash
# Clone the repository
git clone https://github.com/IaTsai/ML-HW1.git
cd ML-HW1

# Install dependencies
pip install numpy matplotlib
```

### Running the Program

```bash
python3 main_regression_2025.py test_data.txt
```

You will be prompted to enter:

1. **n**: Number of polynomial bases (e.g., 2 for linear, 3 for quadratic)
2. **λ**: Regularization parameter (0 for no regularization)

### Example Test Cases

#### Case 1: Linear Regression (n=2, λ=0)

```
Enter the number of polynomial bases n: 2
Enter lambda (for LSE and Steepest descent): 0
```

#### Case 2: Quadratic Polynomial (n=3, λ=0)

```
Enter the number of polynomial bases n: 3
Enter lambda (for LSE and Steepest descent): 0
```

#### Case 3: Quadratic with Strong Regularization (n=3, λ=10000)

```
Enter the number of polynomial bases n: 3
Enter lambda (for LSE and Steepest descent): 10000
```

## Expected Results

### Key Observations

1. **When λ=0**: LSE and Newton's Method produce identical results (both solve the same unregularized problem)

2. **When λ>0**:

   - LSE coefficients shrink due to L2 regularization
   - Steepest Descent may produce sparse solutions (L1 regularization)
   - Newton's Method remains unaffected (no regularization)

3. **Convergence**:
   - LSE: Direct solution (one step)
   - Newton's Method: One step for quadratic functions
   - Steepest Descent: Iterative convergence (may require many iterations)

## Implementation Details

### Algorithm Specifications

| Method           | Regularization | Complexity          | Key Feature                 |
| ---------------- | -------------- | ------------------- | --------------------------- |
| LSE              | L2 (Ridge)     | O(m³)               | Direct closed-form solution |
| Steepest Descent | L1 (Lasso)     | O(nm) per iteration | Produces sparse solutions   |
| Newton's Method  | None           | O(m³) per iteration | Quadratic convergence       |

### Matrix Operations

All matrix operations are implemented from scratch:

- **LU Decomposition**: For computing matrix inverse in LSE
- **Forward/Backward Substitution**: For solving linear systems
- No use of `numpy.linalg.inv()` or similar built-in functions

### Convergence Criteria

- **Steepest Descent**:
  - Learning rate: η = 0.00001
  - Max iterations: 250,000
  - Convergence tolerance: Changes in gradient norm

## Visualization

The program generates three subplots showing:

1. LSE with L2 regularization results
2. Steepest Descent with L1 regularization results
3. Newton's Method results

Each plot displays:

- Red dots: Original data points
- Blue curve: Fitted polynomial

## Mathematical Background

The complete mathematical derivation is available in:

- `ML_HW1_313553058mathDervation.pdf` - Full mathematical proof and derivation

Key concepts covered:

- Normal equation derivation
- Gradient descent convergence analysis
- Newton's method quadratic approximation
- L1 vs L2 regularization effects

## Notes

### Important Implementation Choices

1. **Sign function for L1 regularization**:

   ```python
   sign(w) = { 1 if w > 0, -1 if w < 0, 0 if w = 0 }
   ```

2. **Learning rate selection**:

   - Carefully tuned to balance convergence speed and stability
   - Too large: divergence
   - Too small: slow convergence

3. **Numerical stability**:
   - Added λI to ensure matrix invertibility
   - Used LU decomposition instead of direct inversion
