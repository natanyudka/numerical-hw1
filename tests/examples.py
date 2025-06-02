import numpy as np


#ensure x is a 1D array (vector)
def _ensure_vector(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 0: 
        x = np.array([x], dtype=float)
    elif x.ndim > 1: 
        x = x.flatten() 
    return x

# 3.c.i
def quadratic_example_circle(x, hessian_needed=True):
    """
    Objective function f(x) = x^T Q x for Q = [[1, 0], [0, 1]].
    This results in circular contour lines.
    f(x) = x1^2 + x2^2
    grad(f) = [2*x1, 2*x2]^T
    hess(f) = [[2, 0], [0, 2]]
    """
    x = _ensure_vector(x)
    Q = np.array([[1.0, 0.0], 
                  [0.0, 1.0]])
    
    f_val = x.T @ Q @ x
    grad_val = 2 * Q @ x
    h_val = 2 * Q if hessian_needed else None
    return f_val, grad_val, h_val

# 3.c.ii
def quadratic_example_axis_ellipses(x, hessian_needed=True):
    """
    f(x) = x^T Q x for Q = [[1, 0], [0, 100]] (axis-aligned ellipses)
    f(x) = x1^2 + 100*x2^2
    """
    x = _ensure_vector(x)
    Q = np.array([[1.0, 0.0], 
                  [0.0, 100.0]])

    f_val = x.T @ Q @ x
    grad_val = 2 * Q @ x
    h_val = 2 * Q if hessian_needed else None
    return f_val, grad_val, h_val


# 3.c.iii.
def quadratic_example_rotated_ellipses(x, hessian_needed=True):
    """
    f(x) = x^T Q x for Q = R^T D R (rotated ellipses)
    R = rotation by pi/6 (30 deg)
    D = [[100, 0], [0, 1]]
    """
    x = _ensure_vector(x)
    # R = np.array([
    #     [np.sqrt(3)/2, -0.5],
    #     [0.5, np.sqrt(3)/2]
    # ])
    # D = np.array([[100.0, 0.0],
    #               [0.0, 1.0]])
    # Q = R.T @ D @ R
    # Using pre-calculated Q for simplicity and to match assignment's Q structure
    # Q = [[(sqrt(3)/2), 0.5], [-0.5, (sqrt(3)/2)]] @ [[100,0],[0,1]] @ [[(sqrt(3)/2), -0.5], [0.5, (sqrt(3)/2)]]
    # Q = [[75.25, -24.75*sqrt(3)], [-24.75*sqrt(3), 25.75]]
    Q = np.array([
        [75.25, -24.75 * np.sqrt(3)],
        [-24.75 * np.sqrt(3), 25.75]
    ])

    f_val = x.T @ Q @ x
    grad_val = 2 * Q @ x
    h_val = 2 * Q if hessian_needed else None
    return f_val, grad_val, h_val

# 3.d
def rosenbrock_function(x, hessian_needed=True):
    """
    Rosenbrock function: f(x1, x2) = 100*(x2 - x1^2)^2 + (1 - x1)^2
    """
    x = _ensure_vector(x)
    x1, x2 = x[0], x[1]

    f_val = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    
    grad_val = np.array([
        -400 * x1 * (x2 - x1**2) - 2 * (1 - x1),
        200 * (x2 - x1**2)
    ])

    h_val = None
    if hessian_needed:
        h_val = np.array([
            [1200 * x1**2 - 400 * x2 + 2, -400 * x1],
            [-400 * x1, 200.0]
        ])
    return f_val, grad_val, h_val


# 3.e
def linear_function(x, hessian_needed=True, a_vector=None):
    """
    Linear function: f(x) = a^T x
    Default a = [1, 2]^T
    """
    x = _ensure_vector(x)
    if a_vector is None:
        a = np.array([1.0, 2.0])
    else:
        a = np.asarray(a_vector, dtype=float)
    
    if a.shape != x.shape: # Basic check, assumes a is 1D like x, adjust 'a' if it's for a different dimension than x
        if x.shape[0] == 2 and a.shape[0] != 2: 
             a = np.array([1.0, 1.0])[:x.shape[0]]
        elif a.shape[0] != x.shape[0]:
             raise ValueError(f"Dimension mismatch between x ({x.shape}) and a ({a.shape})")


    f_val = a.T @ x
    grad_val = a

    h_val = None
    if hessian_needed:
        h_val = np.zeros((x.shape[0], x.shape[0]))
    return f_val, grad_val, h_val

# 3.f
def boyd_example_function(x, hessian_needed=True):
    x = _ensure_vector(x)
    x1, x2 = x[0], x[1]

    term1 = np.exp(x1 + 3*x2 - 0.1)
    term2 = np.exp(x1 - 3*x2 - 0.1)
    term3 = np.exp(-x1 - 0.1)

    f_val = term1 + term2 + term3

    grad_val = np.array([
        term1 + term2 - term3,
        3 * term1 - 3 * term2
    ])

    h_val = None
    if hessian_needed:
        h_val = np.array([
            [term1 + term2 + term3, 3 * term1 - 3 * term2],
            [3 * term1 - 3 * term2, 9 * term1 + 9 * term2]
        ])
    return f_val, grad_val, h_val