import numpy as np
from math_utils import RationalMatrix


det = lambda x: np.linalg.det(x)

def cramer(A,b) -> np.ndarray:
    """ applies Cramer's method to the given 
    coefficient matrix. 
    We can't use this method if A is not invertible or 
    b=0 (if system is homogenous)
    """
    assert (det(A) != 0) and (not (np.all(b==0))), "can't use this method if A is not invertible or b=0 (if system is homogenous)"
    det_a = det(A)
    det_ai = []
    for col in range(A.shape[1]):
        matrix = A.copy()
        matrix[:, col] = b
        det_ai.append(det(matrix))
    
    return np.array(det_ai, dtype = np.float64) / det_a


def in_error_bounds(A, b, x, x_a, return_bounds = False):
    norm_vector = lambda a: np.dot(a,a)
    norm_matrix = lambda a: np.sqrt((np.sum(np.multiply(a,a))))
    A_inv = np.linalg.inv(A)
    e = x - x_a
    r = np.dot(A,e)
    kA = norm_matrix(A) * norm_matrix(A_inv)
    lower_error_bound = 1/kA * (norm_vector(r) / norm_vector(b)) 
    upper_error_bound = kA * (norm_vector(r) / norm_vector(b)) 
    relative_error = norm_vector(e) / norm_vector(x)
    if return_bounds:
        return (lower_error_bound <= relative_error <= upper_error_bound), (lower_error_bound, upper_error_bound)
    else:
        return (lower_error_bound <= relative_error <= upper_error_bound)


def is_convergence(T):
    eigvals = np.linalg.eigvals(T)
    max_eigval = max(map(abs, eigvals))
    return (max_eigval < 1)

def jacobi(A: np.ndarray, b: np.ndarray, starting_point: tuple = None,
           error_constraint: float = 1e-5, debug=0, verbose=0):
    """
    Applies Jacobi iteration from given 
    starting point to given Ax=b linear system.
    We can use Jacobi iteration in order to approximate 
    the true solution of given system:
    Let's start by finding A = L + D + U decomposition.
    Here,
    - L is lower triangular part of A
    - D is the diagonal of A
    - U is the upper triangular part of A
    Then we can substite Ax=b to (L+D+U)x=b.
    Therefore,

       Dx+(L+U)x=b
    => Dx=-(L+U)x+b     (1)
    If diag(A) does not contain 0, we conclude that D is invertible.
    Because det(D)=d_1 x d_2 x ... x d_n where d_i is an element of diag(D)
    If diag(A) does not contain any zeros, then det(D) =/= 0 and D is invertible.


    So let's suppose the inverse of D is D' and use this information in (1):

        Dx=-(L+U)x+b
     => x = -D'(L+U)x+D'b

    From given starting point we set x_0 = starting point and start the iteration:
        x_k = -D'(L+U)x_{k-1}+D'b

    Generally we define Jacobi and Gauss-Seidel iterations as
        x = Tx+c

    You can see that T = -D'(L+U) and c = D'b for Jacobi iteration

    Any approximation need to stop, we need a constraint to finish the approximation
    It would be useful if we use the error of approximation and try to minimize that.

    In order to use Jacobi iteration,
    we must be sure that T's infinity-norm, l1-norm, and l2-norm are less than 1.

    For any iterative approximation method, if spectral radius (absolute maximum eigenvalue) is less than 1
    the method converges.
    Another condition for the convergence is iteration matrix T being strictly diagonally dominant
    (diagonals are higher than the other values in the matrix, if diagonals are "greater or equal" than T is partially diagonally dominant)
    
    """
    D = np.diag(np.diag(A))
    L, U = np.tril(A) - D, np.triu(A) - D
    D_inv = np.linalg.inv(D)
    X_i = np.array(starting_point)
    k = 1
    error = np.infty
    if debug:
        print("DEBUG:")
        print("L,D,U MATRICES:")
        print(L)
        print(D)
        print(U)
        print("STARTING POINT:", X_i)
        print("D_INV=",D_inv)
    
    if verbose:
        print("Starting the approximation from point:", X_i)
    while error > error_constraint:
        T = -(D_inv @ (L+U))
        if not is_convergence(T): return "Approximation does not converges"
        X_j = np.dot(T, X_i) + np.dot(D_inv, b)

        error = np.dot((X_j - X_i),(X_j - X_i)) / np.dot(X_j, X_j)
        X_i = X_j

        if verbose:
            print(f"Iteration K={k} | Error = {error:.8f} | Approximate solution: {X_i}")
        k += 1
    return X_i

def gauss_seidel(A: np.ndarray, b: np.ndarray, starting_point: tuple=None, error_constraint: float=1e-5, debug = 0, verbose = 0):
    """
    Applies Gauss-Seiel iteration from given starting point to given Ax=b linear system.
    We can use Gauss-Seiel iteration in order to approximate the true solution of given system:
    Let's start by finding A = L + D + U decomposition.
    Here, 
    - L is lower triangular part of A
    - D is the diagonal of A
    - U is the upper triangular part of A
    Then we can substite Ax=b to (L+D+U)x=b.
    Therefore,

        (D+L)x+Ux=b
    =>  (D+L)x=-Ux+b     (1)
    If diag(A) does not contain 0, we conclude that D+L is invertible.
    Because det(D+L)=d_1 x d_2 x ... x d_n where d_i is an element of diag(D+L)
    since D+L is a lower triangular matrix.
    If diag(A) does not contain any zeros, then det(D+L) =/= 0 and D+L is invertible.


    So let's suppose the inverse of (D+L) is (D+L)' and use this information in (1): 

        (D+L)x=-Ux+b
     => x = -(D+L)'Ux + (D+L)'b
    
    From given starting point we set x_0 = starting point and start the iteration: 
        x_k = -(D+L)'Ux_{k-1} + (D+L)'b

    Generally we define Jacobi and Gauss-Seidel iterations as 
        x = Tx+c
    You can see that T = -(D+L)'U and c = (D+L)'b for Gauss-Seidel iteration

    Any approximation need to stop, we need a constraint to finish the approximation
    It would be useful if we use the error of approximation and try to minimize that.

    In order to use Gauss-Seidel iteration, 
    we must be sure that T's infinity-norm, l1-norm, and l2-norm are less than 1.

    For any iterative approximation method, if spectral radius (absolute maximum eigenvalue) is less than 1
    the method converges. 
    Another condition for the convergence is iteration matrix T being strictly diagonally dominant 
    (diagonals are higher than the other values in the matrix, if diagonals are "greater or equal" than T is partially diagonally dominant)

    """
    D = np.diag(np.diag(A))
    L, U = np.tril(A) - D, np.triu(A) - D
    Inv = np.linalg.inv(D+L)
    X_i = np.array(starting_point)
    k = 1
    error = np.infty 
    if debug:
        print("DEBUG:")
        print("L,D,U MATRICES:")
        print(L) 
        print(D)
        print(U)
        print("STARTING POINT:", X_i)
        print("D+L INV=",Inv)
    
    if verbose:
        print("Starting the approximation from point:", X_i)
    while error > error_constraint:
        T = -(Inv @ U)
        if not is_convergence(T): return "Approximation does not converges"
        X_j = - np.dot(T, X_i) + np.dot(Inv, b)

        error = np.dot((X_j - X_i),(X_j - X_i)) / np.dot(X_j, X_j)
        X_i = X_j

        if verbose:
            print(f"Iteration K={k} | Error = {error:.8f} | Approximate solution: {X_i}")
        k += 1
    return X_i