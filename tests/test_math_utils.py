from fractions import Fraction

import math_utils as mu
from math_utils.iterative import linear, nonlinear

import numpy as np
from math_utils import __version__


def test_version():
    assert __version__ == '0.1.0'

class TestRationalMatrix:
    def test_from_numpy(self):
        M = np.array([
            [2, -2, 4, -2],
            [2, 1, 10, 7],
            [-4, 4, -8, 4],
            [4, -1, 14, 6]
        ])

        rm_output = mu.RationalMatrix(M)
        correct = [[Fraction(2, 1), Fraction(-2, 1), Fraction(4, 1), Fraction(-2, 1)],
                    [Fraction(2, 1), Fraction(1, 1), Fraction(10, 1), Fraction(7, 1)],  
                    [Fraction(-4, 1), Fraction(4, 1), Fraction(-8, 1), Fraction(4, 1)],
                    [Fraction(4, 1), Fraction(-1, 1), Fraction(14, 1), Fraction(6, 1)]]
        assert rm_output == correct, f"{type(rm_output)} =/= {type(correct)}"

    def test_from_list(self):
        M = [
            [2, -2, 4, -2],
            [2, 1, 10, 7],
            [-4, 4, -8, 4], 
            [4, -1, 14, 6]
        ]
        rm_output = mu.RationalMatrix(M)
        correct = [[Fraction(2, 1), Fraction(-2, 1), Fraction(4, 1), Fraction(-2, 1)],
                    [Fraction(2, 1), Fraction(1, 1), Fraction(10, 1), Fraction(7, 1)],
                    [Fraction(-4, 1), Fraction(4, 1), Fraction(-8, 1), Fraction(4, 1)],
                    [Fraction(4, 1), Fraction(-1, 1), Fraction(14, 1), Fraction(6, 1)]]
        
        assert rm_output == correct, rm_output
    
    def test_copy(self):
        M = [
            [2, -2, 4, -2],
            [2, 1, 10, 7],
            [-4, 4, -8, 4], 
            [4, -1, 14, 6]
        ]
        rm = mu.RationalMatrix(M)
        copy = rm.copy()
        assert (not (rm is copy)) and (rm.rmatrix == copy.rmatrix)
    
class TestLinearIterative:
    def test_jacobi_converges(self):
        a = np.array([
            [6, -2, 1],
            [-2, 7, 2],
            [1, 2, -5]
        ])
        b = np.array([
            11,5,-1
        ])
        x = np.linalg.solve(a,b)
        approx_x = linear.jacobi(a,b,(0,0,0), 1e-4, verbose = 0)
        assert linear.in_error_bounds(a, b, x, approx_x)

        
    def test_jacobi_diverges(self):
        a = np.array([
                    [88, 67,  6, 96, 79],
                    [27, 60, 22, 80, 68],
                    [73, 16, 22, 48, 70],
                    [ 7, 19, 42,  8,  5],
                    [83,  2,  7, 82, 52],
                    ])

        b = np.array([45, 92, 34, 34, 23])
        x = np.linalg.solve(a, b)
        assert linear.jacobi(a, b, (0,0,0)) == "Approximation does not converges"
    

    def test_gauss_seidel_converges(self):
        a = np.array([  
            [6, -2, 1], 
            [-2, 7, 2],
            [1, 2, -5]
        ])
        b = np.array([
            11,5,-1
        ])
        x = np.linalg.solve(a,b)
        approx_x = linear.gauss_seidel(a,b, (0,0,0), 1e-4, verbose = 0)
        
        assert linear.in_error_bounds(a, b, x, approx_x)
    
    def test_gauss_seidel_diverges(self):
        a = np.array([
                    [88, 67,  6, 96, 79],
                    [27, 60, 22, 80, 68],
                    [73, 16, 22, 48, 70],
                    [ 7, 19, 42,  8,  5],
                    [83,  2,  7, 82, 52]
                    ])

        b = np.array([45, 92, 34, 34, 23])
        x = np.linalg.solve(a, b)
        assert linear.gauss_seidel(a, b, (0,0,0)) == "Approximation does not converges"


class TestNonlinearIterative:
    def test_bisection(self):
        a = 1
        b = 2
        f = lambda x: (x*x) - 3
        e = 1e-5
        x = nonlinear.bisection(f, a, b, e=e)

        assert f(x)<=e, "Bisection method didn't work well, couldn't approximate the root."
        
