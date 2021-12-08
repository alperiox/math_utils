from typing import ByteString
import numpy as np

from math_utils.utils import differentiate


def bisection(f, a, b, e=1e-5, verbose=True, i=0):
    '''
    Applies bisection method for nonlinear function approximation.
    In this method, we assume that f is a continuous function at I=[a,b]
    Therefore it is possible to use Intermediate Value Theorem or more specifically 
    Bolzano's Theorem to find the root of the function.
    Bolzano's Theorem says that if f(a)<0 and f(b)>0 or f(a)>0 and f(b)<0
    Then the function should have a root in interval [a,b]. It is easy to 
    conclude this result by using Intermediate Value Theorem, 
    one can take a look at the graph below:
    https://www.geogebra.org/calculator/enwxr5hp

    So we can derive some constraints on the function to check if 
    it has a root in given interval

    if f(a)f(b)<0 then we conclude that the function has a root in interval I
    if f(a)f(b)=0 then we can say that f(a) or f(b) is 0 since to multiplication is zero
    if f(a)f(b)>0 we cannot say that the function has a root in between a and b.

    In order to use this method, we will check the function values at the middle point
    of the interval. Let's call that middle point as m, if the function has a root in I
    then we can say that either [a,m] or [m,b] part of the interval has the root. So we check
    f(m)f(a) and f(m)f(b), then look at the interval that has the root again. It is simple as that.
    
    Of course this is an iterative method, so we need a constraint to stop. The error can be
    calculated as the distance from a boundary to calculated root (or middle point to be exact)
    So we can calculate the error as E = abs( (a-b) / 2 )

    '''
    error = abs((a-b)/2)
    m = (a+b)/2
    fa = f(a)
    fb = f(b)
    fm = f(m)

    if error < e: return m

    if verbose:
        print('[iteration %d] a=%.4f, b=%.4f, f(m)=%.4f, error=%f'%(i+1, a, b, fm, error))
        print('fa=%.4f, fb=%.4f'%(fa, fb))

    if fa*fb < 0:
        if fm*fa > 0:
            return bisection(f, m, b, verbose=verbose, i=i+1)
        elif fm*fa == 0:
            return m

        if fm*fb > 0:
            return bisection(f, a, m, verbose=verbose, i=i+1)
        elif fm*fb == 0:
            return m

def regula_falsi(f, a, b, e=1e-5, max_iter=None, verbose=True, i=0, prev_s = None):
    '''
    Applies regule-falsi method for nonlinear function approximation.
    In this method, we assume that f is a continuous function at I=[a,b]
    Therefore it is possible to use Intermediate Value Theorem or more specifically 
    Bolzano's Theorem to find the root of the function.
    Bolzano's Theorem says that if f(a)<0 and f(b)>0 or f(a)>0 and f(b)<0
    Then the function should have a root in interval [a,b]. It is easy to 
    conclude this result by using Intermediate Value Theorem, 
    one can take a look at the graph below:
    https://www.geogebra.org/calculator/enwxr5hp

    So we can derive some constraints on the function to check if 
    it has a root in given interval

    if f(a)f(b)<0 then we conclude that the function has a root in interval I
    if f(a)f(b)=0 then we can say that f(a) or f(b) is 0 since to multiplication is zero
    if f(a)f(b)>0 we cannot say that the function has a root in between a and b.

    In order to use this method, we will build a line that contains boundary points. Then use 
    that line's root to decrease the interval lenght, 
    then we can say that either [a,m] or [m,b] part of the interval has the root. So we check
    f(m)f(a) and f(m)f(b), then look at the interval that has the root again.
    
    Of course this is an iterative method, so we need a constraint to stop.
    We can select the error function as absolute error.

    '''
    if i==max_iter:
        return prev_s

    fa = f(a)
    fb = f(b)
    s = ((fb*a) - (fa*b))/(fb-fa)
    fs = f(s)

    # absolute error for f(x)-0:
    error = abs(fs)
    if verbose:
        print('[iteration %d] a=%.4f, b=%.4f, f(s)=%.4f, error=%f'%(i+1, a, b, fs, error))
        print('fa=%.4f, fb=%.4f'%(fa, fb))

    if error < e: return s

    if fa*fb < 0:
        if fs*fa > 0:
            return regula_falsi(f, s, b, max_iter=max_iter, verbose=verbose, i=i+1, prev_s=s)
        elif fs*fa == 0:
            return s

        if fs*fb > 0:
            return regula_falsi(f, a, s, max_iter=max_iter, verbose=verbose, i=i+1, prev_s=s)
        elif fs*fb == 0:
            return s
    else: 
        return False

def secant(f, a, b, e=1e-8, max_iter=None, verbose=True):
    if f(a) > f(b):
        x0 = b
        x1 = a
    else:
        x0 = a
        x1 = b

    i=0
    while True:
        xk = x1 - (f(x1)*(x1-x0))/(f(x1)-f(x0))
        error = abs(f(xk))
        
        if verbose:
            print('[iter %d] x_k-1 = %.4f, x_k = %.4f, x_k+1 = %.4f | absolute error=%.8f'%(i+1, x0, x1, xk, error))        

        x1, x0 = xk, x1

        if i+1==max_iter:
            return xk    
        i += 1

        if error < e:
            return xk

def fixed_point(f, g, x0, max_iter=1e10, verbose=True):
    '''
    Applies fixed point approximation for given function f
    '''
    result = x0
    prev_result = result
    try:
        i = 0
        while i < max_iter:
            if verbose:
                print(f'[iter {i}] xn={result}')


            prev_result = result
            result = g(result)

            if abs(differentiate(g,result)) >= 1:
                return False

            if prev_result==result:
                return result

            i += 1
         
        return result
    except Exception as e:
        return(e)

def muller(f, x0, x1, x2, i=0, max_iter=1e10, verbose=True, prev_result=None):
    h0, h1 = x1 - x0, x2 - x1
    d0, d1 = (f(x1)-f(x0)) / (x1 - x0 + 1e-5), (f(x2)-f(x1)) / (x2 - x1 + 1e-5)
    a = (d1 - d0) / (h1 + h0)
    b = (a*h1) + d1
    c = f(x2)
    sgn = b/abs(b)

    x3 = x2 - ( (2*c)/(b + (sgn*( (b*b) - (4*a*c) )**(1/2))) )
    x3 = float(x3.real) if type(x3)==complex else x3

    if verbose:
        print(f'[iter {i}] x0={x0:.4f}, x1={x1:.4f}, x2={x2:.4f} | x3={x3:.4f} | f(xn)={f(x3):.4f}')

    if i<max_iter:
        if (prev_result==x3) and (max_iter==1e10):
            return x3
        return muller(f, x1, x2, x3, i=i+1, max_iter=max_iter, verbose=verbose, prev_result=x3)
    else:
        return x3

a=1
b=2
f=lambda x: (x*x)-x-1
g=lambda x: 1 + 1/x
h=lambda x: x*x - 1 
k=lambda x: 1/(1+x)
l=lambda x: x**6 - 2
x=muller(l, .5, 1.5, 1, max_iter=7, verbose=True)
print(x, l(x))