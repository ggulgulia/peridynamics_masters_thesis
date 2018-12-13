import numpy as np
import numpy.linalg as la
import timeit as tm

def jacobi(A, b, TOL=1e-10):
    """
    solves the linear system Ax = b
    using gauss seidel method

    condition needed for convergence is diagonal dominance
    which is not guaranteed for peridynamics

    A: a square nxn matrix 
    b: vector of size n 
    TOL: requeired tolerance
    x: unknown vector

    """

    start = tm.default_timer()
    print("Starting linear solve using Jacobi Iteration")

    n = len(b)
    x_new = np.zeros(n, dtype=float)
    x_temp = np.ones(n, dtype=float)
    x_old = np.ones(n, dtype=float)
    iters = 0
    while (la.norm(abs(x_new - x_temp), np.inf) > TOL):
        iters += 1
        for i in range(n):
            x_new[i] = (b[i] - sum(A[i][0:i]*x_old[0:i]) - sum(A[i][i+1:n]*x_old[i+1:n]))/A[i][i]

        x_temp[:] = x_old[:]
        x_old[:] = x_new[:]

    end = tm.default_timer()
    print("Number of jacobi Solves: %i"%iters)
    print("Time taken to solve a linear system of %i equations with tolearance upto %4.7f is %4.3f seconds" %(n, TOL, end-start))
    return x_new



def gauss_seidel(A, b, TOL=1e-5):
    """
    solves the linear system Ax = b
    using gauss seidel method

    A: a square nxn matrix 
    b: vector of size n 
    TOL: requeired tolerance
    x: unknown vector

    """

    start = tm.default_timer()
    print("Starting linear solve using Gauss Seidel Iteration")

    n = len(b)
    x = np.zeros(n, dtype=float)
    x_old = np.ones(n, dtype=float)
    iters = 0
    while (la.norm(abs(x - x_old), np.inf)> TOL):
        iters += 1
        x_old[:] = x[:]
        for i in range(n):
            x[i] = (b[i] - sum(A[i][0:i]*x[0:i]) - sum(A[i][i+1:n]*x[i+1:n]))/A[i][i]

    end = tm.default_timer()
    print("Number of Gauss Seidel Solves: %i"%iters)
    print("Time taken to solve a linear system of %i equations with tolearance upto %4.7f is %4.3f seconds" %(n, TOL, end-start))
    return x
