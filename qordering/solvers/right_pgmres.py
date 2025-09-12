import numpy as np
import scipy as sp

def right_pgmres(A, b, x0=None, restart=50, tol=1e-8, max_iter=1000, M=None):
    """
    Right-precond Modified Gram-Schmidt GMRES
    
    Parameters:
    A : callable or sparse matrix
        Function A(x) or sparse matrix representing the linear system.
    b : ndarray
        Right-hand side vector.
    x0 : ndarray, optional
        Initial guess (default: zero vector).
    tol : float, optional
        Convergence tolerance.
    max_iter : int, optional
        Maximum number of iterations.
    restart : int, optional
        Restart parameter (default: 50).
    M : callable or sparse matrix, optional
        Preconditioner (default: None).
    
    Returns:
    x : ndarray
        Approximate solution.
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    
    if M is None:
        M_inv = lambda x: x
    else:
        # M_inv = spla.LinearOperator((n, n), matvec=lambda x: spla.spsolve(M, x))
        M_inv = lambda x : M.solve(x)

    x = x0.copy()
    r = b - (A @ x)
    beta = np.linalg.norm(r)
    # print(f"{beta=}")
    
    if beta < tol:
        return x

    for _ in range(max_iter // restart):
        V = np.zeros((n, restart+1))
        H = np.zeros((restart+1, restart))
        g = np.zeros(restart+1)
        cs = np.zeros(restart)
        ss = np.zeros(restart)

        # Start Arnoldi process
        V[:, 0] = r / beta
        g[0] = beta
        # print(f"{g[0]=}")

        print(f"GMRES : outer iter {_} resid {beta=}")

        for j in range(restart):
            z = M_inv(V[:,j])
            w = A @ z

            # Gram-Schmidt with reorthogonalization
            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i], w)
                w -= H[i, j] * V[:, i]

            H[j+1, j] = np.linalg.norm(w)
            if H[j+1, j] == 0:
                print("H break")
                break
            V[:, j+1] = w / H[j+1, j]

            # givens rotations 
            # ----------------

            for i in range(j):
                temp = H[i,j]
                H[i,j] = cs[i] * H[i,j] + ss[i] * H[i+1,j]
                H[i+1,j] = -ss[i] * temp + cs[i] * H[i+1,j]

            cs[j] = H[j,j] / np.sqrt(H[j,j]**2 + H[j+1,j]**2)
            ss[j] = cs[j] * H[j+1,j] / H[j,j]

            g_temp = g[j]
            g[j] *= cs[j]
            g[j+1] = -ss[j] * g_temp

            H[j,j] = cs[j] * H[j,j] + ss[j] * H[j+1,j]
            H[j+1,j] = 0.0

            if (j % 10 == 0): print(f"GMRES [{j}] : {g[j+1]=}")

            # Check convergence
            if abs(g[j+1]) < tol:
                print(f"g break at iteration {j}")
                break

        # Solve the upper triangular system H * y = g
        y = np.linalg.solve(H[:j+1, :j+1], g[:j+1])

        # Update solution
        dz = V[:, :j+1] @ y
        dx = M_inv(dz)
        x += dx

        # Compute new residual
        r = b - (A @ x)
        beta = np.linalg.norm(r)
        if beta < tol:
            break
    
    print(f"GMRES final resid {beta=}")
    return x
