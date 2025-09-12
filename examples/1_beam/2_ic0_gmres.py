# now let's test this out and visualize it
import numpy as np
from qordering import BeamFem
import scipy as sp
from qordering import random_ordering, get_reordered_nofill_matrix, get_LU_fill_matrix
from qordering import get_transpose_matrix, csr_cholesky, CholPrecond
from qordering import right_pgmres
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=False, help="Whether to do random ordering or not")
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
parser.add_argument("--fill", action=argparse.BooleanOptionalAction, default=False, help="Fillin the matrix for debugging")
parser.add_argument("--nxe", type=int, default=4, help="nxe # elements in x-dir")
args = parser.parse_args()

E = 2e7; b = 4e-3; L = 1; rho = 1
qmag, ys, rho_KS = 2e-2, 4e5, 50.0

nxh = args.nxe
hvec = np.array([1e-3] * nxh)

# create and assemble FEA problem
beam_fea = BeamFem(args.nxe, nxh, E, b, L, rho, qmag, ys, rho_KS, dense=False)
helem_vec = beam_fea.get_helem_vec(hvec)
beam_fea._compute_mat_vec(helem_vec)
mat, rhs = beam_fea.Kmat, beam_fea.force

# no reordering IC(0) right-PGMRES
# --------------------------------
# random_order = args.random

A = mat.copy()
N = mat.shape[0]
orig_rowp, orig_cols = A.indptr, A.indices
# if random_order:
#     _perm, _iperm = random_ordering(N, orig_rowp, orig_cols)
#     A = get_reordered_nofill_matrix(A, _perm, _iperm)
#     b = rhs[_iperm]
# else:
A = A
b = rhs.copy()

# TEMP DEBUG
if args.fill:
    A = get_LU_fill_matrix(A)

# do IC(0) cholesky factor
L = csr_cholesky(A)
LT = get_transpose_matrix(L)

# make preconditioner class
ic0_precond = CholPrecond(L, LT)

# solve FEA problem using right-PGMRES with IC0 precond
x = right_pgmres(A, b, x0=None, restart=50, max_iter=200, M=ic0_precond)

# compare to spsolve 
x_truth = sp.sparse.linalg.spsolve(A, b)

# error and resid
r = b - A @ x
e = x - x_truth
r_nrm, e_nrm = np.linalg.norm(r), np.linalg.norm(e)
print(f"{r_nrm=:.4e} {e_nrm=:.4e}")

# plot the solution we just got..
# if random_order:
#     soln = x[_perm]
# else:
soln = x

beam_fea.u = soln.copy()
beam_fea.plot_disp()
