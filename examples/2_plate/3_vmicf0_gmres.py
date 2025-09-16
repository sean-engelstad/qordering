# now let's test this out and visualize it
import numpy as np
from qordering import PlateAssembler
import scipy as sp
from qordering import random_ordering, get_reordered_nofill_matrix, get_LU_fill_matrix
from qordering import get_transpose_matrix, vmicf_cholesky, CholPrecond
from qordering import right_pgmres
import argparse

"""
this VMICF thing doesn't actually help convergence..
"""

parser = argparse.ArgumentParser()
# parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=False, help="Whether to do random ordering or not")
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
parser.add_argument("--fill", action=argparse.BooleanOptionalAction, default=False, help="Fillin the matrix for debugging")
parser.add_argument("--noprec", action=argparse.BooleanOptionalAction, default=False, help="remove preconditioner in GMRES")
parser.add_argument("--not_mmat", action=argparse.BooleanOptionalAction, default=False, help="Make not an M-matrix") # this is kind of dumb in their paper.. (see paper from 4_3_csr_vmicf.py, only conv fast when not M mat, when off-diag are positive?)
parser.add_argument("--nxe", type=int, default=4, help="nxe # elements in x-dir")
args = parser.parse_args()

# create and assemble FEA problem
nxe, nxc = args.nxe, 1
plate_fea = PlateAssembler.aluminum_unitsquare_trigload(
    num_elements=nxe**2,
    num_components=nxc**2,
    rho_KS=200.0,
    qmag=2e-2, 
    can_print=False
)
ncomp = plate_fea.ncomp
hred = np.array([5e-3] * ncomp)
helem_vec = plate_fea.get_helem_vec(hred)
plate_fea._compute_mat_vec(helem_vec)
mat, rhs = plate_fea.Kmat, plate_fea.force

# remove_bcs = True
# if remove_bcs:
#     mat = mat.toarray()
#     mat, rhs = remove_bcs(mat, rhs)
#     if csr:
#         mat = sp.sparse.linalg.csr_matrix(mat)

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

# do IC(0) cholesky factor, VMICF or variational version
L, D, U = vmicf_cholesky(A, make_not_m=args.not_mmat)

# make preconditioner class
ic0_precond = CholPrecond(L, U, D=D)

# compute precond error..
L_np = L.toarray()
R = A - L_np @ D @ L_np.T
factor_resid_nrm = np.linalg.norm(R)
print(f"{factor_resid_nrm=:.2e}")

# solve FEA problem using right-PGMRES with IC0 precond
x = right_pgmres(A, b, x0=None, restart=100, max_iter=200, M=ic0_precond if not(args.noprec) else None)

# compare to spsolve 
x_truth = sp.sparse.linalg.spsolve(A, b)

# error and resid
r = b - A @ x
e = x - x_truth
r_nrm, e_nrm = np.max(np.abs(r)), np.max(np.abs(e))
print(f"{r_nrm=:.4e} {e_nrm=:.4e}")

# plot the solution we just got..
# if random_order:
#     soln = x[_perm]
# else:
soln = x

if args.plot:
    plate_fea.u = soln.copy()
    plate_fea.plot_disp()
