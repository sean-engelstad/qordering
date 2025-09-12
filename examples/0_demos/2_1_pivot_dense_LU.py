from _utils import get_beam_csr_mat_and_rhs, get_plate_csr_mat_and_rhs
from qordering import random_ordering, get_reordered_nofill_matrix, get_LU_fill_matrix
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import argparse

# we now try full column LU with pivoting

parser = argparse.ArgumentParser()
parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=False, help="Whether to do random ordering or not")
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
parser.add_argument("--rem_bcs", action=argparse.BooleanOptionalAction, default=False, help="remove bcs or not from matrix")
parser.add_argument("--nxe", type=int, default=4, help="nxe # elements in x-dir")
parser.add_argument("--case", type=str, default="beam", help="options: [beam, plate]")
args = parser.parse_args()

# see if random order stabilize LU factor... will add fillin though..
random_order = args.random

if (args.case == "beam"):
    A, rhs = get_beam_csr_mat_and_rhs(args.nxe, csr=True)

elif (args.case == "plate"):
    A, rhs = get_plate_csr_mat_and_rhs(args.nxe, csr=True)

# copy the matrix
A0 = A.copy()
# it's a 2DOF per node or Bsr2 matrix stored as CSR
N = A.shape[0]
orig_rowp, orig_cols = A.indptr, A.indices

# compute permutation map, perm sends to permuted nodes, iperm back to unpermuted (standard order)
if random_order:
    _perm, _iperm = random_ordering(N, orig_rowp, orig_cols)
    A0 = get_reordered_nofill_matrix(A, _perm, _iperm)
else:
    A0 = A0

# perform the LU fillin which copies values
A_fill = get_LU_fill_matrix(A0)
rowp, cols = A_fill.indptr, A_fill.indices

# use dense matrix in computations first (prototype)
# --------------------------------------------------

A_orig = A.copy()

A = A_fill.toarray()

# try just submatrix here..
# N = 8
# A = A[:N,:N]

L, U = np.zeros_like(A), np.zeros_like(A)

# swap rows of A for column 0 pivot
ind = np.argmax(np.abs(A[:,0]))
if ind != 0:
    temp = A[0,:].copy()
    A[0,:] = A[ind,:].copy()
    A[ind,:] = temp.copy()

L[0,0] = 1.0
U[0,0] = A[0,0]
L[1:,0] = A[1:,0] / A[0,0]

for j in range(1,N):
    U[:j,j] = sp.linalg.solve_triangular(L[:j,:j], A[:j,j], lower=True)
    z = A[j:,j] - L[j:,:j] @ U[:j,j]
    z_ind = np.argmax(np.abs(z))
    ind = z_ind + j
    z0 = z.copy()

    # swapping rows for A, L and z (j and on only)
    temp = A[j,:].copy()
    A[j,:] = A[ind,:].copy()
    A[ind,:] = temp.copy()
    temp = z[0]
    z[0] = z[z_ind]
    z[z_ind] = temp
    temp = L[j,:].copy()
    L[j,:] = L[ind,:].copy()
    L[ind,:] = temp.copy()

    L[j,j] = 1.0
    U[j,j] = z[0]
    L[(j+1):,j] = z[1:] / z[0]

    # print(f"{z0=}\n{z=}")
    # plt.imshow(A)
    # plt.show()

    # show L, U and A
    # fig, ax = plt.subplots(2,2, figsize=(12, 8))
    # ax[0,0].imshow(L[:j,:j])
    # ax[0,1].imshow(U[:j,:j])
    # ax[1,0].imshow(A[:j,:j])
    # R = A - L @ U
    # ax[1,1].imshow(R[:j,:j])
    # plt.show()


# compute LU factor residual
R = A - L @ U
R_nrm = np.linalg.norm(R)
print(f"LU factor resid nrm = {R_nrm:.4e}")

# # show L, U and A
if args.plot:
    fig, ax = plt.subplots(2,2, figsize=(12, 8))
    ax[0,0].imshow(L)
    ax[0,1].imshow(U)
    ax[1,0].imshow(A)
    ax[1,1].imshow(R)
    plt.show()
    # exit()

# now try computing LU solve
if not args.random:
    true_soln = sp.sparse.linalg.spsolve(A_orig, rhs)

    y = sp.linalg.solve_triangular(L, rhs, lower=True)
    pred_soln = sp.linalg.solve_triangular(U, y, lower=False)

    soln_err = np.linalg.norm(true_soln - pred_soln)
    print(f"{soln_err=:.4e}")

if args.random:
    true_soln = sp.sparse.linalg.spsolve(A_orig, rhs)
    rhs_perm = rhs[_iperm]
    perm_soln1 = sp.sparse.linalg.spsolve(A_fill, rhs_perm)

    y = sp.linalg.solve_triangular(L, rhs_perm, lower=True)
    pred_soln_perm = sp.linalg.solve_triangular(U, y, lower=False)
    pred_soln = pred_soln_perm[_perm]

    # soln_err = np.linalg.norm(true_soln - pred_soln)
    soln_err = np.linalg.norm(true_soln - perm_soln1)
    print(f"{soln_err=:.4e}")


