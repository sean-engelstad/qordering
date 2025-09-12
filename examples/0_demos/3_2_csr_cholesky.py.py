from _utils import get_beam_csr_mat_and_rhs, get_plate_csr_mat_and_rhs
from qordering import random_ordering, get_reordered_nofill_matrix, get_transpose_pattern, get_rows_from_rowp
from qordering import get_elim_tree, get_L_fill_pattern, get_LU_fill_matrix, get_lower_triang_pattern
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import argparse

# we now try full column LU with pivoting

parser = argparse.ArgumentParser()
parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=False, help="Whether to do random ordering or not")
parser.add_argument("--rem_bcs", action=argparse.BooleanOptionalAction, default=False, help="remove bcs or not from matrix")
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
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

# get L fill pattern
rowp, cols = A0.indptr, A0.indices
# parent, ancestor = get_elim_tree(N, rowp, cols)
# L_rowp, L_cols = get_L_fill_pattern(N, rowp, cols, parent, ancestor, strict_lower=False)
# if you want full fillin, but with this way you can now do no fill here too..
A_fill = get_LU_fill_matrix(A0)
L_rowp, L_cols = get_lower_triang_pattern(A_fill, strict_lower=False)
LT_rowp, LT_cols, _ = get_transpose_pattern(N, L_rowp, L_cols)
L_nnz = L_rowp[-1]
L_rows = get_rows_from_rowp(N, L_rowp)

# use csr sparsity pattern to do this..
# --------------------------------------------------

A_orig = A.copy()
A = A0.copy()
L_vals = np.zeros(L_nnz, dtype=np.double)
L = sp.sparse.csr_matrix((L_vals, (L_rows, L_cols)), shape=(N, N))

# # DEBUG to check U_rowp, U_cols correct
debug = False
# debug = True
if debug:
    L_vals[:] = 1.0
    for i in range(N):
        for jp in range(L_rowp[i], L_rowp[i+1]):
            j = L_cols[jp]
            L[i,j] = 1.0
    U_rows = get_rows_from_rowp(N, LT_rowp)
    U = sp.sparse.csr_matrix((L_vals, (U_rows, LT_cols)), shape=(N, N))
    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    ax[0].spy(L)
    ax[1].spy(U)
    # ax[1].spy(true_L)
    plt.show()
    L *= 0.0 # change back to zeros

# dense version of csr cholesky (but I use only csr rowpattern to do this..)
# algorithm 5.1 on page 74 (in-place dense left-looking cholesky)
print(f"begin CSR cholesky factorization")
for j in range(N):
    # use symmetry here.. this reads down a col is like reading down U sparsity or LT sparsity
    for rp in range(LT_rowp[j], LT_rowp[j+1]):
        r = LT_cols[rp]
        L[r,j] = A[r,j]

    for kp in range(L_rowp[j], L_rowp[j+1]):
        k = L_cols[kp]
        if k >= j: continue # comes from for k in range(j) requirement

        for rp in range(LT_rowp[j], LT_rowp[j+1]):
            r = LT_cols[rp]
            L[r,j] -= L[r,k] * L[j,k]
    
    L[j,j] = np.sqrt(L[j,j])
    for rp in range(LT_rowp[j], LT_rowp[j+1]):
        r = LT_cols[rp]
        if r < j+1: continue # supposed to be j+1 to N so slightly larger
        L[r,j] /= L[j,j]

    # compute residual (DEBUG)
    # R = A - L @ L.T
    # R_nrm = np.linalg.norm(R.toarray()[:j,:j])
    # print(f"{R_nrm=}")

    # # show L, U and A
    # if args.plot and abs(R_nrm) > 1e-5:
    #     fig, ax = plt.subplots(2,2, figsize=(12, 8))
    #     ax[0,0].imshow(L.toarray()[:,:j+1])
    #     # ax[0,1].imshow(true_L.toarray()[:,:j])
    #     ax[0,1].imshow(L.toarray()[:,:j+1].T)
    #     ax[1,0].imshow(A.toarray()[:,:j+1])
    #     ax[1,1].imshow(R.toarray()[:j+1,:j+1])
    #     plt.show()
    #     exit()

print("done with csr cholesky factor")

# now check the accuracy of the cholesky factor here..
R = A - L @ L.T
R_nrm = np.linalg.norm(R.toarray())
print(f"{R_nrm=}")

# # show L, U and A
if args.plot:
    fig, ax = plt.subplots(2,2, figsize=(12, 8))
    ax[0,0].imshow(L.toarray())
    ax[0,1].imshow(L.toarray().T)
    ax[1,0].imshow(A.toarray())
    ax[1,1].imshow(R.toarray())
    plt.show()
    exit()