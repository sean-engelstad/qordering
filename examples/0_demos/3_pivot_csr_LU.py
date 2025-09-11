from _utils import get_beam_csr_mat_and_rhs, get_plate_csr_mat_and_rhs
from qordering import compute_LU_fill_pattern, random_ordering, get_reordered_nofill_matrix, get_LU_fill_matrix
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import argparse

# we now try full column LU with pivoting

parser = argparse.ArgumentParser()
parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=False, help="Whether to do random ordering or not")
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

# TODO : interchange rows of A so |a_11| is largest in column 1

# now do LU factorization with algorithm 3.2 on page 36, a CSR column LU factorization with partial pivoting
# PA = LU (only row permutations..)
L, U = A_fill * 0.0, A_fill * 0.0
L[0,0] = 1.0
U[0,0] = A[0,0]
L[1:,0] = A_fill[1:,0] / A_fill[0,0]

perm = np.arange(0, N)
iperm = np.arange(0, N)

for j in range(1, N): # outer steps of LU factor

    # solve for column j of U with L[:j,:j] * U[:j,j] = A[:j, j] with triangular solve
    # based on forward triang solve from Algorithm 3.3 on page 41
    # ------------------------------------------------------------

    # initialize y = b
    for r in range(j):
        U[r,j] = A[iperm[r], j] # row swaps applied here..
    
    # now loop through
    for r in range(j):
        U[r,j] /= L[r,r]
        pr = iperm[r] # permutation
        for ip in range(rowp[pr], rowp[pr+1]): # l_ir neq 0, fine to do this l_ri neq 0 because sym
            i = cols[ip]
            if r < i and i < j:
                # y_i -= l_ir * y_r step here
                U[i,j] -= L[i,r] * U[r,j]

    # get z vector and do pivoting
    # ----------------------------

    # get z column vector (for analyzing next pivot)
    z = A[j:, j] - 



