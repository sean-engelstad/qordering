"""
since MICF has such slow IChol(0) factor time for a large fillin matrix.. I'm not gonna bother coding that one probably.
Just code the best algorithm VMICF which doens't require lots of temp storage for temp fillin that doesn't end up being stored (and just goes on diag)
"""

from _utils import get_beam_csr_mat_and_rhs, get_plate_csr_mat_and_rhs
from qordering import random_ordering, get_reordered_nofill_matrix, get_transpose_pattern, get_rows_from_rowp
from qordering import get_elim_tree, get_L_fill_pattern, get_fillin_only_pattern
from qordering import get_lower_triang_pattern
from qordering import csr_cholesky, get_diag_rowp
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
parser.add_argument("--nxe", type=int, default=4, help="nxe # elements in x-dir")
parser.add_argument("--case", type=str, default="simple-mat", help="options: [beam, plate]")
args = parser.parse_args()
if (args.case == "beam"):
    A, rhs = get_beam_csr_mat_and_rhs(args.nxe, csr=True)

elif (args.case == "plate"):
    A, rhs = get_plate_csr_mat_and_rhs(args.nxe, csr=True)

elif args.case == "simple-mat":
    A = np.array([
        [4, 0, -1, -1],
        [0, 2, 1, -1],
        [-1, 1, 2, 0],
        [-1, -1, 0, 2]
    ], dtype=np.double)
    A = sp.sparse.csr_matrix(A)

# copy the matrix
A0 = A.copy()
# it's a 2DOF per node or Bsr2 matrix stored as CSR
N = A.shape[0]
orig_rowp, orig_cols = A.indptr, A.indices

# compute permutation map, perm sends to permuted nodes, iperm back to unpermuted (standard order)
A0 = A0

# get L fill pattern
rowp, cols = A0.indptr, A0.indices
parent, ancestor = get_elim_tree(N, rowp, cols)
L_fill_rowp, L_fill_cols = get_L_fill_pattern(N, rowp, cols, parent, ancestor, strict_lower=False)
L_rowp, L_cols = get_lower_triang_pattern(A0, strict_lower=False)
# don't think I need transpose patterns.. aren't they the same cause sym matrix? yeah.. less int storage
LT_rowp, LT_cols, LT_map = get_transpose_pattern(N, L_rowp, L_cols, get_map=True)
LT_fill_rowp, LT_fill_cols, LT_fill_map = get_transpose_pattern(N, L_fill_rowp, L_fill_cols, get_map=True)
L_nnz = L_rowp[-1]
L_rows = get_rows_from_rowp(N, L_rowp)
A0_vals = A0.data

# print(f"{L_fill_rowp=} {L_fill_cols=}")

# separately compute fillin rowp, cols only
L_fillin_rowp, L_fillin_cols = get_fillin_only_pattern(N, L_rowp, L_cols, L_fill_rowp, L_fill_cols)
LT_fillin_rowp, LT_fillin_cols, LT_fillin_map = get_transpose_pattern(N, L_fillin_rowp, L_fillin_cols, get_map=True)

# print(f"{L_fillin_rowp=} {L_fillin_cols=}")
# print(f"{LT_rowp=} {LT_cols=} {LT_fillin_map=}")


# L_diag_rowp = get_diag_rowp(N, L_rowp, L_cols) # don't need this for L

# get L_vals copied from A (including diag)
L_vals = np.zeros(L_nnz, dtype=np.double)
next = L_rowp[:-1].copy()
for i in range(N):
    for jp in range(rowp[i], rowp[i+1]):
        j = cols[i]
        L_vals[next[i]] = A0_vals[jp]
        next[i] += 1

L_rows = get_rows_from_rowp(N, L_rowp)
L0 = sp.sparse.csr_matrix((L_vals, (L_rows, L_cols)), shape=(N, N))

# plt.imshow(L0.toarray())
# plt.show()

# use csr sparsity pattern to do this..
# --------------------------------------------------

A_orig = A.copy()
A = A0.copy()

# appears I actually only need the LT?
# print(f"{LT_rowp=} {LT_cols=} {LT_fillin_map=}")

# based off of 4_2_dense_vmicf.py (but now with sparsity patterns)
# TODO : may actually be more convenient + efficient to form U not L then? (form in LT?)
d = np.zeros(N, dtype=np.double)
for i in range(N):
    ip_diag = LT_map[LT_rowp[i]]
    d[i] = 1.0 / L_vals[ip_diag]
    d[i+1:] = 0.0 # otherwise no fill not treated correctly?
    for jp in range(LT_rowp[i]+1, LT_rowp[i+1]):
        j = LT_cols[jp]
        jp = LT_map[jp]
        # A[j,i] * d[i] => d[j]
        d[j] = L_vals[jp] * d[i] # j > i
    # print(f"{d=}")

    # fillin only (zero pattern set or not the nofill entries)
    # LT fillin (transpose to read down cols sparsity)..
    print(f"{LT_fillin_rowp=} {LT_fillin_cols=}")
    for jp in range(LT_rowp[i] + 1, LT_rowp[i+1]):
        j = LT_cols[jp]
        jp = LT_map[jp]
        # only [k,j] needs to be fillin
        for kp in range(LT_fillin_rowp[j], LT_fillin_rowp[j+1]):
            k = LT_fillin_cols[kp]
            # kp = LT_fillin_map[kp]
            # A[k,j] = -d[k] * A[j,i]
            # print(f"{i=}{j=} {L_vals[jp]=:.2e} equiv to A[j,i]")
            A_kj = -d[k] * L_vals[jp]
            # add into A[j,j] and A[k,k]
            Ajj_p = LT_map[LT_rowp[j]]
            Akk_p = LT_map[LT_rowp[k]]
            # print(f"A[{k=},{j=}] Z spot => inc by {np.abs(A_kj):.2e}")

            L_vals[Ajj_p] += np.abs(A_kj)
            L_vals[Akk_p] += np.abs(A_kj)
    
    # print(f"{L_vals=}")

    # nofill entries (or regular nofill pattern now)
    for jp in range(LT_rowp[i] + 1, LT_rowp[i+1]): # starts past diag here..
        j = LT_cols[jp]
        jp = LT_map[jp]

        for kp in range(LT_rowp[j], LT_rowp[j+1]):
            k = LT_cols[kp]
            kp = LT_map[kp]

            val = d[k] * L_vals[jp]
            # print(f"A[{k=},{j=}] dec by {val=}")

            # A[k,j] -= d[k] * A[j,i]
            L_vals[kp] -= d[k] * L_vals[jp]
            
    # print(f"{L_vals=}")
        
# now debugging here..
# print(f"{d=}")
L = sp.sparse.csr_matrix((L_vals, (L_rows, L_cols)), shape=(N, N))
L_np = L.toarray()
D = np.diag(d)
R = A0.toarray() - L_np @ D @ L_np.T

plt.imshow(R)
plt.show()


# L = csr_cholesky(A)

# print("done with csr cholesky factor")

# # now check the accuracy of the cholesky factor here..
# R = A - L @ L.T
# R_nrm = np.linalg.norm(R.toarray())
# print(f"{R_nrm=}")

# # # show L, U and A
# if args.plot:
#     fig, ax = plt.subplots(2,2, figsize=(12, 8))
#     ax[0,0].imshow(L.toarray())
#     ax[0,1].imshow(L.toarray().T)
#     ax[1,0].imshow(A.toarray())
#     ax[1,1].imshow(R.toarray())
#     plt.show()
#     exit()