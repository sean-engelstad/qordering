"""
since MICF has such slow IChol(0) factor time for a large fillin matrix.. I'm not gonna bother coding that one probably.
Just code the best algorithm VMICF which doens't require lots of temp storage for temp fillin that doesn't end up being stored (and just goes on diag)
"""

from _utils import get_beam_csr_mat_and_rhs, get_plate_csr_mat_and_rhs
from qordering import random_ordering, get_reordered_nofill_matrix, get_transpose_pattern, get_rows_from_rowp
from qordering import get_elim_tree, get_L_fill_pattern, get_fillin_only_pattern, get_upper_triang_values
from qordering import get_upper_triang_pattern
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
U_fill_rowp, U_fill_cols, _ = get_transpose_pattern(N, L_fill_rowp, L_fill_cols)
U_rowp, U_cols = get_upper_triang_pattern(A0, strict_upper=False)
# print(f"{U_rowp=} {U_cols=}")
U_nnz = U_rowp[-1]
U_rows = get_rows_from_rowp(N, U_rowp)
U_fillin_rowp, U_fillin_cols = get_fillin_only_pattern(N, U_rowp, U_cols, U_fill_rowp, U_fill_cols)
# print(f"{U_fillin_rowp=} {U_fillin_cols=}")
U_vals = get_upper_triang_values(N, rowp, cols, A0.data, U_rowp, U_cols)
U_rows = get_rows_from_rowp(N, U_rowp)
U0 = sp.sparse.csr_matrix((U_vals, (U_rows, U_cols)), shape=(N, N))

plt.imshow(U0.toarray())
plt.show()

# use csr sparsity pattern to do this..
# --------------------------------------------------

A_orig = A.copy()
A = A0.copy()

# appears I actually only need the LT?
# print(f"{LT_rowp=} {LT_cols=} {LT_fillin_map=}")

# based off of 4_2_dense_vmicf.py (but now with sparsity patterns)
# wrote it in terms of L first, much more efficient to write in terms of U aka UT because reads down cols
# NOTE : old L_vals and LT_rowp, LT_cols, LT_map (less efficient version in archive/ of 0_demos/)
d = np.zeros(N, dtype=np.double)
for i in range(N):
    ip_diag = U_rowp[i]
    d[i] = 1.0 / U_vals[ip_diag]
    d[i+1:] = 0.0 # not sure if this part is necessary tbh

    for jp in range(U_rowp[i]+1, U_rowp[i+1]):
        j = U_cols[jp]
        d[j] = U_vals[jp] * d[i]

    # fillin only (zero pattern set or not the nofill entries)
    # LT fillin (transpose to read down cols sparsity)..
    for jp in range(U_rowp[i]+1, U_rowp[i+1]):
        j = U_cols[jp]
        # A[k,j] needs to be fillin values (all other sparsity are regular nofill)
        for kp in range(U_fillin_rowp[j], U_fillin_rowp[j+1]):
            k = U_fillin_cols[kp]
            A_kj = -d[k] * U_vals[jp]
            jp_diag = U_rowp[j] # cause in U pattern, diag is always first entry in each row
            kp_diag = U_rowp[k]
            U_vals[jp_diag] += np.abs(A_kj)
            U_vals[kp_diag] += np.abs(A_kj)

    # nofill entries (or regular nofill pattern now)
    for jp in range(U_rowp[i] + 1, U_rowp[i+1]):
        j = U_cols[jp]
        for kp in range(U_rowp[j], U_rowp[j+1]):
            k = U_cols[kp]
            U_vals[kp] -= d[k] * U_vals[jp]
        
# now debugging here..
# print(f"{d=}")
U = sp.sparse.csr_matrix((U_vals, (U_rows, U_cols)), shape=(N, N))
U_np = U.toarray()
D = np.diag(d)
R = U_np.T @ D @ U_np - A0.toarray()
R_nrm = np.linalg.norm(R)
print(f"{R_nrm=}")

plt.imshow(R)
plt.show()