"""
since MICF has such slow IChol(0) factor time for a large fillin matrix.. I'm not gonna bother coding that one probably.
Just code the best algorithm VMICF which doens't require lots of temp storage for temp fillin that doesn't end up being stored (and just goes on diag)
"""

from _utils import get_beam_csr_mat_and_rhs, get_plate_csr_mat_and_rhs
from qordering import random_ordering, get_reordered_nofill_matrix, get_transpose_pattern, get_rows_from_rowp
from qordering import get_lower_triang_pattern
from qordering import csr_cholesky
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import argparse

parser = argparse.ArgumentParser()
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
L_rowp, L_cols = get_lower_triang_pattern(A0, strict_lower=False)
LT_rowp, LT_cols, _ = get_transpose_pattern(N, L_rowp, L_cols)
L_nnz = L_rowp[-1]
L_rows = get_rows_from_rowp(N, L_rowp)

# use csr sparsity pattern to do this..
# --------------------------------------------------

A_orig = A.copy()
A = A0.copy()
L = csr_cholesky(A)

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