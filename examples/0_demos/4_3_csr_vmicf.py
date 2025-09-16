"""
since MICF has such slow IChol(0) factor time for a large fillin matrix.. I'm not gonna bother coding that one probably.
Just code the best algorithm VMICF which doens't require lots of temp storage for temp fillin that doesn't end up being stored (and just goes on diag)
"""

from _utils import get_beam_csr_mat_and_rhs, get_plate_csr_mat_and_rhs
from qordering import random_ordering, get_reordered_nofill_matrix, get_transpose_pattern, get_rows_from_rowp
from qordering import get_elim_tree, get_L_fill_pattern, get_fillin_only_pattern, get_upper_triang_values
from qordering import get_upper_triang_pattern, vmicf_cholesky
from qordering import csr_cholesky, get_diag_rowp
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=False, help="Whether to do random ordering or not")
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
parser.add_argument("--nxe", type=int, default=4, help="nxe # elements in x-dir")
parser.add_argument("--case", type=str, default="simple-mat", help="options: [beam, plate]")
parser.add_argument("--not_mmat", action=argparse.BooleanOptionalAction, default=False, help="Make not an M-matrix")
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

# compute permutation map, perm sends to permuted nodes, iperm back to unpermuted (standard order)
if args.random:
    _perm, _iperm = random_ordering(A.shape[0], A.indptr, A.indices)
    A0 = get_reordered_nofill_matrix(A, _perm, _iperm)
else:
    A0 = A

# copy the matrix
A0 = A.copy()
# it's a 2DOF per node or Bsr2 matrix stored as CSR
N = A.shape[0]
orig_rowp, orig_cols = A.indptr, A.indices

# vmicf cholesky factorization
# ----------------------------

L, D, U = vmicf_cholesky(A0, make_not_m=args.not_mmat)

fig, ax = plt.subplots(1, 3, figsize=(10, 7))
ax[0].imshow(L.toarray())
ax[1].imshow(D)
ax[2].imshow(U.toarray())
plt.show()

U_np = U.toarray()
R = U_np.T @ D @ U_np - A0.toarray()
R_nrm = np.linalg.norm(R)
print(f"{R_nrm=}")

plt.imshow(R)
plt.show()