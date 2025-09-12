from _utils import get_beam_csr_mat_and_rhs, get_plate_csr_mat_and_rhs
from qordering import compute_LU_fill_pattern, random_ordering, get_reordered_nofill_matrix
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import argparse

# many steps taken from this book, [Algorithms for Sparse Linear Systems](https://link.springer.com/book/10.1007/978-3-031-25820-6) book
# page 51 gives dense matrix gauss-elim process illustrated in prev example 1_dense_gauss_elim.py
# here I extend to sparse Gauss-elim for symmetric matrix (so I only need elimination tree)

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

# now show the LU factorization using Gaussian elimination..
# A will become U in-place now..
N = A.shape[0]
orig_rowp, orig_cols = A.indptr, A.indices

# compute permutation map, perm sends to permuted nodes, iperm back to unpermuted (standard order)
if random_order:
    perm, iperm = random_ordering(N, orig_rowp, orig_cols)
    A0 = get_reordered_nofill_matrix(A, perm, iperm)
    rowp, cols = A0.indptr, A0.indices
    nnz = rowp[-1]

    # temp check.. see if reordered solve matches orig solve, yes they match
    check = False
    # check = True
    if check:
        rhs_perm = rhs[iperm]
        soln1 = sp.sparse.linalg.spsolve(A, rhs)
        soln2_perm = sp.sparse.linalg.spsolve(A0, rhs_perm)
        soln2 = soln2_perm[perm]
        print(f"{soln1=}")
        print(f"{soln2=}")
        diff_nrm = np.linalg.norm(soln1 - soln2)
        diff_ne_nrm = np.linalg.norm(soln1 - soln2_perm)
        print(f"{diff_nrm=:.2e} {diff_ne_nrm=:.2e}")
        exit()

else:
    rowp, cols = orig_rowp, orig_cols
    A0 = A0


fill_rowp, fill_rows, fill_cols = compute_LU_fill_pattern(N, rowp, cols)
fill_nnz = fill_rowp[-1]

def convert_rowp_cols_to_ones_mat(my_rowp, my_cols):
    # turn rowp into rows
    nnz = my_cols.shape[0]
    rows = np.zeros(nnz, dtype=np.int32)
    for i in range(my_rowp.shape[0] - 1):
        for jp in range(my_rowp[i], my_rowp[i+1]):
            rows[jp] = i
    _vals = np.ones(nnz, dtype=np.double)
    _mat = sp.sparse.csr_matrix((_vals, (rows, my_cols)), shape=(N,N))
    return _mat

ones_nofill = convert_rowp_cols_to_ones_mat(rowp, cols)
# ones_L_fill = convert_rowp_cols_to_ones_mat(fill_L_rowp, fill_L_cols)
ones_fill = convert_rowp_cols_to_ones_mat(fill_rowp, fill_cols)


# now make matrices for A0, L and U with fillin
# ---------------------------------------------

fill_vals = np.zeros(fill_nnz, dtype=np.double)
A0_fill = sp.sparse.csr_matrix((fill_vals, (fill_rows, fill_cols)), shape=(N, N))

# copy values from nofill sparsity
for i in range(N):
    for jp in range(rowp[i], rowp[i+1]):
        j = cols[jp]
        A0_fill[i,j] = A0[i,j]

# plot the fillin matrix.. (check)
# fig, ax = plt.subplots(1, 2, figsize=(10, 7))
# ax[0].imshow(A0.toarray())
# ax[1].imshow(A0_fill.toarray())
# plt.show()


# now do sparse gauss-elim on the full LU fillin pattern..
# --------------------------------------------------------

# hold L and U factors in-place in M matrix (with fillin form)
M = A0_fill.copy()

# in ABOVE, since we had sym matrix, cholesky fill pattern same as LU fill pattern with no pivoting, though LU vs LL^T factorizations
# would be different.. this is an LU factor here..
# though I can also look at Cholesky BSR factorizations in a minute for Q-order (note this is treated as CSR here even though
# it's a BSR)


for k in range(N-1): # step k of zeroing cols
    # part 1, compute L factor
    # ------------------------

    # dense version was, 
    # for i in range(k+1, N):
        # L[i,k] = A[i,k] / A[k,k]

    # use sparsity of row k for col k in L_ik nz since sym
    for ip in range(fill_rowp[k], fill_rowp[k+1]):
        i = fill_cols[ip]
        if not(i > k): continue # want only L_ik from i in [k+1, N)
        M[i,k] /= M[k,k]

    # part 2, compute U factor
    # ------------------------
        # dense version was, (inside other for loop)
        # for j in range(k, N):
            # A[i,j] -= L[i,k] * A[k,j]

    # sparse version, we need ij and kj nz, but based on fillin definitions
    # ik and kj equiv to ij nz, so don't need to check that also I think..
    # fine to stay inside previous for loop, just with a_kj nz sparsity loop
        for jp in range(fill_rowp[k], fill_rowp[k+1]):
            j = fill_cols[jp]
            if not(j >= k): continue
            M[i,j] -= M[i,k] * M[k,j]

    # DEBUG
    # plt.imshow(M.toarray()[:5,:5])
    # plt.show()


# now split out L and U matrices.. for debugging purposes..
L, U = M.copy(), M.copy()
# change L to lower triang with 1s on diag and U to upper triang
for i in range(N):
    for jp in range(fill_rowp[i], fill_rowp[i+1]):
        j = fill_cols[jp]
        
        if (i == j): L[i,j] = 1.0
        if (i < j): L[i,j] = 0.0
        if (i > j): U[i,j] = 0.0

# now compute the LU factor error (to double check this is correct)
R = A0_fill.copy()
# have to do the mat-mult and subtract myself apparently..
for i in range(N):
    for jp in range(fill_rowp[i], fill_rowp[i+1]):
        j = fill_cols[jp]
        for kp in range(fill_rowp[i], fill_rowp[i+1]):
            k = fill_cols[kp]
            R[i,j] -= L[i,k] * U[k,j]
        # yeah only the lower triang ends up having error.. hmm
        # print(f"R[{i=},{j=}]={R[i,j]:.2e}")
# R = A0_fill - L @ U # this doens't have right sparsity.. ?
R_nrm = np.max(R.toarray())
A_nrm = np.max(A0.toarray())
print(f"LU factor error: {R_nrm=:.4e} / {A_nrm=:.4e}")

# now plot the matrices
def plot_sparse_log(_ax, csr_mat):
    np_mat = np.zeros(csr_mat.shape)
    np_mat[:,:] = np.nan # values that are not in sparsity will be nan
    _rowp, _cols = csr_mat.indptr, csr_mat.indices
    for i in range(N):
        for jp in range(_rowp[i], _rowp[i+1]):
            j = _cols[jp]
            _val = csr_mat[i,j]
            _log_val = np.log10(1e-12 + _val**2)
            np_mat[i,j] = _log_val
    _ax.imshow(np_mat)
    return np_mat
# print(f"{L.toarray()=}")

plt.imshow(A0.toarray())
plt.show()
    

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
matrices = [A0, R, L, U]
for ind, my_mat in enumerate(matrices):
    i, j = ind // 2, ind % 2
    plot_sparse_log(ax[i,j], my_mat.copy())
plt.show()
