from _utils import get_beam_csr_mat_and_rhs, get_plate_csr_mat_and_rhs
from qordering import random_ordering, get_reordered_nofill_matrix, get_transpose_pattern, get_rows_from_rowp
from qordering import get_elim_tree, get_L_fill_pattern, get_LU_fill_matrix, get_lower_triang_pattern
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import argparse

# we now try full column LU with pivoting

parser = argparse.ArgumentParser()
parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=False, help="Plot matrices and residual")
parser.add_argument("--nxe", type=int, default=4, help="nxe # elements in x-dir")
parser.add_argument("--case", type=str, default="beam", help="options: [beam, plate]")
args = parser.parse_args()

# see if random order stabilize LU factor... will add fillin though..
if (args.case == "beam"):
    A, rhs = get_beam_csr_mat_and_rhs(args.nxe, csr=True)
    # convert to bsr matrix
    A = sp.sparse.bsr_matrix(A)
    block_dim = A.data.shape[-1] # w, theta should be 2

elif (args.case == "plate"):
    A, rhs = get_plate_csr_mat_and_rhs(args.nxe, csr=True)
    A = sp.sparse.bsr_matrix(A)
    block_dim = A.data.shape[-1] # w, dw/dx, dw/dy should be 3

# copy the matrix
A0 = A.copy()
N = A.shape[0]
nnodes = N // block_dim
orig_rowp, orig_cols = A.indptr, A.indices

# get full fill pattern of L in CSR format first
rowp, cols = A0.indptr, A0.indices
A_fill = get_LU_fill_matrix(A0)

# plt.imshow(A_fill.toarray())
# plt.show()

L_rowp, L_cols = get_lower_triang_pattern(A_fill, strict_lower=False)
U_rowp, U_cols, _ = get_transpose_pattern(nnodes, L_rowp, L_cols)
U_nnzb = U_rowp[-1]

# make U = LT matrix (full fillin), easier to do since uses almost exclusively 
A_orig = A.copy()
A = A0.copy()
U_vals = np.zeros((U_nnzb, block_dim, block_dim), dtype=np.double)
_next = U_rowp[:-1].copy() # temp util array for inserting cols
for ib in range(nnodes):
    for jp in range(rowp[ib], rowp[ib+1]):
        jb = cols[jp]
        if ib <= jb: # for U pattern only
            for inz in range(block_dim**2):
                ii, jj = inz % block_dim, inz // block_dim
                U_vals[_next[ib], ii, jj] = A0.data[jp, ii, jj]
            _next[ib] += 1
U = sp.sparse.bsr_matrix((U_vals, U_cols, U_rowp), shape=(N, N))


# compute block IC(infty) factorization (greater parallelization on GPU or CPU), full fillin
# algorithm 5.2 from "Algorithms for Sparse Linear Systems" book
print(f"begin BSR cholesky factorization")
for jb in range(nnodes):
    # don't think I need the copy from A into U now since I have already done that..

    # now begin the factor steps..
    for kb in range(jb): # try each row of U, or col of L, less than j
        # lots of memory reads here.. oof
        # find the location of col j if exists.. aka l_jk = U_kj
        U_kj_T = None
        for cp in range(U_rowp[kb], U_rowp[kb+1]):
            cb = U_cols[cp]
            if cb == jb:
                U_kj_T = U.data[cp,:,:].T # matrix

        # read down row k find a col...
        for cp_k in range(U_rowp[kb], U_rowp[kb+1]):
            cb_k = U_cols[cp_k]

            # read down row j find a col (need match block col)
            # TODO : isn't this so much mem read overhead?
            for cp_j in range(U_rowp[jb], U_rowp[jb+1]):
                cb_j = U_cols[cp_j]
                if cb_k == cb_j:

                    # U[j,:] -= U[k,j]^T @ U[k,:] @ block cols
                    U.data[cp_j,:,:] -= U_kj_T @ U.data[cp_k,:,:]
            
    # in-place factorization of diagonal entries.. in the block.. of U[jb,jb]
    # store as diagonal same for each, off-diag are copies.. just compute of U first then copy to lower diag
    # based on dense chol here..
    # copy out matrix (temp here method for debug)
    cp = U_rowp[jb] # in U, first entry of each row is the diagonal entry (so convenient here)
    L1 = U.data[cp, :, :].copy()
    for jj in range(block_dim):
        for kk in range(jj):
            L1[jj:,jj] -= L1[jj:,kk] * L1[jj,kk]
        L1[jj,jj] = np.sqrt(L1[jj,jj])
        L1[(jj+1):,jj] /= L1[jj,jj]
    # now copy to upper strict part only
    for jj in range(block_dim):
        for kk in range(jj+1,block_dim):
            L1[jj,kk] = L1[kk,jj]
    U.data[cp,:,:] = L1[:,:]

    # now do dense triang solve update.. on row jb
    # could do more efficiently, don't really care rn
    for cb in range(U_rowp[jb]+1, U_rowp[jb+1]): # +1 to exclude diag with itself
        cp = U_cols[cb]
        
        U.data[cp,:,:] -= np.linalg.solve(L1, U.data[cp,:,:])

# print("done with BSR cholesky factor")

# now check the accuracy of the cholesky factor here..
U_np = U.toarray()
A_np = A.toarray()
R = A_np - U_np.T @ U_np
# not quite how you'd compute LL^T or U^T U here.. as block case is kind of weird.. maybe do triang solves to get inv matrix operator, then see if is inverse?
R_nrm = np.max(np.abs(R))
print(f"{R_nrm=}")

# # show L, U and A
if args.plot:
    fig, ax = plt.subplots(2,2, figsize=(12, 8))
    ax[0,0].imshow(U.toarray().T)
    ax[0,1].imshow(U.toarray())
    ax[1,0].imshow(A.toarray())
    ax[1,1].imshow(R)
    plt.show()
    exit()