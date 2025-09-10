from _utils import get_beam_csr_mat_and_rhs, get_plate_csr_mat_and_rhs
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# many steps taken from this book, [Algorithms for Sparse Linear Systems](https://link.springer.com/book/10.1007/978-3-031-25820-6) book
# page 51 gives dense matrix gauss-elim process illustrated in prev example 1_dense_gauss_elim.py
# here I extend to sparse Gauss-elim for symmetric matrix (so I only need elimination tree)

# see if random order stabilize LU factor... will add fillin though..
# random_order = False
random_order = True

# # nxe = 4
# # nxe = 16 # beam problem is getting less accurate LU factor with standard ordering with more DOF also.. probably due to longer chain lengths
# nxe = 32 # gets way less accurate LU factor.. even for beam as go up in DOF
# # nxe = 64
# # I can make some plots of this stuff as well for my paper?
# A, rhs = get_beam_csr_mat_and_rhs(nxe, csr=True)

# plate case is much larger and blows up numerically right now.. due to long chain lengths
nxe = 4
A, rhs = get_plate_csr_mat_and_rhs(nxe, csr=True)
A0 = A.copy()
# it's a 2DOF per node or Bsr2 matrix stored as CSR

# now show the LU factorization using Gaussian elimination..
# A will become U in-place now..
N = A.shape[0]
orig_rowp, orig_cols = A.indptr, A.indices

# compute permutation map, perm sends to permuted nodes, iperm back to unpermuted (standard order)
if random_order:
    perm = np.random.permutation(N)
    ind = np.arange(0, N)
    iperm = np.zeros(N, dtype=np.int32)
    iperm[perm] = ind

    # compute new A_perm and sparsity
    orig_nnz = orig_cols.shape[0]
    rowp = np.zeros(N + 1, dtype=np.int32)
    rows = np.zeros(orig_nnz, dtype=np.int32)
    cols = np.zeros(orig_nnz, dtype=np.int32)
    for perm_node in range(N):
        node = iperm[perm_node]
        row_ct = orig_rowp[node + 1] - orig_rowp[node]
        rowp[perm_node + 1] = rowp[perm_node] + row_ct
        
        c_perm_cols = np.sort(np.array([perm[orig_cols[jp]] for jp in range(orig_rowp[node], orig_rowp[node+1])]))
        perm_start = rowp[perm_node]
        rows[perm_start:(perm_start+row_ct)] = perm_node
        cols[perm_start:(perm_start+row_ct)] = c_perm_cols

    # now copy values out of A into A0 which is now permuted
    perm_vals = np.zeros(orig_nnz, dtype=np.double)
    A0 = sp.sparse.csr_matrix((perm_vals, (rows, cols)), shape=(N, N))
    for i in range(N):
        for jp in range(orig_rowp[i], orig_rowp[i+1]):
            j = orig_cols[jp]
            pi, pj = perm[i], perm[j]
            A0[pi, pj] = A[i, j]

else:
    rowp, cols = orig_rowp, orig_cols
    A0 = A0

# first we compute the elimination tree using Algorithm 4.2
parent, ancestor = np.zeros(N, dtype=np.int32), np.zeros(N, dtype=np.int32)
for i in range(N):
    parent[i], ancestor[i] = 0, 0
    
    # for all j such that a_ij neq 0 (this row basically in CSR), do:
    for jp in range(rowp[i], rowp[i+1]):
        j = cols[jp]
        if not(j < i): continue
        jroot = j

        while ((ancestor[jroot] != 0) and (ancestor[jroot] != i)):
            l = ancestor[jroot]
            ancestor[jroot] = i # path compression to accel future searches
            jroot = l

        if ancestor[jroot] == 0:
            ancestor[jroot] = i
            parent[jroot] = i

# print(f"{parent=}")
# print(f"{ancestor=}")

# now compute updated rowp, cols with fillin..
fill_L_rowp = np.zeros(N+1, dtype=np.int32)
# first just go through and get row counts.. then we'll come back and alloc cols
mark = -1 * np.ones(N, dtype=np.int32)

for i in range(N):
    c_cols = []
    mark[i] = i # encountered diagonal entry

    for jp in range(rowp[i], rowp[i+1]):
        k = cols[jp]
        # this condition gives you only sparsity of L
        if not(k < i): continue
        j = k
        while (mark[j] != i): # while col j has not encountered row i yet
            mark[j] = i
            c_cols += [j]
            j = parent[j]

    ncols = len(c_cols)
    fill_L_rowp[i+1] = fill_L_rowp[i] + ncols

L_nnz = fill_L_rowp[-1]
# print(F"{nnz=}")
fill_L_rows = np.zeros(L_nnz, dtype=np.int32)
fill_L_cols = np.zeros(L_nnz, dtype=np.int32)
mark = -1 * np.ones(N, dtype=np.int32)

for i in range(N):
    c_cols = []
    mark[i] = i # col i has encountered row i

    for jp in range(rowp[i], rowp[i+1]):
        k = cols[jp]
        # this condition gives you only sparsity of L
        if not(k < i): continue
        j = k
        while (mark[j] != i): # while col j has not encountered row i yet
            mark[j] = i
            c_cols += [j]
            j = parent[j]

    # print(F"{c_cols=}")
    ncols = len(c_cols)
    if ncols > 0:
        c_cols_sort = np.sort(c_cols)
        start, end = fill_L_rowp[i], fill_L_rowp[i+1]
        fill_L_cols[start : end] = c_cols_sort

# now take strict lower triang sparsity L and compute full fillin sparsity fill_rowp, fill_cols
fill_row_cts = np.ones(N, dtype=np.int32) # start with 1 for diags
for i in range(N):
    for jp in range(fill_L_rowp[i], fill_L_rowp[i+1]):
        j = fill_L_cols[jp]

        # add one nz above and below diag
        fill_row_cts[i] += 1
        fill_row_cts[j] += 1

# build new rowp
fill_rowp = np.zeros(N+1, dtype=np.int32)
for i in range(N):
    fill_rowp[i+1] = fill_rowp[i] + fill_row_cts[i]
fill_nnz = fill_rowp[-1]
fill_cols = np.zeros(fill_nnz, dtype=np.int32)
fill_rows = np.zeros(fill_nnz, dtype=np.int32)

# build rows nnz vec (needed only for python CSR)
for i in range(N):
    for jp in range(fill_rowp[i], fill_rowp[i+1]):
        fill_rows[jp] = i

# build new cols
next = fill_rowp.copy() # tracks fill in per row
# go through each row, first only putting in below diag + diag
for i in range(N):
    # put below diag in first..
    for jp in range(fill_L_rowp[i], fill_L_rowp[i+1]):
        j = fill_L_cols[jp]
        fill_cols[next[i]] = j
        next[i] += 1

    # add diagonal
    fill_cols[next[i]] = i
    next[i] += 1
    
# now go back and add above diag in..
for i in range(N):
    # add above diag
    for jp in range(fill_L_rowp[i], fill_L_rowp[i+1]):
        j = fill_L_cols[jp]
        
        # i,j as row,col now flipped for above diag
        fill_cols[next[j]] = i
        next[j] += 1


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
ones_L_fill = convert_rowp_cols_to_ones_mat(fill_L_rowp, fill_L_cols)
ones_fill = convert_rowp_cols_to_ones_mat(fill_rowp, fill_cols)

# # ones_L_Fill to be zero where already exists in nofill.. temporarily
# for i in range(N):
#     for jp in range(fill_rowp[i], fill_rowp[i+1]):
#         j = fill_cols[jp]
#         for jp2 in range(rowp[i], rowp[i+1]):
#             j2 = cols[jp2]
#             if j == j2:
#                 ones_L_fill[i,j] = 0.0

# show A nofill vs L fill and A fill sparsity patterns
# -----------------------------------------------
# fig, ax = plt.subplots(1, 3, figsize=(10, 7))
# # ax[0].spy(ones_nofill)
# # ax[1].spy(ones_L_fill)
# ax[0].imshow(ones_nofill.toarray())
# ax[1].imshow(ones_L_fill.toarray())
# ax[2].imshow(ones_fill.toarray())
# plt.show()

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
# R = A0_fill - L @ U # this doens't have right sparsity.. ?
R_nrm = np.linalg.norm(R.toarray())
A_nrm = np.linalg.norm(A0.toarray())
print(f"LU factor error: {R_nrm=:.4e} / {A_nrm=:.4e}")

# now plot the matrices
def plot_sparse_log(_ax, csr_mat):
    np_mat = csr_mat.toarray()
    np_mat[np_mat == 0.0] = np.nan
    np_mat[np_mat != 0.0] = np.log(np.abs(np_mat[np_mat != 0.0])) #  + 1e-3
    # print(f"{np_mat=}")
    _ax.imshow(np_mat)
    return np_mat
# print(f"{L.toarray()=}")
    

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
matrices = [A0_fill, R, L, U]
for ind, my_mat in enumerate(matrices):
    i, j = ind // 2, ind % 2
    plot_sparse_log(ax[i,j], my_mat.copy())
plt.show()
