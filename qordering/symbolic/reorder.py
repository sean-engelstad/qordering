import numpy as np
import scipy as sp

def random_ordering(N, rowp, cols):
    # give new nofill pattern for random ordering

    # compute permutations
    perm = np.random.permutation(N)
    ind = np.arange(0, N)
    iperm = np.zeros(N, dtype=np.int32)
    iperm[perm] = ind
    return perm, iperm

def get_reordered_nofill_pattern(N, orig_rowp, orig_cols, perm, iperm):
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
    
    return rowp, rows, cols

def get_reordered_nofill_matrix(A_orig:sp.sparse.csr_matrix, perm, iperm) -> sp.sparse.csr_matrix:
    # compute reordered nofill matrix with values
    
    N = A_orig.shape[0]
    orig_rowp, orig_cols = A_orig.indptr, A_orig.indices
    rowp, rows, cols = get_reordered_nofill_pattern(N, orig_rowp, orig_cols, perm, iperm)
    nnz = rowp[-1]

    # now copy values out of A into A0 which is now permuted
    perm_vals = np.zeros(nnz, dtype=np.double)
    A = sp.sparse.csr_matrix((perm_vals, (rows, cols)), shape=(N, N))
    for i in range(N):
        for jp in range(orig_rowp[i], orig_rowp[i+1]):
            j = orig_cols[jp]
            pi, pj = perm[i], perm[j]
            A[pi, pj] = A_orig[i, j]
    
    return A