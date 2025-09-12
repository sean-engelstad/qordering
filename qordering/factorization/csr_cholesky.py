import numpy as np
import scipy as sp
from ..symbolic.fillin import get_transpose_pattern
from ..symbolic.fillin import get_lower_triang_pattern, get_rows_from_rowp

def csr_cholesky(A:sp.sparse.csr_matrix) -> sp.sparse.csr_matrix:
    # get pointers and arrays out of it
    # assume A is reordered and/or has fill pattern or not inside it already..
    N = A.shape[0]

    L_rowp, L_cols = get_lower_triang_pattern(A, strict_lower=False)
    LT_rowp, LT_cols, _ = get_transpose_pattern(N, L_rowp, L_cols)
    L_nnz = L_rowp[-1]
    L_rows = get_rows_from_rowp(N, L_rowp)
    L_vals = np.zeros(L_nnz, dtype=np.double)
    L = sp.sparse.csr_matrix((L_vals, (L_rows, L_cols)))

    # TODO : could write more efficient version that doesn't use operator[], and actually slices through arrays itself..
    # probably is doing extra internal for loops in operator[] than are necessary, but for qorder accuracy questions and conv
    # don't care yet..

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
        
        # print(f"{L[j,j]=}")
        L[j,j] = np.sqrt(L[j,j])
        for rp in range(LT_rowp[j], LT_rowp[j+1]):
            r = LT_cols[rp]
            if r < j+1: continue # supposed to be j+1 to N so slightly larger
            L[r,j] /= L[j,j]
    return L

def get_transpose_matrix(L):
    # when you have L factored, construct LT sparse matrix in CSR format
    N = L.shape[0]
    L_rowp, L_cols = L.indptr, L.indices
    LT_rowp, LT_cols, _ = get_transpose_pattern(N, L_rowp, L_cols)
    LT_rows = get_rows_from_rowp(N, LT_rowp)
    nnz = LT_rowp[-1]
    LT_vals = np.zeros(nnz, dtype=np.double)
    LT = sp.sparse.csr_matrix((LT_vals, (LT_rows, LT_cols)), shape=(N, N))
    
    # not the most comp efficient way yet, but works.. (cause operator[] prob doing extra unnecessary for loops)
    for i in range(N):
        for jp in range(L_rowp[i], L_rowp[i+1]):
            j = L_cols[jp]
            LT[j,i] = L[i,j]

    return LT

class CholPrecond:
    def __init__(self, _L, _LT):
        self._L = _L
        self._LT = _LT

    def solve(self, _b):
        _y =  sp.sparse.linalg.spsolve_triangular(self._L, _b, lower=True)
        _x =  sp.sparse.linalg.spsolve_triangular(self._LT, _y, lower=False)
        return _x