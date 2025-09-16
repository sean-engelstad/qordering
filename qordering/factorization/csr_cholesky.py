import numpy as np
import scipy as sp
from ..symbolic.fillin import get_transpose_pattern, get_elim_tree, get_L_fill_pattern, get_upper_triang_pattern
from ..symbolic.fillin import get_lower_triang_pattern, get_rows_from_rowp, get_fillin_only_pattern
from ..symbolic.reorder import get_upper_triang_values

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

        for kp in range(L_rowp[j], L_rowp[j+1]): # would prefer to use the same rowp everywhere..
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

def vmicf_cholesky(A0:sp.sparse.csr_matrix, make_not_m:bool=False) -> sp.sparse.csr_matrix:

    N = A0.shape[0]
    rowp, cols = A0.indptr, A0.indices

    # temporarily make all negative off diag entries positive (for SPD testing?), 
    # from paper https://www.researchgate.net/publication/264189677_Modified_incomplete_Cholesky_factorization_preconditioners_for_a_symmetric_positive_definite_matrix/link/543fa8830cf23da6cb5b91c8/download?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19
    if make_not_m: # make not an M-matrix?
        A = A0.copy()
        for i in range(N):
            for jp in range(rowp[i], rowp[i+1]):
                j = cols[jp]
                if not(i == j): # off-diag entries
                    A[i,j] = np.abs(A[i,j])
        A0 = A.copy()

    # get U fill pattern, then split into fillin and nofill pattern
    parent, ancestor = get_elim_tree(N, rowp, cols)
    L_fill_rowp, L_fill_cols = get_L_fill_pattern(N, rowp, cols, parent, ancestor, strict_lower=False)
    U_fill_rowp, U_fill_cols, _ = get_transpose_pattern(N, L_fill_rowp, L_fill_cols)
    U_rowp, U_cols = get_upper_triang_pattern(A0, strict_upper=False)
    U_nnz = U_rowp[-1]
    U_rows = get_rows_from_rowp(N, U_rowp)
    U_fillin_rowp, U_fillin_cols = get_fillin_only_pattern(N, U_rowp, U_cols, U_fill_rowp, U_fill_cols)
    U_vals = get_upper_triang_values(N, rowp, cols, A0.data, U_rowp, U_cols)
    U_rows = get_rows_from_rowp(N, U_rowp)
    U0 = sp.sparse.csr_matrix((U_vals, (U_rows, U_cols)), shape=(N, N))

    # use csr sparsity pattern to do this..
    # --------------------------------------------------
    A = A0.copy()

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
    L = get_transpose_matrix(U)
    return L, D, U

class CholPrecond:
    def __init__(self, _L, _LT, D=None):
        self._L = _L
        self._LT = _LT
        self._D = D
        self._d = np.diag(D) if D is not None else None
        self._dinv = 1.0 / self._d if self._d is not None else None

    def solve(self, _b):
        _y =  sp.sparse.linalg.spsolve_triangular(self._L, _b, lower=True)
        if self._d is not None:
            _y *= self._dinv
        _x =  sp.sparse.linalg.spsolve_triangular(self._LT, _y, lower=False)
        return _x