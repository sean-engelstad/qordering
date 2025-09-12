import numpy as np
import scipy as sp

def get_elim_tree(N, rowp, cols):
    # first we compute the elimination tree using Algorithm 4.2 (for sym matrix..)
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

    return parent, ancestor

def get_L_fill_pattern(N, rowp, cols, parent, ancestor, strict_lower:bool=True):
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
    # fill_L_rows = np.zeros(L_nnz, dtype=np.int32)
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

    if not strict_lower:
        # add diagonal to L (it gives just strict L originally)
        L_rowp = np.zeros(N+1, dtype=np.int32)
        for i in range(N):
            row_ct = fill_L_rowp[i+1] - fill_L_rowp[i]
            L_rowp[i+1] = L_rowp[i] + (row_ct + 1)
        L_nnz = L_rowp[-1]
        L_rows = np.zeros(L_nnz, dtype=np.int32)
        for i in range(N):
            for jp in range(L_rowp[i], L_rowp[i+1]):
                L_rows[jp] = i
        L_cols = np.zeros(L_nnz, dtype=np.int32)
        for i in range(N):
            strict_start = fill_L_rowp[i]
            start = L_rowp[i]
            for jp in range(fill_L_rowp[i], fill_L_rowp[i+1]):
                offset = jp - strict_start
                L_cols[start + offset] = fill_L_cols[jp]
            L_cols[L_rowp[i+1]-1] = i # diag entry

        fill_L_rowp, fill_L_cols = L_rowp, L_cols

    return fill_L_rowp, fill_L_cols

def get_lower_triang_pattern(A, strict_lower:bool=False):
    # get general lower triang pattern from A
    N = A.shape[0]
    rowp, cols = A.indptr, A.indices

    L_row_cts = np.zeros(N, dtype=np.int32)
    for i in range(N):
        for jp in range(rowp[i], rowp[i+1]):
            j = cols[jp]
            if strict_lower and i > j:
                L_row_cts[i] += 1
            elif not(strict_lower) and i >= j:
                L_row_cts[i] += 1

    L_rowp = np.zeros(N+1, dtype=np.int32)
    for i in range(N):
        L_rowp[i+1] = L_rowp[i] + L_row_cts[i]

    nnz = L_rowp[-1]
    L_rows = np.zeros(nnz, dtype=np.int32)
    for i in range(N):
        for jp in range(L_rowp[i], L_rowp[i+1]):
            L_rows[jp] = i

    L_cols = np.zeros(nnz, dtype=np.int32)
    next = np.zeros(N, dtype=np.int32) # keeps track of how much we've filled each row in CSR format
    for i in range(N):
        for jp in range(rowp[i], rowp[i+1]):
            j = cols[jp]

            insert = False
            if strict_lower and i > j:
                insert = True
            elif not(strict_lower) and i >= j:
                insert = True

            if insert:
                start = L_rowp[i]
                L_cols[start + next[i]] = j
                next[i] += 1
    return L_rowp, L_cols

def get_transpose_pattern(N, rowp, cols, get_map:bool=False):
    # general get transpose pattern
    # construct also the strict upper triangular part..
    # so we can slice by columns as well..
    tr_row_cts = np.zeros(N, dtype=np.int32)
    for i in range(N):
        for jp in range(rowp[i], rowp[i+1]):
            j = cols[jp]
            tr_row_cts[j] += 1

    tr_rowp = np.zeros(N+1, dtype=np.int32)
    for i in range(N):
        tr_rowp[i+1] = tr_rowp[i] + tr_row_cts[i]

    nnz = tr_rowp[-1]
    tr_rows = np.zeros(nnz, dtype=np.int32)
    for i in range(N):
        for jp in range(tr_rowp[i], tr_rowp[i+1]):
            tr_rows[jp] = i

    tr_cols = np.zeros(nnz, dtype=np.int32)
    next = np.zeros(N, dtype=np.int32) # keeps track of how much we've filled each row in CSR format
    for i in range(N):
        for jp in range(rowp[i], rowp[i+1]):
            j = cols[jp]
            # flip i,j as row,col => j,i in our head
            start = tr_rowp[j]
            tr_cols[start + next[j]] = i
            next[j] += 1

    # map from nnz in transpose to un-transpose storage
    if get_map:
        tr_map = np.zeros(nnz, dtype=np.int32)
        for i in range(N):
            for jp in range(rowp[i], rowp[i+1]):
                j = cols[jp]

                # find equivalent indptr in transpose matrix
                for ip in range(tr_rowp[j], tr_rowp[j+1]):
                    i2 = tr_cols[ip]
                    if i == i2:
                        tr_map[ip] = jp
                        break
    else:
        tr_map = None

    return tr_rowp, tr_cols, tr_map

def get_rows_from_rowp(N, rowp):
    nnz = rowp[-1]
    rows = np.zeros(nnz, dtype=np.int32)
    for i in range(N):
        for jp in range(rowp[i], rowp[i+1]):
            rows[jp] = i
    return rows

def strict_L_to_full_LU_patern(N, L_rowp, L_cols):
    # now take strict lower triang sparsity L and compute full fillin sparsity fill_rowp, fill_cols
    fill_row_cts = np.ones(N, dtype=np.int32) # start with 1 for diags
    for i in range(N):
        for jp in range(L_rowp[i], L_rowp[i+1]):
            j = L_cols[jp]

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
        for jp in range(L_rowp[i], L_rowp[i+1]):
            j = L_cols[jp]
            fill_cols[next[i]] = j
            next[i] += 1

        # add diagonal
        fill_cols[next[i]] = i
        next[i] += 1
        
    # now go back and add above diag in..
    for i in range(N):
        # add above diag
        for jp in range(L_rowp[i], L_rowp[i+1]):
            j = L_cols[jp]
            
            # i,j as row,col now flipped for above diag
            fill_cols[next[j]] = i
            next[j] += 1
    
    return fill_rowp, fill_rows, fill_cols

def compute_LU_fill_pattern(N, rowp, cols):
    """procedure for full LU fill pattern from symmetric square matrix"""
    parent, ancestor = get_elim_tree(N, rowp, cols)
    L_rowp, L_cols = get_L_fill_pattern(N, rowp, cols, parent, ancestor, strict_lower=True)
    fill_rowp, fill_rows, fill_cols = strict_L_to_full_LU_patern(N, L_rowp, L_cols)
    return fill_rowp, fill_rows, fill_cols

def get_LU_fill_matrix(A_orig:sp.sparse.csr_matrix) -> sp.sparse.csr_matrix:
    """full process of getting LU fill pattern and copying values for symmetric square matrix"""

    # get fill pattern first from original matrix
    N = A_orig.shape[0]
    rowp, cols = A_orig.indptr, A_orig.indices
    fill_rowp, fill_rows, fill_cols = compute_LU_fill_pattern(N, rowp, cols)
    fill_nnz = fill_rowp[-1]

    fill_vals = np.zeros(fill_nnz, dtype=np.double)
    A_fill = sp.sparse.csr_matrix((fill_vals, (fill_rows, fill_cols)), shape=(N, N))

    # copy values from nofill sparsity
    for i in range(N):
        for jp in range(rowp[i], rowp[i+1]):
            j = cols[jp]
            A_fill[i,j] = A_orig[i,j]
    return A_fill
    