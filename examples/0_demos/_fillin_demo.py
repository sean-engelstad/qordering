import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from qordering import compute_LU_fill_pattern

# matrix from page 57 of book [Algorithms for Sparse Linear Systems](https://link.springer.com/book/10.1007/978-3-031-25820-6)

N = 8
rowp = np.array([0, 3, 7, 10, 13, 16, 18, 20, 24])
cols = np.array([0, 4, 5, 1, 3, 4, 7, 2, 3, 7, 1, 2, 3, 0, 1, 4, 0, 5, 6, 7, 1, 2, 6, 7])
nnz = rowp[-1]
rows = np.zeros(nnz, dtype=np.int32)
for i in range(N):
    for jp in range(rowp[i], rowp[i+1]):
        rows[jp] = i
vals = np.ones(nnz, dtype=np.double)
A = sp.sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))

# plt.spy(A)
# plt.show()

fill_rowp, fill_rows, fill_cols = compute_LU_fill_pattern(N, rowp, cols)
fill_nnz = fill_rowp[-1]

# now make the matrix for fillin
fill_vals = np.ones(fill_nnz, dtype=np.double)
A_fill = sp.sparse.csr_matrix((fill_vals, (fill_rows, fill_cols)), shape=(N,N))

# plot
fig, ax = plt.subplots(1, 2, figsize=(10, 7))
ax[0].spy(A)
ax[1].spy(A_fill)
plt.show()

# it worked!