"""
modified incomplete cholesky factorization (MICF) (nofill) example from the paper, algorithm 3.1 on page 6, from paper,
https://www.researchgate.net/profile/Jae-Yun-6/publication/264189677_Modified_incomplete_Cholesky_factorization_preconditioners_for_a_symmetric_positive_definite_matrix/links/543fa8830cf23da6cb5b91c8/Modified-incomplete-Cholesky-factorization-preconditioners-for-a-symmetric-positive-definite-matrix.pdf
the modified incomplete chol algorithms do need knowledge of the fillin as 'zero pattern set'.
* they add contributions that would belong to fillin back on the diagonal to stabilize the preconditioner (improves performance)

* this MICF algorithm though, may use a lot more memory in the intermediate stages as it does have to store intermediate
fillin values before it adds to diagonal then zero them out. The second or next algorithm VMICF is more efficient and doesn't
appear to require much additional temp memory storage
* first I show the simple example from the book with dense, then I will implement it for a larger matrix using CSR pattern
* most algorithms including cusparse IC(0) ignore fillin contributions completely in IChol, which can drastically
reduce the accuracy of the nofill preconditioner (or stability). These modified algorithms should greatly improve performance,
variants also exist for modified incomplete ILU(0)
"""

import numpy as np
import matplotlib.pyplot as plt

# original matrix (example 3.3)
A0 = np.array([
    [4, 0, -1, -1],
    [0, 2, 1, -1],
    [-1, 1, 2, 0],
    [-1, -1, 0, 2]
], dtype=np.double)
A = A0.copy()

# zero pattern set (in this simple example just one entry per col)
# so I use simpler shortcut here (fillin for more general may be harder)
Z = np.array([1, 0, 3, 2]) # which col per row in zero pattern set

# algorithm 3.1 (MICF)
N = 4
d = np.zeros(N)
d[0] = 1.0 / A[0,0]
# print(f"{d=} {A0=}")
for i in range(1,N):
    for j in range(i):
        for k in range(i,N):
            A[k,i] -= A[k,j] * A[i,j] * d[j]

    for k in range(i+1,N):
        if Z[k] == i:
            # MICF step (zero pattern or fillin spots will add to diag
            # to stabilize incomplete chol(0))
            A[i,i] += np.abs(A[k,i])
            A[k,k] += np.abs(A[k,i])
            A[k,i] = 0.0 # enforce zero
    print(f"{i}b - {d=}\n{A=}")
    d[i] = 1.0 / A[i,i]

# now check UT, D, R
D = np.diag(d)
L = A.copy()
for i in range(N):
    for j in range(i+1,N):
        L[i,j] = 0.0
R = L @ D @ L.T - A0
fig, ax = plt.subplots(1, 3, figsize=(12, 9))
ax[0].imshow(L)
ax[1].imshow(D)
ax[2].imshow(R)
plt.show()