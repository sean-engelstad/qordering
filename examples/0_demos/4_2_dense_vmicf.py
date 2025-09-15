"""
variational modified incomplete cholesky factorization (VMICF) (nofill) example from the paper, algorithm 3.1 on page 6, from paper,
https://www.researchgate.net/profile/Jae-Yun-6/publication/264189677_Modified_incomplete_Cholesky_factorization_preconditioners_for_a_symmetric_positive_definite_matrix/links/543fa8830cf23da6cb5b91c8/Modified-incomplete-Cholesky-factorization-preconditioners-for-a-symmetric-positive-definite-matrix.pdf
the modified incomplete chol algorithms do need knowledge of the fillin as 'zero pattern set'.
* they add contributions that would belong to fillin back on the diagonal to stabilize the preconditioner (improves performance)

* this VMICF algorithm probably uses way less memory because it does use fillin contributions and add to diag
* but it doesn't need full fill storage, it immediately goes into diag upon calc
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

# algorithm 3.2 (VMICF)
# VMICF should be slightly worse smoother (but very close to MICF)
# but doesn't need fillin pattern at all for intermed, WAY faster precond time so better than MICF (see paper)
N = 4
d = np.zeros(N)
for i in range(N):
    d[i] = 1.0 / A[i,i]
    for j in range(i+1,N):
        d[j] = A[j,i] * d[i]
    print(f"{d=}")
    for j in range(i+1,N):
        for k in range(j,N):
            if Z[k] == j:
                # A[k,j] -= d[k] * A[j,i]
                # you can immediately add this fillin value (would be zero no need to subtract)
                # into the diagonals..
                # so equiv is.. this will be important for efficient CSR nofill VMICF(0) implementation
                A[k,j] = -d[k] * A[j,i] # no fillin storage required..

                val = np.abs(A[k,j])
                print(f"A[{k=},{j=}] Z spot => inc Akk,Ajj by {val:.2e}")

                A[k,k] += np.abs(A[k,j])
                A[j,j] += np.abs(A[k,j])
                A[k,j] = 0.0
            else:
                val = d[k] * A[j,i]
                print(f"A[{k=},{j=}] nofill dec by {val:.2e}")

                A[k,j] -= d[k] * A[j,i]
            # in an efficient CSR implementaton,
            # since the nofill values are immediately added into the diagonals
            # you don't need to allocate memory spots for fillin still.. and you can add to diags first
            # cause the second part of the j & k for loops (nofill spots) don't depend on diags..
    print(f"{A=}")

# now check UT, D, R
D = np.diag(d)
L = A.copy()
for i in range(N):
    for j in range(i+1,N):
        L[i,j] = 0.0
R = L @ D @ L.T - A0

print(f"{L=}")
print(f"{d=}")

fig, ax = plt.subplots(1, 3, figsize=(12, 9))
ax[0].imshow(L)
ax[1].imshow(D)
ax[2].imshow(R)
plt.show()