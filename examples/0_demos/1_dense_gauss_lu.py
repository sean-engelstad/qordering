from _utils import get_beam_csr_mat_and_rhs
import numpy as np
import matplotlib.pyplot as plt

# from page 51 of [Algorithms for Sparse Linear Systems](https://link.springer.com/book/10.1007/978-3-031-25820-6) book
# a simple dense matrix gauss elimination

nxe = 4
# nxe = 300
# full matrix here
A, rhs = get_beam_csr_mat_and_rhs(nxe, csr=False)
A0 = A.copy()
# it's a 2DOF per node or Bsr2 matrix
# but here it's stored as CSR

# now show the LU factorization using Gaussian elimination..
# A will become U in-place now..
N = A.shape[0]
L = np.eye(N)

# plt.imshow(np.log(np.abs(A) + 1e-12))
# plt.show()

for k in range(N-1):
    for i in range(k+1, N):
        L[i,k] = A[i,k] / A[k,k]
        
        for j in range(k, N):
            A[i,j] -= L[i,k] * A[k,j]
    
    # plt.imshow(np.log(np.abs(A) + 1e-12))
    # plt.show()

U = A.copy()

fig, ax = plt.subplots(1, 2, figsize=(10, 7))
ax[0].imshow(np.log(np.abs(L) + 1e-12))
ax[1].imshow(np.log(np.abs(U) + 1e-12))
plt.show()

A_hat = L @ U
R = A0 - A_hat
plt.imshow(np.log(np.abs(R) + 1e-12))
plt.show()
R_norm = np.linalg.norm(R)
print(f"{R_norm=}")