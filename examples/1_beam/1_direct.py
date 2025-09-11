# now let's test this out and visualize it
import numpy as np
from qordering import BeamFem
import scipy as sp

E = 2e7; b = 4e-3; L = 1; rho = 1
qmag, ys, rho_KS = 2e-2, 4e5, 50.0
nxe = num_elements = int(3e2) #100, 300, 1e3
nxh = 100 
hvec = np.array([1e-3] * nxh)

# create and assemble FEA problem
beam_fea = BeamFem(nxe, nxh, E, b, L, rho, qmag, ys, rho_KS, dense=False)
helem_vec = beam_fea.get_helem_vec(hvec)
beam_fea._compute_mat_vec(helem_vec)
mat, rhs = beam_fea.Kmat, beam_fea.force

# solve FEA problem
beam_fea.u = sp.sparse.linalg.spsolve(mat, rhs)
beam_fea.plot_disp()

# see sparsity of rowp and cols in beam..
rowp, cols = mat.indptr, mat.indices
# this is CSR not BSR format here.. that's why it repeats columns twice..
# dof per node is 2 here
for inode in range(4): # look only at first four nodes..
    ncols = rowp[inode + 1] - rowp[inode]
    start, stop = rowp[inode], rowp[inode + 1]
    cols_slice = cols[start : stop]
    print(f"{inode=} : {cols_slice=}")