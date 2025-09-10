# now let's test this out and visualize it
import numpy as np
from qordering import PlateAssembler
import scipy as sp

nxe, nxc = 35, 5
plate_fea = PlateAssembler.aluminum_unitsquare_trigload(
    num_elements=nxe**2,
    num_components=nxc**2,
    rho_KS=200.0,
    qmag=2e-2, 
    can_print=False
)

ncomp = plate_fea.ncomp
hred = np.array([5e-3] * ncomp)

helem_vec = plate_fea.get_helem_vec(hred)
plate_fea._compute_mat_vec(helem_vec)
mat, rhs = plate_fea.Kmat, plate_fea.force

# solve FEA problem
plate_fea.u = sp.sparse.linalg.spsolve(mat, rhs)
plate_fea.plot_disp()

# see sparsity of rowp and cols in beam..
rowp, cols = mat.indptr, mat.indices
# this is CSR not BSR format here.. that's why it repeats columns twice..
# dof per node is 3 here and each col should have more values..
for inode in range(4): # look only at first four nodes..
    ncols = rowp[inode + 1] - rowp[inode]
    start, stop = rowp[inode], rowp[inode + 1]
    cols_slice = cols[start : stop]
    print(f"{inode=} : {cols_slice=}")