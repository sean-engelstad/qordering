import numpy as np
from qordering import BeamFem, PlateAssembler
import scipy as sp

def get_beam_csr_mat_and_rhs(nxe=300, csr:bool=True, remove_bcs:bool=False):
    E = 2e7; b = 4e-3; L = 1; rho = 1
    qmag, ys, rho_KS = 2e-2, 4e5, 50.0
    nxh = nxe
    hvec = np.array([1e-2] * nxh)

    # create and assemble FEA problem
    beam_fea = BeamFem(nxe, nxh, E, b, L, rho, qmag, ys, rho_KS, dense=False)
    helem_vec = beam_fea.get_helem_vec(hvec)
    beam_fea._compute_mat_vec(helem_vec)
    mat, rhs = beam_fea.Kmat, beam_fea.force
    if not csr:
        mat = mat.toarray()

    if remove_bcs:
        mat = mat.toarray()
        mat, rhs = remove_bcs(mat, rhs)
        if csr:
            mat = sp.sparse.linalg.csr_matrix(mat)

    return mat, rhs

def get_plate_csr_mat_and_rhs(nxe=300, csr:bool=True, remove_bcs:bool=False):
    nxc = 1
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
    if not csr:
        mat = mat.toarray()

    if remove_bcs:
        mat = mat.toarray()
        mat, rhs = remove_bcs(mat, rhs)
        if csr:
            mat = sp.sparse.linalg.csr_matrix(mat)

    return mat, rhs

def remove_bcs(mat, rhs, csr:bool=False):
    # TODO : may do csr later..

    N = mat.shape[0]
    bcs = []
    for i in range(N):
        if mat[i,i] == 1.0:
            bcs += [i]
    
    free_vars = np.array([_ for _ in range(N) if not(_ in bcs)])
    mat = mat[free_vars,:][:,free_vars]
    rhs = rhs[free_vars]
    return mat, rhs