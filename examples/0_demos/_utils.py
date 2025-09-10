import numpy as np
from qordering import BeamFem, PlateAssembler

def get_beam_csr_mat_and_rhs(nxe=300, csr:bool=True):
    E = 2e7; b = 4e-3; L = 1; rho = 1
    qmag, ys, rho_KS = 2e-2, 4e5, 50.0
    nxh = np.min([100, nxe])
    hvec = np.array([1e-3] * nxh)

    # create and assemble FEA problem
    beam_fea = BeamFem(nxe, nxh, E, b, L, rho, qmag, ys, rho_KS, dense=False)
    helem_vec = beam_fea.get_helem_vec(hvec)
    beam_fea._compute_mat_vec(helem_vec)
    mat, rhs = beam_fea.Kmat, beam_fea.force
    if not csr:
        mat = mat.toarray()
    return mat, rhs

def get_plate_csr_mat_and_rhs(nxe=300, csr:bool=True):
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
    return mat, rhs

