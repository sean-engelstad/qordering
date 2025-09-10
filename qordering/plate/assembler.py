__all__ = ["PlateAssembler"]

import numpy as np
import scipy as sp
from ._plate_elem import *
from ._numba import *
from ._helper_classes import PlateLoads, PlateFemGeom, IsotropicMaterial
# from ..plots.style import plot_init
from matplotlib.ticker import FormatStrFormatter

def plot_init():
    plt.rcParams.update({
        # 'font.family': 'Courier New',  # monospace font
        'font.family' : 'monospace', # since Courier new not showing up?
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.titlesize': 20
    }) 

class PlateAssembler:
    def __init__(
        self, 
        material:IsotropicMaterial, 
        plate_fem_geom:PlateFemGeom,
        plate_loads:PlateLoads,
        rho_KS:float, 
        dense:bool=False, 
        use_numba:bool=True, 
        can_print:bool=False
    ):
        
        # unpack data from the objects
        E, nu, rho, ys = material.E, material.nu, material.rho, material.ys
        nxe, nye, nxh, nyh, a, b = plate_fem_geom.nxe, plate_fem_geom.nye, plate_fem_geom.nxh, plate_fem_geom.nyh, plate_fem_geom.a, plate_fem_geom.b
        qmag, load_fcn = plate_loads.qmag, plate_loads.load_fcn

        self.nxe = nxe
        self.nye = nye

        # nxh by nyh components (separate from elements)
        self.nxh = nxh
        self.nyh = nyh
        self.ncomp = nxh * nyh
        # nnxh by nnyh number of elements per component in each direction
        self.nnxh = nxe // nxh
        self.nnyh = nye // nyh

        self.E = E
        self.nu = nu
        self.a = a
        self.b = b
        self.rho = rho
        self.rho_KS = rho_KS
        self.qmag = qmag
        self.ys = ys
        self.load_fcn = load_fcn

        # solve settings
        self._dense = dense
        self.use_numba = use_numba # makes the code way faster.. (use it, but then can't do complex step test with it active)
        self.can_print = can_print

        # solve states
        self.Kmat = None
        self.force = None
        self.u = None
        # adjoint only required for stress function
        self.psis = None

        # prelim mesh info
        self.num_elements = nxe * nye
        self.nnx = nxe + 1; self.nny = nye + 1
        self.num_nodes = (nxe + 1) * (nye+1)
        self.num_dof = 3 * self.num_nodes
        self.xscale = a / nxe
        self.yscale = b / nye
        self.dx = self.xscale
        self.dy = self.yscale

        self.dof_per_node = 3

        # define bcs
        self.bcs = []
        for iy in range(self.nny):
            for ix in range(self.nnx):
                inode = iy * self.nnx + ix
                w_idof = 3 * inode
                if ix == 0 or ix == self.nnx - 1 or iy == 0 or iy == self.nny-1:
                    self.bcs += [w_idof]
        self.bcs = np.array(self.bcs)
    
        # define element connectivity (node dof)
        self.conn = []
        for iy in range(nye):
            for ix in range(nxe):
                istart = ix + self.nnx * iy
                istart2 = istart + self.nnx

                self.conn += [[istart, istart+1, istart2, istart2+1]]

        # define the node locations here
        self.xpts = []
        xoffset = -a/2 # center at origin
        yoffset = -b/2
        for iy in range(self.nny):
            for ix in range(self.nnx):

                xval = xoffset + ix * self.dx
                yval = yoffset + iy * self.dy 
                zval = 0.0
                self.xpts += [xval, yval, zval]

        # define the element loads here
        self.elem_loads = []
        for iy in range(nye):
            for ix in range(nxe):
                xval = (ix+0.5)*self.dx
                yval = (iy+0.5)*self.dy
                self.elem_loads += [load_fcn(xval, yval)]

        self.nelem_per_comp = self.num_elements // self.ncomp

        # compute CSR rowp, cols (don't think I need BSR for simple plate in just bending, prob overkill and BSR not well supported in python anyways)
        self.rowp = None
        self.cols = None
        self.rows_scipy = None
        self._compute_CSR_pattern()

        # compute elem to component DV map
        self.elem_comp_map = None
        self._compute_elem_to_comp_map()

        # get the integer elem => CSR assembly map (for faster assembly)
        self.elem_to_csr_map = None
        self._compute_elem_to_CSR_assembly_map()

        # similar bc cols CSR map (for fast bc cols application, bc rows is easier)
        self.bc_cols_csr_map = None
        self._compute_bc_cols_map()

        # store EI=1 (un-scaled Kelem and felem for faster assembly)
        self.Kelem_nom = get_kelem(1.0, self.xscale, self.yscale)
        self.Kelem_nom_flat = self.Kelem_nom.flatten()
        self.felem_nom = get_felem(self.xscale, self.yscale)

    @classmethod
    def aluminum_unitsquare_trigload(
        cls,
        num_elements:int, # num elements, num_components must be perfect squares and divide evenly
        num_components:int,
        rho_KS:float=100.0, 
        qmag:float=2e-2,
        can_print:bool=False
    ):
        # main test case I'm using for snopt, FSD, INK comparison rn
        return cls(
            material=IsotropicMaterial.aluminum(),
            plate_fem_geom=PlateFemGeom.unit_square(num_elements, num_components),
            plate_loads=PlateLoads.game_of_life_trig_load(qmag=qmag),
            rho_KS=rho_KS,
            can_print=can_print,
            dense=False, # sparse is faster
            use_numba=True, # way faster assembly
        )

    @property
    def dof_conn(self):
        return np.array([[3*ind+_ for ind in elem_conn for _ in range(3)] for elem_conn in self.conn])
    
    def _compute_CSR_pattern(self):
        """jelper method upon construction compute the CSR matrix rowp, cols pattern (nofill) """
        import time
        t0 = time.time()
        if self.can_print: print("compute CSR pattern")

        if self.use_numba:
            # code found in _numba.py in plate folder
            nnz, rowp, cols, rows_scipy = _compute_CSR_pattern_numba(
                self.num_nodes, self.num_elements, np.array(self.conn)
            )
            self.nnz = nnz
            self.rowp = rowp
            self.cols = cols
            self.rows_scipy = rows_scipy

        else: # slower but verified version
            self.rowp = [0]
            self.rows_scipy = [] # needs row ind for each nz (only used to make the scipy CSR matrix)
            self.cols = []
            self.nnz = 0
            self.N = 3 * self.num_nodes
            for row in range(self.N):
                inode = row // 3
                all_attached_nodes = [_node for elem_conn in self.conn if inode in elem_conn for _node in elem_conn]
                all_attached_nodes = np.unique(np.array(all_attached_nodes))

                # num nonzeros on this row is 3 * num attached nodes
                nnz_in_row = 3 * all_attached_nodes.shape[0]
                self.nnz += nnz_in_row
                attached_dof = [3*_node + idof for _node in all_attached_nodes for idof in range(3)]
                self.rowp += [self.nnz]
                self.cols += list(attached_dof)
                self.rows_scipy += [row] * nnz_in_row

            # convert to np arrays
            self.rowp = np.array(self.rowp)
            self.cols = np.array(self.cols)
            self.rows_scipy = np.array(self.rows_scipy)

        dt = time.time() - t0
        if self.can_print: print(f"CSR pattern computed in {dt=:.4e} sec")

    def _compute_elem_to_comp_map(self):
        self.elem_comp_map = np.zeros((self.num_elements,), dtype=np.int32)

        for iy in range(self.nye):
            iy_comp = iy // self.nnyh
            for ix in range(self.nxe):
                ix_comp = ix // self.nnxh
                icomp = iy_comp * self.nxh + ix_comp
                ielem = iy * self.nxe + ix

                self.elem_comp_map[ielem] = icomp

    def _compute_elem_to_CSR_assembly_map(self):
        """upon construction pre-compute an integer map for the assembly"""
        import time
        t0 = time.time()
        dof_per_elem = 12

        
        if self.use_numba:
            
            # code found in _numba.py in plate folder
            self.elem_to_csr_map = _compute_elem_to_CSR_assembly_map_numba(
                self.num_elements, dof_per_elem, self.dof_conn, self.rowp, self.cols)

        else: # verified but slow version
            dof_per_elem2 = dof_per_elem**2 # 144 DOF per element
            self.elem_to_csr_map = np.zeros((self.num_elements, dof_per_elem2), dtype=np.int32)

            for ielem in range(self.num_elements): 
                local_conn = self.dof_conn[ielem]

                # scatter the Kelem to global Kelem (small for loops here)
                for idof_in_elem in range(dof_per_elem2):
                    elem_row = idof_in_elem // dof_per_elem
                    elem_col = idof_in_elem % dof_per_elem
                    global_row = local_conn[elem_row]
                    global_col = local_conn[elem_col]

                    # determine where these values go in the CSR data
                    this_csr_ind = None
                    for csr_ind in range(self.rowp[global_row], self.rowp[global_row+1]):
                        csr_col = self.cols[csr_ind]
                        if global_col == csr_col:
                            this_csr_ind = csr_ind

                    if this_csr_ind is None:
                        # TODO : does this affect performance?
                        raise AssertionError("didn't find CSR ind in constructing elem to CSR assembly map")
                    
                    # this_cp must be None to work here
                    self.elem_to_csr_map[ielem, idof_in_elem] = this_csr_ind

        dt = time.time() - t0
        if self.can_print: print(f"CSR assembly map computed in {dt=:.4e} sec")
        return
    
    def _compute_bc_cols_map(self):
        """on construction pre-compute for each bc, which rows does the bc col appear in (col bcs slower to apply without this)"""
        import time
        t0 = time.time() 

        
        if self.use_numba:

            # this code is found in _numba.py
            bc_counts = _compute_colbc_row_counts_numba(self.bcs, self.num_dof, self.rowp, self.cols)
            bc_offsets, colbc_rows = _compute_bc_cols_csr_map_numba(self.bcs, self.num_dof, self.rowp, self.cols, bc_counts)

            self.bc_cols_csr_offsets = bc_offsets
            self.bc_cols_csr_rows = colbc_rows

        else:
            self.bc_cols_csr_map = {bc : [] for bc in self.bcs}
            for bc in self.bcs:
                # get which rows include the bc col
                for glob_row in range(self.num_dof):
                    this_row_cols = self.cols[self.rowp[glob_row] : self.rowp[glob_row+1]]
                    if bc in this_row_cols:
                        self.bc_cols_csr_map[bc] += [glob_row]

        dt = time.time() - t0
        if self.can_print: print(f"CSR bc cols map computed in {dt=:.4e} sec")
        return

    """
    end of constructor or constructor helper methods
    ------------------------------------------------------
    start of general red-space solve utils section
        includes primal, adjoint methods used in both INK and SNOPT KS-aggreg
    ------------------------------------------------------
    """

    def get_helem_vec(self, hcomp_vec):
        return np.array([hcomp_vec[self.elem_comp_map[ielem]] for ielem in range(self.num_elements)])
    
    def _compute_mat_vec(self, hcomp_vec):
        helem_vec = self.get_helem_vec(hcomp_vec)
        if self._dense:
            self._compute_dense_mat_vec(helem_vec)
        else:
            self._compute_sparse_mat_vec(helem_vec)
    
    # @njit # python compilation for faster for loop here (if doesn't work may have to put it on )
    def _compute_dense_mat_vec(self, helem_vec):
        """dense Kmat and RHS assembly"""
        # copy states out
        E = self.E
        dof_per_node = self.dof_per_node

        # compute Kelem without EI scaling
        Kelem_nom = get_kelem(1.0, self.xscale, self.yscale)
        felem_nom = get_felem(self.xscale, self.yscale)

        dof_per_node = self.dof_per_node
        num_dof = dof_per_node * self.num_nodes
        Kmat = np.zeros((num_dof, num_dof), dtype=helem_vec.dtype)
        force = np.zeros((num_dof,))

        for ielem in range(self.num_elements): 
            local_conn = np.array(self.dof_conn[ielem])
            D = E * helem_vec[ielem]**3 / 12.0 / (1-self.nu**2)
            np.add.at(Kmat, (local_conn[:,None], local_conn[None,:]), D * Kelem_nom)

            q = self.elem_loads[ielem]
            # print(f"{local_conn.shape=} {q.shape=} {felem_nom.shape=}")
            np.add.at(force, local_conn, q * felem_nom)

        # now apply simply supported BCs
        bcs = self.bcs

        # apply dirichlet w=0 BCs
        for bc in bcs:
            Kmat[bc,:] = 0.0
            Kmat[:, bc] = 0.0
            Kmat[bc, bc] = 1.0

        # zero out bcs in vector
        for bc in bcs:
            force[bc] = 0.0

        # store in object
        self.Kmat = Kmat
        self.force = force
        return Kmat, force

    def _compute_sparse_mat_vec(self, helem_vec):
        """sparse CSR matrix assembly (and RHS)"""
        import time
        t0 = time.time() # TODO : remove timing later.. after temp debugging it

        # to hold the csr data (for making scipy CSR Kmat at end of method)
        csr_data = np.zeros((self.nnz), dtype=helem_vec.dtype)       
        dof_per_node = self.dof_per_node 
        num_dof = dof_per_node * self.num_nodes
        global_force = np.zeros((num_dof,))

        t1 = time.time()
        dt1 = t1 - t0
        # print(f"assembly init phase {dt1=:.4f}")

         # seems to work now
        # self.use_numba = False
        if self.use_numba:

            kelem_scales = self.E * helem_vec**3 / 12.0 / (1 - self.nu**2) # = D the flexural modulus
            felem_scales = np.array(self.elem_loads)
            _helper_numba_assembly_serial(
                self.num_elements, self.elem_to_csr_map, self.dof_conn,
                csr_data, self.Kelem_nom_flat, kelem_scales,
                global_force, self.felem_nom, felem_scales)
            # self._helper_numba_assembly_parallel
            # for some reason _helper_numba_assembly_serial is actually faster than _helper_numba_assembly_parallel
            # think maybe multithreading isn't configured the best in my conda env? no worries, it's very fast with parallel=False
            
        else: # the slower version but it's verified

            for ielem in range(self.num_elements): 
                local_conn = self.dof_conn[ielem]
                # local_node_conn = self.conn[ielem]

                # get true Kelem (now scaled by D with the element thickness)
                D = self.E * helem_vec[ielem]**3 / 12.0 / (1-self.nu**2)
                Kelem = D * self.Kelem_nom_flat

                # add Kelem into csr data using the elem_to_csr map
                loc_elem_to_csr_map = self.elem_to_csr_map[ielem,:]
                csr_data[loc_elem_to_csr_map] += Kelem

            # add the forces (this should be pretty fast.. could still parallel maybe too
            for ielem in range(self.num_elements):
                local_conn = self.dof_conn[ielem]
                q = self.elem_loads[ielem]
                np.add.at(global_force, local_conn, q * self.felem_nom)

        # now apply simply supported BCs
        bcs = self.bcs

        # apply dirichlet w=0 BCs
        # first to zero out rows for each bc (except diag entry)
        for bc in bcs:
            glob_row = bc
            for csr_ind in range(self.rowp[glob_row], self.rowp[glob_row+1]):
                glob_col = self.cols[csr_ind]
                csr_data[csr_ind] = glob_col == glob_row
        
        # now zero out cols
        for i_bc,bc in enumerate(bcs):
            # use the bc_cols_map to get which rows include this bc col (otherwise we need like a quadruple for loop and it's very slow 
            # (so use pre-computed map on construction)
            # included_rows = self.bc_cols_csr_map[bc] # without bc numba
            included_rows = self.bc_cols_csr_rows[self.bc_cols_csr_offsets[i_bc] : self.bc_cols_csr_offsets[i_bc + 1]]

            for glob_row in included_rows:
                for csr_ind in range(self.rowp[glob_row], self.rowp[glob_row+1]):
                    glob_col = self.cols[csr_ind]
                    if glob_col == bc:
                        csr_data[csr_ind] = glob_col == glob_row

        # zero out bcs in force vector
        global_force[bcs] = 0.0

        # store in object
        self.Kmat = sp.sparse.csr_matrix((csr_data, (self.rows_scipy, self.cols)), shape=(num_dof, num_dof))
        self.force = global_force

        return
    
    def helem_to_hcomp_vec(self, dhelem_vec):
        dhcomp_vec = np.zeros((self.ncomp,))
        for ielem in range(self.num_elements):
            icomp = self.elem_comp_map[ielem]
            dhcomp_vec[icomp] += dhelem_vec[ielem]
        return dhcomp_vec

    def solve_forward(self, hcomp_vec):
        # now solve the linear system
        import time
        t0 = time.time()
        self._compute_mat_vec(hcomp_vec)
        # helem_vec = self.get_helem_vec(hcomp_vec)
        # helem_vec = np.ones(self.num_elements)
        # self._compute_sparse_mat_vec(helem_vec)
        t1 = time.time()

        if self._dense:
            self.u = np.linalg.solve(self.Kmat, self.force)
        else:
            # direct sparse linear solve 
            self.u = sp.sparse.linalg.spsolve(self.Kmat, self.force)
        t2 = time.time()

        dt1 = t1 - t0
        dt2 = t2 - t1
        dt_tot = t2 - t0
        if self.can_print: print(f"solve_forward: assembly in {dt1:.4e}, linear solve in {dt2:.4e}, total {dt_tot:.4e}")

        return self.u

    def solve_adjoint(self, hcomp_vec):
        # now solve the adjoint system
        self._compute_mat_vec(hcomp_vec)

        # adjoint solve only required for stress function (mass is non-adjoint no state var dependence)
        KT = self.Kmat.T
        adj_rhs = self._get_dkstot_du(hcomp_vec)

        if self._dense:
            self.psis = np.linalg.solve(KT, -adj_rhs)
        else:
            # direct sparse linear solve 
            self.psis = sp.sparse.linalg.spsolve(KT, -adj_rhs)
        return self.psis
    
    def _get_elem_bending_moments(self, helem_vec):
        """helper function (for debugging to check whether the bending moments are relatively constant with uniform thick scalings)"""

        # it's a slow function not using numba rn
        moments_vec = np.zeros((self.num_elements, 3), dtype=helem_vec.dtype)
        for ielem in range(self.num_elements):
            D = self.E * helem_vec[ielem]**3 / 12.0 / (1.0 - self.nu**2)

            local_conn = self.dof_conn[ielem]
            local_disp = self.u[local_conn]
            dx2 = dy2 = dxy = 0
            for ibasis in range(12):
                xi = 0.0; eta = 0.0
                hessians = get_hessians(ibasis, xi, eta, self.dx, self.dy)
                dx2 += hessians[0] * local_disp[ibasis]
                dy2 += hessians[1] * local_disp[ibasis]
                dxy += hessians[2] * local_disp[ibasis]
            
            # compute bending moments
            Mxx = -D * (dx2 + self.nu * dy2)
            Myy = -D * (dy2 + self.nu * dx2)
            Mxy = -D * (1-self.nu) * dxy

            moments_vec[ielem,:] = np.array([Mxx, Myy, Mxy])
        return moments_vec

    def _get_elem_fails(self, helem_vec):
        """helper function that computes the failure index in each element"""

        if self.use_numba:
            # pre-compute hessian data
            hess_basis = np.zeros((12, 3))
            for ibasis in range(12):
                for idof in range(3):
                    hess_basis[ibasis, idof] = get_hessians(ibasis, xi=0.0, eta=0.0, xscale=self.dx, yscale=self.dy)[idof]
            Dvec = self.E * helem_vec**3 / 12.0 / (1 - self.nu**2)

            # this code is found in _numba.py
            elem_fails = _get_elem_fails_numba(
                self.num_elements, Dvec, self.dof_conn, self.u,
                hess_basis, self.nu, helem_vec, self.ys)

        else:
            elem_fails = np.zeros((self.num_elements,), dtype=helem_vec.dtype)
            for ielem in range(self.num_elements):
                D = self.E * helem_vec[ielem]**3 / 12.0 / (1.0 - self.nu**2)

                local_conn = self.dof_conn[ielem]
                local_disp = self.u[local_conn]
                dx2 = dy2 = dxy = 0
                for ibasis in range(12):
                    xi = 0.0; eta = 0.0
                    hessians = get_hessians(ibasis, xi, eta, self.dx, self.dy)
                    dx2 += hessians[0] * local_disp[ibasis]
                    dy2 += hessians[1] * local_disp[ibasis]
                    dxy += hessians[2] * local_disp[ibasis]
                
                # compute bending moments
                Mxx = -D * (dx2 + self.nu * dy2)
                Myy = -D * (dy2 + self.nu * dx2)
                Mxy = -D * (1-self.nu) * dxy

                I = helem_vec[ielem]**3/12.0
                z = helem_vec[ielem]/2

                # compute the 2d stresses
                sxx = Mxx * z / I
                syy = Myy * z / I
                sxy = Mxy * z / I

                # now compute von mises stress
                vm_stress = np.sqrt(sxx**2 + syy**2 - sxx * syy + 3 * sxy**2)
                nd_stress = vm_stress / self.ys
                elem_fails[ielem] = nd_stress
        return elem_fails
    
    def _get_elem_fails_DVsens(self, delem_fails, helem_vec, elem_fails):
        dhelem_vec = [0.0] * self.num_elements
        for ielem in range(self.num_elements):           
            # shouldn't the coefficient be -2.0 here? for some reason it works with -1.0 
            dhelem_vec[ielem] += elem_fails[ielem] * 1.0 / helem_vec[ielem] * delem_fails[ielem]
        return np.array(self.helem_to_hcomp_vec(dhelem_vec))
    
    def _get_elem_fails_SVsens(self, delem_fails, helem_vec):
        """generic helper method to compute dfail/du given the df/dfail backproped partials
           this allows this method to be modular for one overall KSfail (SNOPT) vs KSfail by component (INK)"""
        
        if self.use_numba:
            # pre-compute hessian data
            hess_basis = np.zeros((12, 3))
            for ibasis in range(12):
                for idof in range(3):
                    hess_basis[ibasis, idof] = get_hessians(ibasis, xi=0.0, eta=0.0, xscale=self.dx, yscale=self.dy)[idof]
            Dvec = self.E * helem_vec**3 / 12.0 / (1 - self.nu**2)
            du_global = _get_elem_fails_SVsens_numba(
                self.num_dof, self.num_elements, Dvec, self.dof_conn, self.u,
                hess_basis, self.nu, helem_vec, self.ys, delem_fails)
            
        else:
            du_global = np.zeros((self.num_dof,))
            nu = self.nu
            for ielem in range(self.num_elements):
                D = self.E * helem_vec[ielem]**3 / 12.0 / (1.0 - nu**2)

                local_conn = self.dof_conn[ielem]
                local_disp = self.u[local_conn]
                dx2 = dy2 = dxy = 0
                dx2_du_vec = []
                dy2_du_vec = []
                dxy_du_vec = []
                for ibasis in range(12):
                    xi = 0.0; eta = 0.0
                    hessians = get_hessians(ibasis, xi, eta, self.dx, self.dy)
                    dx2 += hessians[0] * local_disp[ibasis]
                    dy2 += hessians[1] * local_disp[ibasis]
                    dxy += hessians[2] * local_disp[ibasis]

                    dx2_du_vec += [hessians[0]]
                    dy2_du_vec += [hessians[1]]
                    dxy_du_vec += [hessians[2]]
                
                # compute bending moments
                Mxx = -D * (dx2 + nu * dy2)
                Myy = -D * (dy2 + nu * dx2)
                Mxy = -D * (1-nu) * dxy

                I = helem_vec[ielem]**3/12.0
                z = helem_vec[ielem]/2

                # compute the 2d stresses
                sxx = Mxx * z / I
                syy = Myy * z / I
                sxy = Mxy * z / I

                # now compute von mises stress
                vm_stress = np.sqrt(sxx**2 + syy**2 - sxx * syy + 3 * sxy**2)
                # nd_stress = vm_stress / self.ys

                # now backprop the stress derivatives here
                dvm_dsxx = (2 * sxx - syy) / 2.0 / vm_stress / self.ys
                dvm_dsyy = (2 * syy - sxx) / 2.0 / vm_stress / self.ys
                dvm_dsxy = (6 * sxy) / 2.0 / vm_stress / self.ys

                # backprop through the bending moments
                dvm_dMxx = dvm_dsxx * z / I
                dvm_dMyy = dvm_dsyy * z / I
                dvm_dMxy = dvm_dsxy * z / I

                # backprop through to the hessian derivatives
                dvm_dx2 = -D * (dvm_dMxx + nu * dvm_dMyy)
                dvm_dy2 = -D * (dvm_dMyy + nu * dvm_dMxx)
                dvm_dxy = -D * (1-nu) * dvm_dMxy

                du_vec = dvm_dx2 * np.array(dx2_du_vec) + \
                        dvm_dy2 * np.array(dy2_du_vec) + \
                        dvm_dxy * np.array(dxy_du_vec)
                du_global[local_conn] += du_vec * delem_fails[ielem]
        return du_global
    
    def _get_elem_fails_SVsens_prod(self, delem_fails, phi_u, helem_vec):
        """generic helper method to compute dfail_i / du_j * phi_j (no backprop, here phi is one testvec)
             with output is basically (dc/du) * phi_u => units and dim of c  (for ROM code)"""
        
        if self.use_numba:
            # pre-compute hessian data
            hess_basis = np.zeros((12, 3))
            for ibasis in range(12):
                for idof in range(3):
                    hess_basis[ibasis, idof] = get_hessians(ibasis, xi=0.0, eta=0.0, xscale=self.dx, yscale=self.dy)[idof]
            Dvec = self.E * helem_vec**3 / 12.0 / (1 - self.nu**2)
            dc_elem_vec = _get_elem_fails_SVsens_prod_numba(
                self.num_dof, self.num_elements, Dvec, self.dof_conn, self.u,
                hess_basis, self.nu, helem_vec, self.ys, delem_fails, phi_u)
            
        else:
            raise AssertionError("_get_elem_fails_SVsens_prod code not written without numba")
        return dc_elem_vec
    
    def _compute_psi_Ku_prod(self, hcomp_vec):
        # update kmat, compute psi^T * fint = psi^T * Ku
        # u and psi optional inputs only for full space case
        self._compute_mat_vec(hcomp_vec)
        resid = self.Kmat.dot(self.u)
        return np.dot(self.psis, resid)
    
    def _compute_adj_dRdx(self, hcomp_vec):
        """compute the adj resid product psi^T dR/dx (partial deriv on R)"""
        helem_vec = self.get_helem_vec(hcomp_vec)
        
        if self.use_numba:
            Dvec = self.E * helem_vec**3 / 12.0 / (1 - self.nu**2)

            # code in _numba.py
            dthick_elem = _compute_adj_dRdx_numba(self.num_elements, self.Kelem_nom, self.dof_conn,
                                    self.psis, self.u, Dvec, helem_vec, self.bcs)

        else:
            dthick_elem = [0.0]*self.num_elements

            # compute Kelem without EI scaling
            Kelem_nom = get_kelem(1.0, self.xscale, self.yscale)

            # compute the gradient at the element level so more efficient
            for ielem in range(self.num_elements):
                
                # get local psi vector
                local_conn = np.array(self.dof_conn[ielem])
                psi_local = self.psis[local_conn]

                # get local disp vector
                u_local = self.u[local_conn]

                # compute local Kelem thickness derivative
                D = self.E * helem_vec[ielem]**3 / 12.0 / (1-self.nu**2)
                dKelem = Kelem_nom * D * 3 / helem_vec[ielem]

                # apply local bcs to dKelem matrix
                # start_node = local_node_conn[0]
                for local_dof,global_dof in enumerate(local_conn):
                    if global_dof in self.bcs:
                        dKelem[local_dof,:] = 0.0
                        dKelem[:,local_dof] = 0.0
                        # dKelem[local_dof, local_dof] = 0.0
                
                # now compute quadratic product for gradient
                dthick_elem[ielem] = np.dot(psi_local, np.dot(dKelem, u_local))

        # dt = time.time() - t0
        # print(f"psi^T dR/dx in {dt:.4e} sec")
        # then convert from element thick to comp thick gradients
        return np.array(self.helem_to_hcomp_vec(dthick_elem))
    
    def get_mass(self, hcomp_vec):
        helem_vec = self.get_helem_vec(hcomp_vec)
        # copy states out
        a = self.a; b = self.b; rho = self.rho

        # compute mass
        mass_vec = [a * b * helem_vec[ielem] * rho for ielem in range(self.num_elements)]
        mass = sum(mass_vec)
        return mass
    
    def get_mass_gradient(self):
        # copy states out
        a = self.a; b = self.b; rho = self.rho
        dmass = np.array([rho * a * b] * self.num_elements)
        dmass_red = self.helem_to_hcomp_vec(dmass)
        return dmass_red
    
    def _ks_func(self, vec):
        """helper generic ks func on a vector"""
        rhoKS = self.rho_KS
        true_max = np.max(vec)
        ks_max = true_max + 1.0 / rhoKS * np.log(np.sum(np.exp(rhoKS * (vec - true_max))))
        return ks_max
    
    def _ks_grad(self, vec):
        """helper generic derivative through the generic ks func"""
        rhoKS = self.rho_KS
        true_max = np.max(vec)

        dvec = np.exp(rhoKS * (vec - true_max))
        dvec /= np.sum(dvec)
        return dvec
    
    """
    -------------------------------------------------------------------
        end of general utils for any method (like primal, adjoint, elem_fails)
    -------------------------------------------------------------------
    start of SNOPT KS-aggregation section, uses one overall ksfail constraint (not by component)
    -------------------------------------------------------------------
    """
    
    def get_ks_fail(self, hcomp_vec):
        """KS-aggregate failure index of the whole structure for SNOPT, ks-aggreg approach"""
        helem_vec = self.get_helem_vec(hcomp_vec)
        elem_fails = self._get_elem_fails(helem_vec)
        ks_fail = self._ks_func(elem_fails)
        return ks_fail
    
    def _get_dkstot_du(self, hcomp_vec):
        """SV partial derivs for KS-aggregate fail index of whole structure"""
        helem_vec = self.get_helem_vec(hcomp_vec)
        elem_fails = self._get_elem_fails(helem_vec)
        dfails = self._ks_grad(elem_fails)
        du_global = self._get_elem_fails_SVsens(dfails, helem_vec)
        return du_global
    
    def _get_dkstot_dx(self, hcomp_vec):
        """DV partial derivatives for KSfail of whole structure, for SNOPT KS-aggreg"""        
        # backprop from scalar stress to stress vec 
        helem_vec = self.get_helem_vec(hcomp_vec)
        elem_fails = self._get_elem_fails(helem_vec)
        dfails = self._ks_grad(elem_fails)
        dhcomp_vec = self._get_elem_fails_DVsens(dfails, helem_vec, elem_fails)
        return dhcomp_vec

    def get_functions(self, hcomp_vec):
        """get functions for SNOPT with one aggreg ksfail"""
        mass = self.get_mass(hcomp_vec)
        ksfail = self.get_ks_fail(hcomp_vec)
        return np.array([mass, ksfail])
    
    def get_function_gradients(self, hcomp_vec):
        """get function gradients for SNOPT with one aggreg ksfail"""
        dmass = self.get_mass_gradient()

        # TODO : check shapes here..
        dks_fail = self._get_dkstot_dx(hcomp_vec)
        dks_fail += self._compute_adj_dRdx(hcomp_vec)

        return np.array([dmass, dks_fail])
    
    """
    ------------------------------------------------
    FSD methods here
    ------------------------------------------------
    """
    def _get_comp_true_fails(self, elem_fails):
        """for FSD method (true max in each component of fails, no KS) """
        # can be sensitive to stress-singularities with this approach
        comp_fails = np.zeros((self.ncomp,), dtype=elem_fails.dtype)

        for ielem in range(self.num_elements):
            icomp = self.elem_comp_map[ielem]
            comp_fails[icomp] = np.max([comp_fails[icomp], elem_fails[ielem]])
        # print(f"{stress_red=}")
        return comp_fails
    
    def get_comp_true_fails(self, hcomp_vec):
        """
        for FSD Method
        """
        helem_vec = self.get_helem_vec(hcomp_vec)
        elem_fails = self._get_elem_fails(helem_vec)
        true_comp_fails = self._get_comp_true_fails(elem_fails)
        return true_comp_fails
    
    """
        end of FSD methods
    ------------------------------------------------------
    start of plot utils section
    ------------------------------------------------------
    """

    @property
    def xvec(self) -> list:
        return [i*self.dx for i in range(self.num_nodes)]    
    
    def _plot_field_on_ax(self, field, ax, log_scale:bool=True, cmap='viridis', elem_to_node_convert:bool=True):
        """helper method to plot a scalar field as a contour"""
        x = self.xpts[0::3]
        y = self.xpts[1::3]

        X = np.reshape(x, (self.nnx, self.nny))
        Y = np.reshape(y, (self.nnx, self.nny))
        if elem_to_node_convert:
            H = np.reshape(field, (self.nxe, self.nye))
            H = self._elem_to_node_arr(H)
        else: 
            H = np.reshape(field, (self.nnx, self.nny))

        if log_scale: 
            H = np.log10(H)
        
        # 'seismic', 'twilight', 'magma', 'plasma', 'cividis'
        cmaps = ['viridis', 'turbo', 'RdBu_r']
        if isinstance(cmap, int): # allows you to put an int in
            cmap = cmaps[cmap]
        cf = ax.contourf(X, Y, H, cmap=cmap, levels=100)

        self._plot_dv_grid(ax) # plot the DV boundaries on the plot

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        return cf

    def plot_disp(self, filename:str=None, figsize=(10, 7), dpi:int=100, format_str="%.1e"):
        """plot the transverse disp w(x,y)"""

        plot_init()
        fig, ax = plt.subplots(figsize=figsize)
        cf = self._plot_field_on_ax(
            field=self.u[0::3],
            ax=ax,
            log_scale=False,
            cmap='turbo',
            elem_to_node_convert=False,
        )
        cb = fig.colorbar(cf, ax=ax, format=FormatStrFormatter(format_str)) # can change string format as needed
        cb.set_label("w(x,y)")
        fig.set_dpi(dpi)

        if filename is None:
            plt.show() 
        else:
            plt.savefig(filename, dpi=dpi)    
            plt.close('all')

    def plot_thickness(self, hcomp_vec:np.ndarray, filename:str=None, figsize=(10, 7), dpi:int=100, format_str="%.2f", log_scale:bool=True):
        """plot the component thickness DVs"""
        # convert from component DVs to each element (for plotting)
        helem_vec = self.get_helem_vec(hcomp_vec)

        plot_init()
        fig, ax = plt.subplots(figsize=figsize)
        cf = self._plot_field_on_ax(
            field=helem_vec,
            ax=ax,
            log_scale=log_scale,
            cmap='viridis',
            elem_to_node_convert=True,
        )
        cb = fig.colorbar(cf, ax=ax, format=FormatStrFormatter(format_str)) # can change string format as needed
        cb.set_label("log10(thick DV)")
        fig.set_dpi(dpi)

        if filename is None:
            plt.show() 
        else:
            plt.savefig(filename, dpi=dpi)    
            plt.close('all')

    def plot_failure_index(self, hcomp_vec:np.ndarray, filename=None, 
                           solve_forward:bool=True, figsize=(10, 7), dpi:int=100,
                           format_str:str="%.2f"):
        """plot the failure index for the given thickness DVs (computes it first)"""

        # compute element failure indexes first
        if solve_forward: self.solve_forward(hcomp_vec)
        helem_vec = self.get_helem_vec(hcomp_vec)
        elem_failure_indexes = self._get_elem_fails(helem_vec)

        plot_init()
        fig, ax = plt.subplots(figsize=figsize)
        cf = self._plot_field_on_ax(
            field=elem_failure_indexes,
            ax=ax,
            log_scale=False,
            cmap='RdBu_r',
            elem_to_node_convert=True,
        )
        cb = fig.colorbar(cf, ax=ax, format=FormatStrFormatter(format_str)) # can change string format as needed
        cb.set_label("log10(fail index)")
        fig.set_dpi(dpi)

        if filename is None:
            plt.show() 
        else:
            plt.savefig(filename, dpi=dpi)    
            plt.close('all')

    def plot_component_failure_index(self, hcomp_vec:np.ndarray, filename=None, 
                           solve_forward:bool=True, figsize=(10, 7), dpi:int=100,
                           format_str:str="%.2f"):
        """plot the failure index for the given thickness DVs (computes it first)"""

        # compute element failure indexes first
        if solve_forward: self.solve_forward(hcomp_vec)
        comp_fails = self.get_comp_true_fails(hcomp_vec)
        elem_comp_fails = self.get_helem_vec(comp_fails) # make each element in comp just have that comp fail

        plot_init()
        fig, ax = plt.subplots(figsize=figsize)
        cf = self._plot_field_on_ax(
            field=elem_comp_fails,
            ax=ax,
            log_scale=True,
            cmap='RdBu_r',
            elem_to_node_convert=True,
        )
        cb = fig.colorbar(cf, ax=ax, format=FormatStrFormatter(format_str)) # can change string format as needed
        cb.set_label("log10(comp max fail index)")
        fig.set_dpi(dpi)

        if filename is None:
            plt.show() 
        else:
            plt.savefig(filename, dpi=dpi)    
            plt.close('all')

    def _elem_to_node_arr(self, arr):
        """convert """

        # strategy #2
        from scipy.interpolate import RegularGridInterpolator
        xorig = np.linspace(-self.a/2, self.a/2, self.nxe)
        yorig = np.linspace(-self.b/2, self.b/2, self.nye)
        # print(f"{xorig.shape=} {yorig.shape=} {arr.shape=}")
        # 'linear', 'cubic'
        interp_func = RegularGridInterpolator((xorig, yorig), arr, method='linear')

        xnew = np.linspace(-self.a/2, self.a/2, self.nnx)
        ynew = np.linspace(-self.b/2, self.b/2, self.nny)
        grid_points = np.array(np.meshgrid(xnew, ynew)).T.reshape(-1, 2)
        arr_v2_vec = interp_func(grid_points)
        arr_v2 = arr_v2_vec.reshape(self.nnx, self.nny)
        # print(f'{arr_v2.shape=}')

        return arr_v2
    
    def _plot_dv_grid(self, ax):
        """plot black boundaries for each DV region"""
        grid_rows, grid_cols = self.nxh, self.nyh # the actual DV regions
        row_step = self.a / grid_rows
        col_step = self.b / grid_cols

        for i in range(grid_rows):
            for j in range(grid_cols):
                ax.add_patch(plt.Rectangle((-self.a/2 + j * col_step, -self.b/2 + i * row_step), col_step, row_step, 
                                        edgecolor='white', facecolor='none', linewidth=1))

    """
    ------------------------------------------------------
        end of plot utils section
    ------------------------------------------------------
    """

    def finite_diff_test(self, hcomp_vec, h:float=1e-6):
        # central finite diff test for all func grads used in regular KS-aggregated opt (one overall KS failure index like in SNOPT)
        p = np.random.rand(self.ncomp)
        hcomp_vec = np.array(hcomp_vec)

        # adjoint value
        self.solve_forward(hcomp_vec)
        self.solve_adjoint(hcomp_vec)
        # funcs = self.get_functions(hcomp_vec)
        func_grads = self.get_function_gradients(hcomp_vec)
        
        nfunc = 2
        adj_prods = [
            np.dot(func_grads[ifunc,:], p) for ifunc in range(nfunc)
        ]

        # FD [rpdict]
        hcomp_vec2 = hcomp_vec + h * p
        self.solve_forward(hcomp_vec2)
        funcs2 = self.get_functions(hcomp_vec2)

        hcomp_vecn1 = hcomp_vec - h * p
        self.solve_forward(hcomp_vecn1)
        funcsn1 = self.get_functions(hcomp_vecn1)
        fd_prods = [
            (funcs2[i] - funcsn1[i]) / 2 / h for i in range(len(funcs2))
        ]

        print(f"{adj_prods=}")
        print(f"{fd_prods=}")