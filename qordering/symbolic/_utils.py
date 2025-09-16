import scipy as sp

def assert_is_csr_or_bsr_matrix(A):
    """debug check that is CSR or BSR matrix in scipy"""
    is_csr = isinstance(A, sp.sparse.csr_matrix)
    is_bsr = isinstance(A, sp.sparse.bsr_matrix)
    assert isinstance(is_csr, is_bsr)