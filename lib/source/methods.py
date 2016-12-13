import numpy as np
import sys
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import svds, norm, eigs
from data import *

sys.path.append('./../../SoftImpute/')
sys.path.append('./../../SVT/')
from SVT import *


from SoftImpute import SoftImpute, BiScaler

def get_completion(Omega, m, r, dims, method, 
                   max_iter=1e4, tol=1e-4, sparse_type='n'):
    """
    Matrix completion by given methods
    Input:
    Omega : array of sampling indices
    m : array like with sampling elements
    method : str with the name of completion method
    dims : size of the matrix
    r : max rank
    Output:
    X : completed matrix
    rank - rank of X
    """
    if method == 'SVT':
        return SVT(Omega, m, dims, max_iter, tol, sparse_type)
    if method == 'SoftImpute':
        solver = SoftImpute(max_rank=r, convergence_threshold=tol, verbose=False, 
            max_iters=max_iter, normalizer=BiScaler(verbose=False))
        return solver.complete(Omega, m, dims)
    else:
        print 'This method is not implemented'

def SI(Omega, m, r, dims, max_iter=1e4, tol=1e-4):
    solver = SoftImpute(max_rank=r, convergence_threshold=tol, verbose=False, max_iters=max_iter)
    X_incomplete = np.ones(dims)*np.nan
    for i in range(Omega.shape[0]):
        X_incomplete[Omega[i][0], Omega[i][1]] = m[i]

    X_filled = solver.complete(X_incomplete)
    return X_filled