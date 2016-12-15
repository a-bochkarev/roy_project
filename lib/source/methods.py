import numpy as np
import sys
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import svds, norm, eigs

sys.path.append('./SoftImpute/')
sys.path.append('./SVT/')
sys.path.append('./SVP/')
sys.path.append('./RISMF/')
sys.path.append('./lib/source/')
from data import *
from SVP import *
from SVT import *
from RISMF import *


from SoftImpute import SoftImpute

def get_completion(Omega, m, r, dims, method, 
                   max_iter=1e4, tol=1e-4, sparse_type='n', verbose=False):
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
        return SVT(r, Omega, m, dims, max_iter, tol, sparse_type)
    elif method == 'SoftImpute':
        solver = SoftImpute(max_rank=r, convergence_threshold=tol, verbose=verbose, 
            max_iters=max_iter)
        return solver.complete(Omega, m, dims)
    if method == 'SVP':
        return SVP(Omega=Omega, vec_data=m, dims=dims, k=r, 
            n_iter=max_iter, tol=tol, tau=2.5)
    elif method == 'RISMF' :
        return  RISMF(Omega, m, dims ,learningRate = 0.1, regularizedFactor = 0.1 , K = 1, percentageTrainingSet = 0.1, nbIterMax = 100)
    else:
        print 'This method is not implemented'
