import numpy as np
import sys
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import svds, norm, eigs
from data import *

sys.path.append('./../../SoftImpute/')

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
        return SVT(Omega, m, dims)
    if method == 'SoftImpute':
        return SI(Omega, m, r, dims)
    else:
        print 'This method is not implemented'

def SVT(omega, vec_data, dims, max_iter=1e4, tol=1e-4, sparse_type='n'):
    delta = 1.2 * np.prod(dims) / omega.shape[0]
    tau = 5 * np.max(np.array(dims))
    l = 5    
    norm_of_data = np.linalg.norm(vec_data)
    mat_data = reconstruct_matrix(vec_data, omega, dims, sparse_type)
    _, second_norm, _ = svds(mat_data, 1)
    k0 = np.ceil(1. * tau / delta / second_norm[0])
    Y = k0 * delta * vec_data
    r = 0
    num_iter = 0
    while num_iter < max_iter:
        s = r + 1
        sigma = tau + 1
        Y = reconstruct_matrix(Y, omega, dims, sparse_type)
        while sigma > tau:
            U, S, V = svds(Y, s)
            sigma = S[0]
            s += l
        r = np.sum(S > tau)
        U = U[:, -r:].copy()
        S = S[-r:].copy()
        S -= tau
        V = V[-r:, :].copy()
        if sparse_type == 'n':
            X = U.dot(np.diag(S).dot(V))
        elif sparse_type == 'y':
            U = csr_matrix(U)
            S = csr_matrix(np.diag(S))
            V = csr_matrix(V)
            X = U.dot(S.dot(V))
        X_opt = X.copy()
        
        X = get_sampling_vector(X, omega)
        if np.linalg.norm(X - vec_data) < tol * norm_of_data:
            break
        
        Y = get_sampling_vector(Y, omega)
        Y = Y + delta * (vec_data - X)
        num_iter += 1
    
    return X_opt

def SI(Omega, m, r, dims):
    solver = SoftImpute(max_rank=r, convergence_threshold=1e-3, verbose=False, max_iters=100)
    X_incomplete = np.ones(dims)*np.nan
    for i in range(Omega.shape[0]):
        X_incomplete[Omega[i][0], Omega[i][1]] = m[i]

    X_filled = solver.complete(X_incomplete)
    return X_filled