import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import svds, norm
from data import *

def SVT(omega, vec_data, dims, max_iter=1e4, tol=1e-4, sparse_type='n'):
    delta = 1.2 * np.prod(dims) / omega.shape[0]
    tau = 5 * np.max(np.array(dims))
    l = 5    
    norm_of_data = np.linalg.norm(vec_data)
    mat_data = reconstruct_matrix(vec_data, omega, dims, sparse_type)
    _, second_norm, _ = svds(mat_data, 1)
    k0 = np.ceil(1. * tau / delta / second_norm[0])
    Y = k0 * delta * mat_data
    r = 0
    num_iter = 0
    min_size = np.min(np.array(dims)) - 1
    while num_iter < max_iter:
        s = np.min([r + 1, min_size])
        sigma = tau + 1
        rpt_inside = 0
        while (sigma > tau) & (rpt_inside <= 1):
            U, S, V = svds(Y, s)
            if np.sum(S == 0) > 0:
                break
            sigma = S[0]
            s = np.min([s+l, min_size])
            if s == min_size:
                rpt_inside += 1
        r = np.sum(S > tau)
        U = U[:, -r:]
        S = S[-r:] - tau
        V = V[-r:, :]
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
        
        X = reconstruct_matrix(X, omega, dims, sparse_type)
        Y = Y + delta * (mat_data - X)
        num_iter += 1
    
    return X_opt