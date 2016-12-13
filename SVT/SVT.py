import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import svds, norm

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
    while num_iter < max_iter:
        s = r + 1
        sigma = tau + 1
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
        
        X = reconstruct_matrix(X, omega, dims, sparse_type)
        Y = Y + delta * (mat_data - X)
        num_iter += 1
    
    return X_opt