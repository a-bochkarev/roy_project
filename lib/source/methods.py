import numpy as np
import data

def SVT(omega, data, dims, max_iter=1e4, tol=1e-4):
    data = reconstruct_matrix(data, omega, dims)
    delta = 1.2 * np.prod(dims) / omega.shape[0]
    tau = 5 * np.max(np.array(dims))
    l = 5    
    norm_of_data = np.linalg.norm(data, 'fro')
    k0 = get_k(data, delta, tau)
    Y = k0 * delta * data
    r = 0
    num_iter = 0
    while num_iter < max_iter:
        s = r + 1
        sigma = tau + 1
        while sigma > tau:            
            U, S, V = np.linalg.svd(Y)
            U = U[:, :s]
            S = S[:s]
            V = V[:s, :]
            sigma = S[-1]
            s += l
        r = np.sum(S > tau)
        U = U[:, :r].copy()
        S = S[:r].copy()
        S -= tau
        V = V[:r, :].copy()
        X = U.dot(np.diag(S).dot(V))
        X_opt = X.copy()
        
        X = get_sampling_matrix(X, omega)
        if np.linalg.norm(X - data, 'fro') < tol * norm_of_data:
            break
        
        Y = get_sampling_matrix(Y, omega)
        Y = Y + delta * (data - X)
        num_iter += 1
    
    return X_opt