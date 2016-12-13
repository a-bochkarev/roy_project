import numpy as np
from scipy.sparse.linalg import svds

def SVP(Omega, vec_data, dims, k=5, tol=1e-3, maxtol=1000, tau=7.5, n_iter=100, verbose=False):
    X = np.zeros((dims[0], dims[1]))
    
    prev_rmse = np.inf
    t = 1
    while(t <= n_iter):
        X_Omega = get_sampling_vector(X, Omega)

        rmse = RMSE(X_Omega, vec_data)
        if rmse < tol:
            return X
        if rmse - prev_rmse > maxtol:
            print 'Divergence! Try to decrease tau. Current tau is %s'%tau
            return X
        
        step = tau*reconstruct_matrix(vec_data - X_Omega, Omega, dims)
        X = X + step
        u, s, v = svds(X, k=k)
        X = u.dot(np.diag(s)).dot(v)
        prev_rmse = rmse
        if verbose == True:
            print 'Error on iteration %s is %.5f'%(t, rmse)
        t = t + 1
        
    return X