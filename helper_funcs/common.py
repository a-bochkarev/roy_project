def get_matrix(rank, dims, seed=1):
	"""
	TO DO
	"""
    np.random.seed(seed)
    M_l = np.random.randn(dims[0], rank)
    M_r = np.random.randn(dims[1], rank)
    
    return M_l.dot(M_r.T)

def sampling_operator(nsamp, dims, seed=1):
	"""
	TO DO
	"""
    np.random.seed(seed)
    idx = np.random.choice(np.prod(dims), nsamp, replace=False)
    
    return np.vstack(np.unravel_index(idx, dims)).T

def sampling_matrix(Omega, M):
	"""
	TO DO
	"""
    P = np.zeros_like(M)
    for i in range(Omega.shape[0]):
        P[Omega[i][0], Omega[i][1]] = M[Omega[i][0], Omega[i][1]]
    return P