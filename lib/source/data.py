import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

def sampling_operator(nsamp, dims, seed=1):
    """
    Sampling operator (uniform distribution)
    Input:
    nsamp : number of samples
    dims : size of the matrix
    seed : set random seed (default 1)
    Output:
    array of indices (size nsamp x 2)
    """
    np.random.seed(seed)
    idx = np.random.choice(np.prod(dims), nsamp, replace=False)
    
    return np.vstack(np.unravel_index(idx, dims)).T

def get_matrix(rank, dims, noise='n', sigma=1e-3, seed=1):
    """
    Get synthetic matrices of desired rank
    Input:
    rank : rank of the matrix
    dims : size of the matrix
    noise : 'y' - with noise or 
            'n' - without noise (default 'n')
            Noise from N(mean=0, variance=sigma).
    sigma : variance of the noise (default 1e-3)
    seed :set random seed (default 1)
    Output:
    M : matrix of the desired rank
    """
    np.random.seed(seed)
    M_l = np.random.randn(dims[0], rank)
    M_r = np.random.randn(dims[1], rank)
    M = M_l.dot(M_r.T)
    if noise == 'y':
        M += sigma * np.random.randn(M.shape[0], M.shape[1])
    
    return M

def get_sampling_matrix(M, omega):
    """
    Get sampling matrix
    Input:
    M : array like
    omega : array of sampling indices
    Output:
    data: array like with sampling elements
    """
    data = np.zeros(M.shape)
    for i in xrange(omega.shape[0]):
        data[omega[i, 0], omega[i, 1]] = M[omega[i, 0], omega[i, 1]]
        
    return data

def get_sampling_vector(M, omega):
    """
    Get sampling vector
    Input:
    M : array like
    omega : array of sampling indices
    Output:
    data: vector with sampling elements
    """
    data = np.zeros(omega.shape[0])
    for i in xrange(omega.shape[0]):
        data[i] = M[omega[i, 0], omega[i, 1]]
        
    return data

def reconstruct_matrix(vec, omega, dims, sparse_type='n'):
    """
    Reconstruct sampling matrix from vector
    Input:
    vec : vector with sampling elements
    omega : array of sampling indices
    dims : size of the matrix
    sparse_type : 'y' or 'n' (default 'n')
    Output:
    data: array like with sampling elements
    """
    if sparse_type == 'n':
        data = np.zeros(dims)
        for i in xrange(omega.shape[0]):
            data[omega[i, 0], omega[i, 1]] = vec[i]
    elif sparse_type == 'y':
        data = csr_matrix((vec, omega.T), dims)
        
    return data

def get_data(data_type, rank, dims, noise='n', sigma=1e-3, seed=1, row_num=2000):
    """
    Get data
    Input:
    data_type : type of desired data
    If data_type='synthetic' :
        rank : rank of the matrix
        dims : size of the matrix
        noise : 'y' - with noise or 
                'n' - without noise (default 'n')
                Noise from N(mean=0, variance=sigma).
        sigma : variance of the noise (default 1e-3)
        seed :set random seed (default 1)
        row_num : Number of rows from Jester dataset
    Output:
    M : matrix
    """ 
    if data_type == 'synthetic':
        M = get_matrix(rank, dims, noise, sigma, seed)
    if data_type == 'real':
        M = pd.read_excel('./data/jester-data-2.xls', header=None, na_values=99)
        M = M.fillna(0)
        M = np.array(M)[:, 1:]
        M = M[:row_num, :]
    else:
        print 'Not yet!'
        return -1
    
    return M