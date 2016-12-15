sample_fraction_array = np.linspace(0.1, 0.9, 9)
rank = 5
max_iter = int(1e3)
tol = 1e-3

M = get_data('real', -1, -1, row_num=1000)
max_M = np.max(M)
M /= max_M
dims = M.shape