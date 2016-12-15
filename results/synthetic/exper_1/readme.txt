sample_fraction_array = np.linspace(0.1, 0.9, 15)
rank = 5
dims = (500, 500)
max_iter = int(1e3)
tol = 1e-3

M = get_data('synthetic', rank, dims, noise='y')