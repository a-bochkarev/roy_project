sample_fraction = 0.2
rank_array = range(1, 32, 3)
dims = (300, 300)
max_iter = int(1e3)
tol = 1e-3

nsamp = int(sample_fraction * np.prod(dims))
omega = sampling_operator(nsamp, dims)