cimport numpy as np

cpdef prox_tv1d(np.ndarray[ndim=1, dtype=double] w, double stepsize)

cpdef prox_tv2d(np.ndarray[ndim=2, dtype=double] w, double stepsize, int max_iter=*, double tol=*)
