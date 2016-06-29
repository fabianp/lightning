# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Authors: Fabian Pedregosa
#

"""
These are some helper functions to compute the proximal operator of some common penalties
"""

import numpy as np
cimport numpy as np

cpdef prox_tv2d(np.ndarray[ndim=2, dtype=double] w, double stepsize,
                int max_iter=100, double tol=1e-3, callback=None):
    """
    Computes the proximal operator of the 2-dimensional total variation
    operator by Douglas-Rachford splitting.

    This solves a problem of the form

         argmin_x TV(x) + (1/(2 stepsize)) ||x - w||^2

    where TV(x) is the one-dimensional total variation

    Parameters
    ----------
    w: array
        vector of coefficieents

    stepsize: float
        step size (sometimes denoted gamma) in proximal objective function

    max_iter: int
        maximum number of iterations

    tol: float
        Absolute tolerance of stopping criterion

    Returns
    -------
    None, operates in-place

    References
    ----------
    Barbero, Á., & Sra, S. (2014). Modular proximal optimization for multidimensional
    total-variation regularization. Retrieved from http://arxiv.org/abs/1411.0589
    """
    cdef np.ndarray[ndim=2, dtype=double] z = w.T.copy()
    cdef size_t n_rows = w.shape[0]
    cdef size_t n_cols = w.shape[1]
    cdef size_t it, i, j
    cdef double current_tol
    for it in range(max_iter):
        # apply the proximal operator through columns
        w[:] = z.T[:]
        for i in range(n_rows):
            prox_tv1d(w[i], stepsize)
        z[:] = 2 * w.T[:] - z[:]
        for j in range(n_cols):
            prox_tv1d(z[j], stepsize)

        if it % 10 == 9:
            print(np.abs(w - z.T).sum())
            if np.abs(w - z.T).sum() < tol:
                break
        if callback is not None:
            callback(w)


cpdef prox_tv1d(np.ndarray[ndim=1, dtype=double] w, double stepsize):
    """
    Computes the proximal operator of the 1-dimensional total variation operator.

    This solves a problem of the form

         argmin_x TV(x) + (1/(2 stepsize)) ||x - w||^2

    where TV(x) is the one-dimensional total variation

    Parameters
    ----------
    w: array
        vector of coefficieents
    stepsize: float
        step size (sometimes denoted gamma) in proximal objective function

    Returns
    -------
    None, operates in-place

    References
    ----------
    Condat, Laurent. "A direct algorithm for 1D total variation denoising."
    IEEE Signal Processing Letters (2013)
    """
    cdef long width, k, k0, kplus, kminus
    cdef double umin, umax, vmin, vmax, twolambda, minlambda
    width = w.size

    # /to avoid invalid memory access to input[0] and invalid lambda values
    if width > 0 and stepsize >= 0:
        k, k0 = 0, 0			# k: current sample location, k0: beginning of current segment
        umin = stepsize  # u is the dual variable
        umax = - stepsize
        vmin = w[0] - stepsize
        vmax = w[0] + stepsize	  # bounds for the segment's value
        kplus = 0
        kminus = 0 	# last positions where umax=-lambda, umin=lambda, respectively
        twolambda = 2.0 * stepsize  # auxiliary variable
        minlambda = -stepsize		# auxiliary variable
        while True:				# simple loop, the exit test is inside
            while k == width-1: 	# we use the right boundary condition
                if umin < 0.0:			# vmin is too high -> negative jump necessary
                    while True:
                        w[k0] = vmin
                        k0 += 1
                        if k0 > kminus:
                            break
                    k = k0
                    kminus = k
                    vmin = w[kminus]
                    umin = stepsize
                    umax = vmin + umin - vmax
                elif umax > 0.0:    # vmax is too low -> positive jump necessary
                    while True:
                        w[k0] = vmax
                        k0 += 1
                        if k0 > kplus:
                            break
                    k = k0
                    kplus = k
                    vmax = w[kplus]
                    umax = minlambda
                    umin = vmax + umax -vmin
                else:
                    vmin += umin / (k-k0+1)
                    while True:
                        w[k0] = vmin
                        k0 += 1
                        if k0 > k:
                            break
                    return
            umin += w[k + 1] - vmin
            if umin < minlambda:       # negative jump necessary
                while True:
                    w[k0] = vmin
                    k0 += 1
                    if k0 > kminus:
                        break
                k = k0
                kminus = k
                kplus = kminus
                vmin = w[kplus]
                vmax = vmin + twolambda
                umin = stepsize
                umax = minlambda
            else:
                umax += w[k + 1] - vmax
                if umax > stepsize:
                    while True:
                        w[k0] = vmax
                        k0 += 1
                        if k0 > kplus:
                            break
                    k = k0
                    kminus = k
                    kplus = kminus
                    vmax = w[kplus]
                    vmin = vmax - twolambda
                    umin = stepsize
                    umax = minlambda
                else:                   # no jump necessary, we continue
                    k += 1
                    if umin >= stepsize:		# update of vmin
                        kminus = k
                        vmin += (umin - stepsize) / (kminus - k0 + 1)
                        umin = stepsize
                    if umax <= minlambda:	    # update of vmax
                        kplus = k
                        vmax += (umax + stepsize) / (kplus - k0 + 1)
                        umax = minlambda