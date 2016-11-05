import numpy as np
from lightning.impl import penalty

# def test_tv1_denoise():
#     # test the prox of TV1D
#     # since its not trivial to check the KKT conditions
#     # we check that the proximal point algorithm converges
#     # to a solution to the TV minimization
#     n_iter = 100
#     n_features = 100
#
#     # repeat the test 10 times
#     for nrun in range(10):
#         x = np.random.randn(n_features)
#         for _ in range(n_iter):
#             x = prox_fast.prox_tv1d(x, 1.0)
#         # check that the solution is flat
#         np.testing.assert_allclose(x, x.mean() * np.ones(n_features))

def test_tv1_prox():
    """
    Use the properties of strongly convex functions to test the implementation
    of the TV1D proximal operator. In particular, we use the following inequality
    applied to the proximal objective function: if f is mu-strongly convex then

          f(x) - f(x^*) >= ||x - x^*||^2 / (2 mu)

    where x^* is the optimum of f.
    """
    n_features = 10
    gamma = np.random.rand()
    pen = penalty.TotalVariation1DPenalty()

    for nrun in range(5):
        x = np.random.randn(1, n_features)
        x2 = pen.projection(x, gamma, 1)
        diff_obj = pen.regularization(x) - pen.regularization(x2)
        assert diff_obj >= ((x - x2) ** 2).sum() / (2 * gamma)


def test_tv2_prox():
    """
    Same
    """
    n_features = 36
    gamma = np.random.rand()
    pen = penalty.TotalVariation2DPenalty(6, 6)

    for nrun in range(5):
        x = np.random.randn(1, n_features)
        x2 = pen.projection(x, gamma, 1)
        diff_obj = pen.regularization(x) - pen.regularization(x2)
        assert diff_obj >= ((x - x2) ** 2).sum() / (2 * gamma)