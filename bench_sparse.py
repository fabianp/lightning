"""
================================
Classification of text documents
================================

"""

import numpy as np
from scipy import linalg
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.sag import get_auto_step_size
from sklearn.linear_model import logistic
from lightning.impl.sag import SAGClassifier, SAGAClassifier
import pylab as plt

from lightning.impl.sgd_fast import Log
logloss = Log().loss
def logistic_loss(w, X, y, alpha):
    Xw = X.dot(w)
    n_samples = y.size
    return np.sum([logloss(Xw[i], y[i]) for i in range(n_samples)]) +\
        0.5 * alpha * w.dot(w)

# Load News20 dataset from scikit-learn.
bunch = fetch_20newsgroups_vectorized(subset="all")
X = bunch.data
y = bunch.target

# transform into binary class
y[y < y.mean()] = -1.
y[y >= y.mean()] = 1.

# Train / test split.
X_tr, X_te, y_tr, y_te = train_test_split(X, y,
                                          train_size=0.75,
                                          test_size=0.25,
                                          random_state=0)
alpha = 1e-10
diag_probas = X_tr.astype(np.bool).mean(axis=0)
print("Done probas")

alpha_scaled = alpha / X_tr.shape[0]
step = get_auto_step_size(1., alpha, 'log', False)
step_scaled = step / X_tr.shape[0]
print("Step size: %s" % step)

# get the "true" solution using scikit-learn
clf = LogisticRegression(
  C=1./alpha, fit_intercept=False, max_iter=10000,
  tol=1e-12)
clf.fit(X_tr, y_tr)
true_loss = logistic_loss(clf.coef_.ravel(), X_tr, y_tr, alpha)
print("True loss: %s" % true_loss)
# print("Gradient: %s" % linalg.norm(logistic._logistic_loss_and_grad(
#     clf.coef_.ravel(), X_tr, y_tr, alpha)[1]))

# # callback to get the loss function of SAGA
# loss_saga = []
# it = 0
# def cb_saga(model):
#     w_ = model.coef_.ravel() * model.coef_scale_[0]
#     tmp = logistic_loss(w_, X_tr, y_tr, alpha)
#     # grad = logistic._logistic_loss_and_grad(w_, X_tr, y_tr, alpha)[1]
#     # print(linalg.norm(grad))
#     global it
#     it += 1
#     print(it, tmp)
#     loss_saga.append(tmp - true_loss)


# clf1 = SAGAClassifier(
#   loss="log", eta=step / 20., alpha=alpha_scaled,
#   beta=0.0, max_iter=5000, verbose=False, random_state=0, callback=cb_saga,
#   tol=1e-12)
# clf1.fit(X, y)

# callback to get the loss function of SAG
loss_sag = []
it = 0
def cb_sag(model):
    w_ = model.coef_.ravel() * model.coef_scale_[0]
    tmp = logistic._logistic_loss(w_, X_tr, y_tr, alpha)
    grad = logistic._logistic_loss_and_grad(w_, X_tr, y_tr, alpha)[1]
    print(linalg.norm(grad))
    global it
    it += 1
    print(it, tmp)
    loss_sag.append(tmp - true_loss)

clf1 = SAGClassifier(
  loss="log", eta=step, alpha=alpha_scaled,
  max_iter=1000, verbose=False, random_state=0, callback=cb_sag,
  tol=1e-24)
clf1.fit(X, y)

# plt.plot(np.array(loss_saga), label="SAGA", lw=3)
plt.plot(np.array(loss_sag), label="SAG", lw=3)
plt.yscale('log')
plt.legend()
plt.grid()
plt.show()

# clf2 = SAGAClassifier(
#   loss="log", eta=step, alpha=alpha, beta=0.0, max_iter=500, verbose=False, random_state=0)

