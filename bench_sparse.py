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
from datetime import datetime

from lightning.impl.sgd_fast import Log
logloss = Log().loss
def logistic_loss(w, X, y, alpha, beta):
    Xw = X.dot(w)
    n_samples = y.size
    return np.sum([logloss(Xw[i], y[i]) for i in range(n_samples)]) +\
        0.5 * alpha * w.dot(w) + beta * np.sum(np.abs(w))

# Load News20 dataset from scikit-learn.
bunch = fetch_20newsgroups_vectorized(subset="all")
X = bunch.data
y = bunch.target

# transform into binary class
y[y < y.mean()] = -1.
y[y >= y.mean()] = 1.

# Train / test split.
alpha = 0.
alpha_scaled = alpha / X.shape[0]
beta = 1.
beta_scaled = beta / X.shape[0]
diag_probas = np.array(X.astype(np.bool).mean(axis=0)).ravel()
print("Done probas")

step = get_auto_step_size(1., alpha, 'log', False)
step_scaled = step / X.shape[0]
print("Step size: %s" % step)

# get the "true" solution using scikit-learn
clf = LogisticRegression(
  C=1./beta, fit_intercept=False, max_iter=200,
  penalty='l1', tol=1e-12)
clf.fit(X, y)
true_loss = logistic_loss(clf.coef_.ravel(), X, y, alpha, beta)
print("True loss: %s" % true_loss)
# print("Gradient: %s" % linalg.norm(logistic._logistic_loss_and_grad(
#     clf.coef_.ravel(), X, y, alpha)[1]))

# callback to get the loss function of SAGA
loss_saga = []
time1 = []
it = 0
def cb_saga(model):
    w_ = model.coef_.ravel() * model.coef_scale_[0]
    tmp = logistic_loss(w_, X, y, alpha, beta)
    loss_saga.append(tmp - true_loss)
    time1.append((datetime.now() - start).total_seconds())


clf1 = SAGAClassifier(
  loss="log", eta=step / 10., alpha=alpha_scaled, penalty='l1',
  beta=beta_scaled, max_iter=1000, verbose=True, random_state=0, callback=cb_saga,
  tol=1e-12)
start = datetime.now()
clf1.fit(X, y)

# callback to get the loss function of SAG
loss_saga2 = []
time2 = []
def cb_saga2(model):
    w_ = model.coef_.ravel() * model.coef_scale_[0]
    tmp = logistic_loss(w_, X, y, alpha, beta)
    loss_saga2.append(tmp - true_loss)
    time2.append((datetime.now() - start).total_seconds())


clf2 = SAGAClassifier(
  loss="log", eta=step / 10., alpha=alpha_scaled, penalty='l1',
  beta=beta_scaled, max_iter=1000, verbose=True, random_state=0, callback=cb_saga2,
  tol=1e-12, diag_probas=diag_probas)
start = datetime.now()
clf2.fit(X, y)

plt.title("Comparison with L1 regularization")
plt.plot(time1, np.array(loss_saga), label="SAGA - lagged updates", lw=3)
plt.plot(time2, np.array(loss_saga2), label="SAGA - sparse updates", lw=3, linestyle='--')
plt.yscale('log')
plt.ylabel("Function suboptimality")
plt.xlabel("Time (in seconds)")
plt.ylim((.1, None))
plt.xlim((0, 52))
plt.legend()
plt.grid()
plt.savefig('comparison_l1.png')
plt.show()
