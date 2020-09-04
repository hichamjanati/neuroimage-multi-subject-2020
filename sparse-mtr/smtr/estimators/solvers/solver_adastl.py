"""Solvers for multitask group- adaptive-stl."""
import numpy as np
from celer import Lasso


def solver_adastl(X, Y, alpha=None, callback=None, positive=False,
                  maxiter=3000, tol=1e-4):
    """Perform CD to solve adaptive positive Lasso."""

    n_tasks, n_samples, n_features = X.shape
    theta = np.ones((n_features, n_tasks)) / 4
    loss = []
    XX = X.copy()
    if callback:
        callback(theta)

    if alpha is None:
        alpha = np.ones(n_tasks)
    alpha = np.asarray(alpha).reshape(n_tasks)
    for k in range(n_tasks):
        loss.append([])
        lasso = Lasso(alpha=alpha[k], tol=tol, max_iter=maxiter,
                      positive=positive, fit_intercept=False)
        for i in range(1000):
            # ll = 0.5 * np.linalg.norm(Y[k] - X[k].dot(theta[:, k])) ** 2
            # ll /= n_samples
            # ll += alpha[k] * np.sqrt(abs(theta[:, k])).sum()
            # loss[k].append(ll)
            weights = 2 * abs(theta[:, k]) ** 0.5
            XX = X[k] * weights[None, :]
            lasso.fit(XX, Y[k])
            theta_new = lasso.coef_ * weights
            cstr = abs(theta_new - theta[:, k]).max()
            cstr /= max(abs(theta_new).max(), abs(theta[:, k]).max(), 1)
            theta[:, k] = theta_new
            if cstr < tol:
                break
        if callback:
            callback(theta)

    return theta, loss
