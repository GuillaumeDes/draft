from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt

pca = PCA()

logisitic_reg = LogisticRegression(max_iter=10000, tol=0.1)
pipe = Pipeline(steps=[("pca", pca), ("log_reg", logisitic_reg)])

X_digits, y_digits = datasets.load_digits(return_X_y=True)

param_grid = {
    'pca__n_components': [5, 15, 30, 45, 64],
    'log_reg__C': np.logspace(-4, 4, 4),
}
search = GridSearchCV(pipe, param_grid, n_jobs=-1, cv=10)
search.fit(X_digits, y_digits)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

pca.fit(X_digits)

pca_explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
ax0.plot(np.arange(1, pca.n_components_ + 1),
         pca.explained_variance_ratio_, '+', linewidth=2)
ax0.set_ylabel('PCA explained variance ratio')

ax0.axvline(search.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
ax0.legend(prop=dict(size=12))

print("ok")
