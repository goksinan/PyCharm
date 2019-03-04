

"""
How to use Principal Component Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA


## LOAD DATASET
iris = load_iris()

X = iris.data
y = iris.target

## IMPLEMENT PCA
pca = PCA(n_components=4) # PCA will return the first 2 components
pca.fit(X) # Fit a PCA model to the data
coef = pca.components_ # The pca coefficients
score = pca.transform(X) # The principal components
latent = pca.explained_variance_ # Explained variance
explained = pca.explained_variance_ratio_ # Explained variance normalized

## CONFIRM THE PCA OPERATION USING THE RETURNED COEFFICIENTS
# "Coef" is a transformation matrix that transforms the demeaned input data into orthagonal PCs
X_centered = X - np.mean(X, axis=0)
my_score  = np.dot(X_centered, coef.T)
# Visualize the two results to confirm that they are the same
plt.plot(score, '.', label='PCA score')
plt.plot(my_score, 'o', label='MY score', mfc='none')
plt.legend(loc='best')

## HOW TO RECOVER ORIGINAL SIGNAL? (Actually, the centered version)
X_recovered = np.dot(score, coef)
# Verify
plt.plot(X_centered, '.', label='Actual centered X')
plt.plot(X_recovered, 'o', label='Recovered centered X', mfc='none')
plt.legend(loc='best')


