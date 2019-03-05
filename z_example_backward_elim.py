print(__doc__)

import numpy as np
import matplotlib.pyplot as plt


def backward_elimination(XT, yT, SL=0.05):
    pass


# Import some data to work on
from sklearn.datasets import load_iris
iris = load_iris()

# Generate noise to be added as unimportant deatures
noise_features = np.random.uniform(0, 0.1, size=(len(iris.data), 20))

# Create the input and output data for classification
X = np.hstack((iris.data, noise_features))
y = iris.target


import statsmodels.formula.api as sm
XT = np.hstack((np.ones((len(X),1)), X))
for i in range(len(XT[0])):
    model = sm.OLS(y, XT).fit()
    p_values = model.pvalues.astype(float)
    highest_p = max(model.pvalues).astype(float)


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X, y)

y_hat = reg.predict(X)

## My function
def adjusted_r2_score(y, y_hat, n=None, k=None):
    SSresid = np.sum(np.square(y_hat-y))
    SStotal = np.sum(np.square(y-np.mean(y)))
    r2 = 1 - SSresid/SStotal
    if n is None or k is None:
        print('Adjusted r2 not returned. n and k values were expected.')
        return r2
    else:
        adjr2 = 1 - (SSresid/(n-k-1))/(SStotal/(n-1))
        return r2, adjr2

## Load data
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

## Build model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X, y)
y_hat = reg.predict(X)

## Evaluate model
from sklearn.metrics import r2_score, mean_squared_error
r2_score(y, y_hat)
mean_squared_error(y, y_hat)

## OLS method
import statsmodels.formula.api as sm
reg_OLS = sm.OLS(y, X).fit()
reg_OLS.summary()







