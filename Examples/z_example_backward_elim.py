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
import numpy as np

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
X_new = np.hstack((np.ones((len(X), 1)), X[:,[2, 8, 3]]))
reg_OLS = sm.OLS(y, X_new).fit()
reg_OLS.summary()

##
def forward_selection(input_X, output_y, plot_flag=False):
    # Work inside local scope
    X = np.array(input_X)
    y = np.array(output_y)
    # Null model comparision
    from sklearn.dummy import DummyRegressor
    null_model = DummyRegressor('mean')
    null_model.fit(X, y)
    y_hat = null_model.predict(X)
    #null_model = r2_score(y, y_hat)
    null_model = adjusted_r2_score(y, y_hat, len(y), 0)

    num_iter = [i for i in range(X.shape[1])]
    ind = []
    best_models = []
    while num_iter:
        M = []
        for i in num_iter:
            reg = LinearRegression()
            X_new = X[:, ind + [i]]
            reg.fit(X_new, y)
            y_hat = reg.predict(X_new)
            M.append(r2_score(y, y_hat))

        best_models.append(max(M))
        max_ind = M.index(max(M))
        ind.append(num_iter[max_ind])
        del num_iter[max_ind]

    if plot_flag is True:
        import matplotlib.pyplot as plt
        plt.plot(np.arange(0,X.shape[1]+1), np.append(null_model, best_models), label='All');
        plt.plot(best_models.index(max(best_models))+1 ,max(best_models), 'ro', label='Best');
        plt.xlabel('Number of features')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.grid()

    if not any(best_models > null_model):
        print('No better than null hypothesis')
        return 0
    else:
        max_ind = best_models.index(max(best_models))
        return max(best_models), ind[:max_ind+1]


##

