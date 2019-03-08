"""
Linear regression using subset selection method
"""

##
def adjusted_r2_score(y, y_hat, n=None, k=None):
    """
    Compute r2 and adjusted r2 scores
    :param y: actual output
    :param y_hat: predicted output
    :param n: number of observations
    :param k: number of features
    :return: r2, adjusted r2
    """
    # Imports
    import numpy as np

    SSresid = np.sum(np.square(y_hat-y))  # residual sums of squares
    SStotal = np.sum(np.square(y-np.mean(y)))  # total sums of squares
    r2 = 1 - SSresid/SStotal  # r-squared
    if n is None or k is None:
        print('Adjusted r2 not returned. n and k values were expected.')
        return r2
    else:
        adjr2 = 1 - (SSresid/(n-k-1))/(SStotal/(n-1))  # adjusted r-squared
        return r2, adjr2


def cross_val(input_X, output_y, k=10, method='rsquared'):
    """
    Computes performance criteria by applying k-fold cross validation
    :param input_X: Input matrix (predictors)
    :param output_y: Target vector (response)
    :param k: Number of folds
    :param method: Performance criteria (rsquared, adjusted-rsquared)
    :return: Result of k-fold CV
    """
    # Imports
    import numpy as np
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LinearRegression

    # Work inside local scope
    X = np.array(input_X)
    y = np.array(output_y)

    # Work on splits
    kf = KFold(n_splits=k, random_state=0)
    scores = []
    for train_index, test_index in kf.split(X):
        XT = X[train_index]
        yT = y[train_index]
        Xt = X[test_index]
        yt = y[test_index]
        reg = LinearRegression()  # Build the model
        reg.fit(XT, yT)
        yt_hat = reg.predict(Xt)

        # Apply the performance criteria
        if method is 'rsquared' or None:
            scores.append(adjusted_r2_score(yt, yt_hat, Xt.shape[0], Xt.shape[1])[0])
        elif method is 'adjusted_rsquared':
            scores.append(adjusted_r2_score(yt, yt_hat, Xt.shape[0], Xt.shape[1])[1])

    return np.mean(scores)


def regular_regress(input_X, output_y, method='rsquared'):
    """
    Ordinady linear regression operation
    :param input_X: Input matrix (predictors)
    :param output_y: Target vector (response)
    :param method: Performance criteria (rsquared, adjusted-rsquared)
    :return:
    """
    # Imports
    import numpy as np
    from sklearn.linear_model import LinearRegression

    # Work inside local scope
    X = np.array(input_X)
    y = np.array(output_y)
    # Build the model
    reg = LinearRegression()
    # Fit the model
    reg.fit(X, y)
    # Make prediction
    y_hat = reg.predict(X)

    # Apply the performance criteria
    if method is 'rsquared' or None:
        return adjusted_r2_score(y, y_hat, X.shape[0], X.shape[1])[0]
    elif method is 'adjusted_rsquared':
        return adjusted_r2_score(y, y_hat, X.shape[0], X.shape[1])[1]


##
def forward_selection(input_X, output_y, method=None, validation=False, plot_flag=False):
    """
    Sequential forward selection algorithm
    :param input_X: Input matrix (predictors)
    :param output_y: Target vector (response)
    :param method: Performance criteria (rsquared, adjusted-rsquared)
    :param validation: If it is True, 10-fold cross validation will be applied
    :param plot_flag: Plot all best results
    :return: Best model result, features used in best model
    """
    # Imports
    import numpy as np

    # Work inside local scope
    X = np.array(input_X)
    y = np.array(output_y)

    # Null model for comparision
    from sklearn.dummy import DummyRegressor
    null_model = DummyRegressor('mean')
    null_model.fit(X, y)
    y_hat = null_model.predict(X)

    # Choose the performance criteria
    if method is 'rsquared' or None:
        null_model = adjusted_r2_score(y, y_hat, len(y), 0)[0]
    elif method is 'adjusted_rsquared':
        null_model = adjusted_r2_score(y, y_hat, len(y), 0)[1]

    # Iterations
    num_iter = [i for i in range(X.shape[1])]  # Will loop this list. Delete feature once chosen.
    ind = []  # This list keeps the features chosen
    best_models = []  # This list keeps the result of the best models
    while num_iter:
        M = []
        for i in num_iter:
            X_new = X[:, ind + [i]]  # Use a subset of features only. Keep them as column vectors

            # Check if cross validation is wanted
            if validation is True:
                M.append(cross_val(X_new, y, 10, method))
            else:
                M.append(regular_regress(X_new, y, method))

        best_models.append(max(M))  # append the best of this iteration
        max_ind = M.index(max(M))
        ind.append(num_iter[max_ind])  # add feature to the best-feaures-list
        del num_iter[max_ind]  # remove feature from the will-be-tested-next list

    # Insert full model performance to the list of best performances
    best_models = [null_model] + best_models

    # Plotting
    if plot_flag is True:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(best_models, label='All');
        plt.plot(best_models.index(max(best_models)) ,max(best_models), 'ro', label='Best');
        plt.title('Forward selection, ' + method + ', CV:' + str(validation))
        plt.xlabel('Number of features')
        plt.ylabel('Score')
        plt.xticks(np.arange(0,len(best_models)))
        plt.legend(loc='best')
        plt.grid()

    # Check if any of the models is better than null model
    # If so, decide the best model and return the feautures used in that model
    if not any(best_models > null_model):
        print('No better than null hypothesis')
        return 0
    else:
        max_ind = best_models.index(max(best_models))
        return max(best_models), ind[:max_ind+1]


##
def backward_elimination(input_X, output_y, method=None, validation=False, plot_flag=False):
    """
    Sequential backward selection algorithm
    :param input_X: Input matrix (predictors)
    :param output_y: Target vector (response)
    :param method: Performance criteria (rsquared, adjusted-rsquared)
    :param validation: If it is True, 10-fold cross validation will be applied
    :param plot_flag: Plot all best results
    :return: Best model result, features used in best model
    """
    # Imports
    import numpy as np

    # Work inside local scope
    X = np.array(input_X)
    y = np.array(output_y)

    # Null model for comparision
    from sklearn.dummy import DummyRegressor
    null_model = DummyRegressor('mean')
    null_model.fit(X, y)
    y_hat = null_model.predict(X)

    # Choose the performance criteria
    if method is 'rsquared' or None:
        null_model = adjusted_r2_score(y, y_hat, len(y), 0)[0]
    elif method is 'adjusted_rsquared':
        null_model = adjusted_r2_score(y, y_hat, len(y), 0)[1]

    # Full model for comparision
    if validation is True:
        full_model = cross_val(X, y, 10, method)
    else:
        full_model = regular_regress(X, y, method)

    # Iterations
    num_iter = [i for i in range(X.shape[1])]  # Will loop this list. Delete feature once chosen.
    all_indices = []
    all_indices.append(list(num_iter))
    removed_indices = []  # This list keeps the features chosen
    best_models = []  # This list keeps the result of the best models
    while len(num_iter) > 1:
        M = []
        for i in num_iter:
            ind = list(num_iter)  # Make a copy of the current elements
            ind.remove(i)  # Remove a certain element
            X_new = X[:, ind]  # Use a subset of features only. (Keep them as column vectors)

            # Check if cross validation is wanted
            if validation is True:
                M.append(cross_val(X_new, y, 10, method))
            else:
                M.append(regular_regress(X_new, y, method))

        best_models.append(max(M))  # append the best of this iteration
        max_ind = M.index(max(M))
        removed_indices.append(num_iter[max_ind])
        del num_iter[max_ind]  # remove feature from the will-be-tested-next list
        all_indices.append(list(num_iter))

    # Insert full model performance to the list of best performances
    best_models = [full_model] + best_models + [null_model]

    # Plotting
    if plot_flag is True:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(best_models, label='All');
        plt.plot(best_models.index(max(best_models)), max(best_models), 'ro', label='Best');
        plt.title('Backward Elimination, ' + method + ', CV:' + str(validation))
        plt.xlabel('Number of features')
        plt.ylabel('Score')
        xtick_numbers = list(np.arange(0, len(best_models)))
        xtick_labels = list(map(lambda n: str(n), list(reversed(xtick_numbers))))
        plt.xticks(xtick_numbers, xtick_labels)
        plt.legend(loc='best')
        plt.grid()

    # Check if any of the models is better than null model
    # If so, decide the best model and return the feautures used in that model
    if not any(best_models > null_model):
        print('No better than null hypothesis')
        return 0
    else:
        max_ind = best_models.index(max(best_models))
        return max(best_models), all_indices[max_ind]

## Load data
# Imports
import numpy as np
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

noise = np.random.uniform(0, 1, size=(len(X),10))

X_new = np.hstack((noise, X))

forward_selection(X_new, y, method='adjusted_rsquared', validation=True, plot_flag=True)

backward_elimination(X_new, y, method='adjusted_rsquared', validation=True, plot_flag=True)

