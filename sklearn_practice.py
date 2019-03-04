
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


## IRIS DATA
# Load the data
from sklearn.datasets import load_iris
iris = load_iris()
# See what kind of parts it has
iris.keys()
# See number of observations and features
n_samples, n_features = iris.data.shape
print('number of samples: ', n_samples)
print('number of features: ', n_features)
print(iris.data[0])
print(iris.target.shape)

# See how many different labels there are
np.bincount(iris.target)

# Plot histogram of one feature
x_index = 3
for label in range(len(iris.target_names)):
    print(label)
    plt.hist(iris.data[iris.target==label, x_index],
             label=iris.target_names[label],
             alpha=0.5)

plt.xlabel(iris.feature_names[x_index])
plt.legend(loc='best')
plt.show()

# Plot scatter plot of two features
x_index = 2
y_index = 2

for label in range(len(iris.target_names)):
    plt.scatter(iris.data[iris.target==label, x_index],
                iris.data[iris.target==label, y_index],
                label=iris.target_names[label])

plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
plt.legend(loc='best')
plt.show()

# Grid of scatter plots using Pandas
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
pd.plotting.scatter_matrix(iris_df, c=iris.target, figsize=(8,8))

## DIGITS DATA
from sklearn.datasets import load_digits
digits = load_digits()

digits.keys()
print('Num of samples: ', digits.data.shape[0])
print('Num of features: ', digits.data.shape[1])

print(digits.data[0])
print(digits.target)

# Set up the figure
fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(64):
    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # Put the target label
    ax.text(0, 7, str(digits.target[i]))

## LOAD FACES DATA
from sklearn.datasets import fetch_olivetti_faces
faces = fetch_olivetti_faces()
type(faces)
faces.keys()
faces.data.shape
print('Num of samples: ', faces.data.shape[0])
print('Num of features: ', faces.data.shape[1])

print(faces.data[0])
print(faces.target)

# Set up the figure
fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    ax.imshow(faces.images[i], cmap=plt.cm.bone, interpolation='nearest')

## START MACHINE LEARNING
iris = load_iris()
X, y = iris.data, iris.target

# Train-test split
from sklearn.model_selection import train_test_split

XT, Xt, yT, yt = train_test_split(X, y, train_size=0.5, test_size=0.5, random_state=123)

print("Training set labels:")
print(yT)

print("Test set label:")
print(yt)

""" 
When we did splitting, we didn't pay attention to balance. In other words, the ratio of different classes in the 
original data were not preserved. Check out the ratios in the original dataset, the trainin set and the test set below
"""
print('Original: ', np.bincount(y)/len(y)*100)
print('Training: ', np.bincount(yT)/len(yT)*100)
print('Test: ', np.bincount(yt)/len(yt)*100)

""" The remedy is to use stratify parameter: """
XT, Xt, yT, yt = train_test_split(X, y, train_size=0.5, test_size=0.5, random_state=123, stratify=y)

print('Original: ', np.bincount(y)/len(y)*100)
print('Training: ', np.bincount(yT)/len(yT)*100)
print('Test: ', np.bincount(yt)/len(yt)*100)

# K-nearest neighbors
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier().fit(XT, yT)
yt_hat = classifier.predict(Xt)

print('Accuracy: ')
print(np.sum(yt_hat == yt)/len(yt))

# Visualize the results
incorrect_idx = np.where(yt_hat != yt)[0]

for n in np.unique(yt):
    print(n)
    idx = np.where(yt == n)[0]
    plt.scatter(Xt[idx,0], Xt[idx,3], label=iris.target_names[n])

plt.scatter(Xt[incorrect_idx,0], Xt[incorrect_idx,3], s=100, facecolors='none', edgecolors='r', label='missclassified')

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[3])
plt.legend(loc='best')
plt.show()


## Generate synthetic data and work on it
from sklearn.datasets import make_blobs

X, y = make_blobs(centers=2, random_state=0, cluster_std=1.5)
X[:5,:]
y[:5]

plt.scatter(X[y==0,0], X[y==0,1], s=50, label='0')
plt.scatter(X[y==1,0], X[y==1,1], s=50, label='1')
plt.xlabel('first feature')
plt.ylabel('second feature')
plt.legend(loc='best')

XT, Xt, yT, yt = train_test_split(X, y, test_size=0.25, random_state=1234, stratify=y)
print('Original: ', np.bincount(y)/len(y)*100)
print('Training: ', np.bincount(yT)/len(yT)*100)
print('Test: ', np.bincount(yt)/len(yt)*100)

## Use the logistic regression classifier
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

XT.shape
yT.shape

# Build the model
classifier.fit(XT, yT)
# Make prediction
yt_hat = classifier.predict(Xt)
# Examine prediction results
print('Actual    :', yt)
print('Predicted :', yt_hat)
print('Accuracy  :', np.mean(yt == yt_hat))

# Evaluate the results in short way
classifier.score(Xt,yt)

# Visualize
def plot_decision_boundary(clf):
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-3, 7)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    plt.plot(xx, yy, 'k-')

plt.scatter(X[y==0,0], X[y==0,1], s=50, label='0')
plt.scatter(X[y==1,0], X[y==1,1], s=50, label='1')
plt.xlabel('first feature')
plt.ylabel('second feature')
plt.legend(loc='best')
plot_decision_boundary(classifier)

## Use KNN classifier
from sklearn.neighbors import KNeighborsClassifier
from my_functions.plot_2d_separator import plot_2d_separator

knn = KNeighborsClassifier(n_neighbors=30)

knn.fit(XT, yT)

plt.scatter(X[y==0,0], X[y==0,1], s=50, label='0')
plt.scatter(X[y==1,0], X[y==1,1], s=50, label='1')
plt.xlabel('first feature')
plt.ylabel('second feature')
plt.legend(loc='best')
plot_2d_separator(knn, X)

knn.score(Xt,yt)


## Using the Iris data set, pick the optimum K for KNN Classifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

# Choose the test set
XT, Xt, yT, yt = train_test_split(X, y, test_size=0.25, random_state=1234, stratify=y)
# Choose a validation set to optimize K value
XT_sub, XT_val, yT_sub, yT_val = train_test_split(XT, yT, test_size=0.5, random_state=1234, stratify=yT)

for k in range(1,20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(XT_sub,yT_sub)
    score_Tsub = knn.score(XT_sub, yT_sub)
    score_Tval = knn.score(XT_val, yT_val)
    print('{}: Train/Val Accuracy: {:.3f} / {:.3f}'.format(k, score_Tsub, score_Tval))

# We decided that our optimal K-value is 2
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(XT,yT)
print('k=2 Test Acc: {:.3f}'.format(knn.score(Xt,yt)))



## REGRESSION
x = np.linspace(-3, 3, 100)
rng = np.random.RandomState(42)
y = np.sin(4*x) + x + rng.uniform(size=len(x))

plt.plot(x, y, 'o')
plt.xlabel('x')
plt.ylabel('y')

X = x[:, np.newaxis]

from sklearn.model_selection import train_test_split
XT, Xt, yT, yt = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(XT, yT)

print('Weight coefficients :', regressor.coef_)
print('Intercept           :', regressor.intercept_)

# To plot the regression line
min_pt = X.min() * regressor.coef_[0] + regressor.intercept_
max_pt = X.max() * regressor.coef_[0] + regressor.intercept_

plt.plot(XT, yT, 'o')
plt.plot([X.min(), X.max()], [min_pt, max_pt]) # the regression line

# Prediction
yt_hat = regressor.predict(Xt)

plt.plot(Xt, yt, 'o', label='data')
plt.plot(Xt, yt_hat, 'o', label='prediction')
plt.plot([X.min(), X.max()], [min_pt, max_pt], label='fit') # the regression line
plt.legend(loc='best')

# Evaluate the model
regressor.score(Xt,yt) # Coef of determination

from sklearn.metrics import mean_squared_error, mean_absolute_error

mean_squared_error(yt, yt_hat)
mean_absolute_error(yt, yt_hat)


## Non-linear regression
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(XT, yT)

# Training prediction
yT_hat = knn.predict(XT)
plt.plot(XT, yT, 'o', label='data')
plt.plot(XT, yT_hat, '.', label='prediction')
plt.legend(loc='best')
plt.show()

# Test prediction
yt_hat = knn.predict(Xt)
plt.plot(Xt, yt, 'o', label='data')
plt.plot(Xt, yt_hat, '.', label='prediction')
plt.legend(loc='best')
plt.show()

knn.score(Xt,yt)

