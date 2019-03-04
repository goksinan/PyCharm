import os
import pandas as pd
import numpy as np

# Load data from local directory
titanic = pd.read_csv(os.path.join('Datasets', 'titanic3.csv'))
print(titanic.columns)

# Assign features and labels
labels = titanic.survived.values
features = titanic[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]

""" 
We have categorical variables. We need to transform them into dummy variables
Pandas has a function to do that
"""
# Take a look at a few of dummy variables
features_with_dummies = pd.get_dummies(features, columns=['pclass', 'sex', 'embarked'])
features_with_dummies.head()

"""
All inputs are turned into numerical values. We don't need pandas anymore.
"""
# We can create a numpy array
data = features_with_dummies.values

# If any of the inputs had a non-numerical value, the following command will generate "object" data type
data.dtype

"""
# Dealing with missing values (NaNs).
"""
# Let's chekc if there is any missing value:
np.isnan(data).any()

"""
There are many ways to deal with the missing values. Here, we will use Imputer from sklearn
It replaces missing values with median value
"""
from sklearn.model_selection import train_test_split

# Split the data
XT, Xt, yT, yt = train_test_split(data, labels, random_state=0)

# Handle missing values. Note that we use the same transformation on test data.
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='median')
imp.fit(XT)
XT_new = imp.transform(XT)
Xt_new = imp.transform(Xt)

print('Missing values? ', 'Yes' if np.isnan(XT_new).any() is True else 'No')

"""
VER IMPORTANT! Use a "baseline classifier" for comparision. This is a "dummy" model, with no learning
"""
from sklearn.dummy import DummyClassifier

clf = DummyClassifier('most_frequent')
clf.fit(XT_new, yT)
print("Baseline prediction accuracy: %f" % clf.score(Xt, yt))

"""
CLASSIFICATION
"""
# Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='lbfgs', random_state=0).fit(XT_new, yT)
print('Logistic regression score: %f' % lr.score(Xt_new, yt))

# Random forest classifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=0).fit(XT_new, yT)
print('Random forest score: %f' % rf.score(Xt_new, yt))






