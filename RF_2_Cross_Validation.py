from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import geopandas as gpd
import pandas as pd
import numpy as np
from matplotlib import *
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


# cross-validation script on RF, input is an building shapefile. User must define what is target and what is data (columns in atribute table)
buildings = gpd.read_file(r'C:\Users\vitak\Desktop\DP_V2\RF\Data\Praha3_budovy.shp')

buildings.target = buildings['typ']
buildings.data = buildings.drop(['typ', 'geometry'], axis = 1)

# data preparation
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(buildings.data,buildings.target,test_size=0.3)

# classifier application
rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

# calculation of parameter importance
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

feature_names = []
for col in buildings.data.columns:
    feature_names.append(col)

forest_importances = pd.Series(importances, index=feature_names)

# plotting the importance of parameters
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using mean decrease in impurity")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()

# cross-validation settings
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

# on how many k-fold to split the data
from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=4)

scores_rf = []

# cross-validation calculation
for train_index, test_index in folds.split(buildings.data,buildings.target):
    X_train, X_test, y_train, y_test = buildings.data.iloc[train_index], buildings.data.iloc[test_index], \
                                       buildings.target.iloc[train_index], buildings.target.iloc[test_index]
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))
print(scores_rf)