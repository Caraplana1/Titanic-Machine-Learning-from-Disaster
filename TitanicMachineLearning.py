import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#change the size of the tree and the number of trees in the forrest
def Set_forrest_size(number_of_trees, size_of_tree):

    return RandomForestClassifier(random_state=0, n_estimators=number_of_trees, max_leaf_nodes=size_of_tree)


# Datasets paths
titanic_data_train_path = "Data/train.csv"
titanic_data_test_path = "Data/test.csv"

# Read and drop non aviable data
titanic_data_train = pd.read_csv (titanic_data_train_path)
titanic_data_train.dropna(axis=0)


# features to train the model
features = ["Pclass", "Sex", "Age", "Embarked"]

# Matrixes how keep th data.
X = titanic_data_train[features]
y = titanic_data_train.Survived

# This solve a SettingWithCopyWarning for the next two lines, but i don't have idea what that mean
X = X.copy(deep=True)

# Transform the sex string into booleans
X["Sex"]= X["Sex"].replace(["male","female"], [1,0])
X["Embarked"] = X["Embarked"].replace(["S","C","Q"],[1,2,3])

# Repace the non aviable data with the mean
X.Age = np.nan_to_num(X.Age, nan=29.7)
X.Sex = np.nan_to_num(X.Sex,nan=1)
X.Embarked = np.nan_to_num(X.Embarked)
X.Pclass = np.nan_to_num(X.Pclass, nan=2)


train_X, val_X, train_y, val_y = train_test_split(X, y)

for i in []:
    # Create and train the model
    model = Set_forrest_size()
    model.fit(train_X, train_y)

    # Predicts
    prediction = model.predict(val_X)

    mae = mean_absolute_error(val_y,prediction)

    print (mae * 100)
