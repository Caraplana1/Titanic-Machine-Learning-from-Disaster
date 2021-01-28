import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

"""The Model is suffering a high Bias"""

# Datasets paths
titanic_data_train_path = "Data/train.csv"
titanic_data_test_path = "Data/test.csv"

# Read and drop non aviable data
titanic_data_train = pd.read_csv (titanic_data_train_path)
#  titanic_data_train.dropna(axis=0)

# features to train the model
features = ["Pclass", "Sex", "Age"]

# Read the test data
titanic_data_test = pd.read_csv(titanic_data_test_path)
#  titanic_data_test.dropna(axis=0)

# Matrixes how keep th data.
X = titanic_data_train[features]
y = titanic_data_train.Survived

test_X = titanic_data_test[features]

# This solve a SettingWithCopyWarning for the next two lines, but i don't have idea what that mean
X = X.copy(deep=True)

# Transform the sex string into booleans
X["Sex"]= X["Sex"].replace(["male","female"], [1,0])

# Repace the non aviable data with the mean
X.Age = np.nan_to_num(X.Age, nan=29.7)
X.Sex = np.nan_to_num(X.Sex,nan=1)
X.Pclass = np.nan_to_num(X.Pclass, nan=2)


# This solve a SettingWithCopyWarning for the next two lines, but i don't have idea what that mean
test_X = test_X.copy(deep=True)

# Transform the sex string into booleans
test_X["Sex"] = test_X["Sex"].replace(["male","female"], [1,0])

# Repace the non aviable data with the mean
test_X.Age = np.nan_to_num(test_X.Age, nan=29.7)
test_X.Sex = np.nan_to_num(test_X.Sex,nan=1)
test_X.Pclass = np.nan_to_num(test_X.Pclass, nan=2)


# Split the train data and the test data
train_X, val_X, train_y, val_y = train_test_split(X, y)


# Create and train the RandomForestClassifier model.
#  model = RandomForestClassifier(random_state=0)
#  model.fit(X, y)

model = LogisticRegression(random_state=0)
model.fit(train_X,train_y)

# Predicts
prediction = model.predict(val_X)
predictionTrain = model.predict(train_X)

print("Validation error")
print(mean_absolute_error(val_y,prediction)*100)
print("Traint error")
print(mean_absolute_error(train_y,predictionTrain)*100)

# Create the final data frame to print in the csv.
#  final_response = pd.DataFrame({"PassengerId" : titanic_data_test["PassengerId"], "Survived" : prediction})

#final_response.to_csv(r'Data/submission.csv', index=False)
