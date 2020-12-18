import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Datasets paths
titanic_data_train_path = "Data/train.csv"
titanic_data_test_path = "Data/test.csv"

# Read and drop non aviable data
titanic_data_train = pd.read_csv (titanic_data_train_path)
titanic_data_train.dropna(axis=0)


# features to train the model
features = ["Pclass", "Sex", "Age", "Cabin"]

# Matrixes how keep th data.
X = titanic_data_train[features]
y = titanic_data_train.Survived
