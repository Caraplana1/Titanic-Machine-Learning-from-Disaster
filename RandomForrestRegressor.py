import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

train_data_path = "Data/train.csv"
test_data_path = "Data/test.csv"

house_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

house_data.dropna(axis=0)

features = []

X = house_data[features]
y = house_data.SalePrice