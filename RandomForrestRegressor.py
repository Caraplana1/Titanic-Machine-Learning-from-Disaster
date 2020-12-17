import pandas as pd


train_data_path = "C:/Users/angel/Desktop/Programación/Python/ML/House Pricing competition/home-data-for-ml-course/train.csv"
test_data_path = "C:/Users/angel/Desktop/Programación/Python/ML/House Pricing competition/home-data-for-ml-course/test.csv"

house_data = pd.read_csv(train_data_path)

house_data.dropna(axis=0)