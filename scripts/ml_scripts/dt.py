import sys
import io
import os
import yaml
import pickle
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier


input_file = os.path.join("data", "stage4", "train.csv")
output_file = os.path.join("models", "model.pkl")
os.makedirs(os.path.join("models"), exist_ok=True)

params = yaml.safe_load(open("params.yaml"))["train"]
p_max_depth = params["max_depth"]
p_max_features = params["max_features"]
p_min_samples_leaf = params["min_samples_leaf"]

df_gps = pd.read_csv(input_file)
print(df_gps)
x_train= df_gps.drop(labels=["Rating"], axis = 1)
y_train = df_gps['Rating']

tree = DecisionTreeClassifier(max_depth=p_max_depth,
                              max_features=p_max_features,
                              min_samples_leaf=p_min_samples_leaf)
tree.fit(x_train, y_train)


with open(output_file, "wb") as f:
    pickle.dump(tree, f)
