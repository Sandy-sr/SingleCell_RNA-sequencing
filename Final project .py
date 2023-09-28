
import pandas as pd
import numpy as np
import requests
import json

from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

df = pd.read_csv('data_tr.csv',delimiter="\t", header=None)
dg = pd.read_csv('',delimiter="\t", header=None)

np.random.seed(seed=0)

cols_to_drop = [1099, 1113, 1875, 2093, 2446, 2620, 2937, 3141, 4431, 9586, 10778]

X = df.drop(cols_to_drop,axis = 1)
Y = dg.drop(cols_to_drop,axis = 1)

from sklearn.preprocessing import MinMaxScaler

# fit scaler on training data
norm = MinMaxScaler(feature_range=(0, 10))
norm.fit(X)

# transform training data
X_train_norm = norm.transform(X)

X_test_norm = norm.transform(Y)

pca_100 = PCA(n_components=100, random_state=0)
pca_100.fit(X_train_norm)
X_train_norm_100 = pca_100.transform(X_train_norm)
X_test_norm_100 = pca_100.transform(X_test_norm)

bir = Birch(n_clusters=16,threshold=3.5,branching_factor=400)
bir.fit(X_train_norm_100)
prediction = bir.predict(X_test_norm_100)

url = "https://www.csci555competition.online/score"
headers = {
  'Content-Type': 'application/json'
}
response = requests.request("POST", url, headers=headers, data=json.dumps(prediction.tolist()))

print(response.text)

score_agglo = silhouette_score(X_train_norm_100, prediction, metric='euclidean')

score_agglo





