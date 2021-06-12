# Below proces starts after the features dataset is created at customer id level
# Please have required steps done prior to this step to have the features dataset ready

import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

# load features df
df = pd.read_csv('./../data/clustering_features.csv').set_index('customer_id')

# fill na
df = df.fillna(0)

# scaling
cols_scale = [
    'sales','units','upt','aur','aov','unique_categories_bought','unique_payments_used',
    'unique_products_bought','orders']

# Load scaler  from file
pkl_filename = "./../files/model_objects/kmeans_scaler_model.pkl"
with open(pkl_filename, 'rb') as file:
    scaler = pickle.load(file)

df[cols_scale] = scaler.transform(df[cols_scale])

# Load k-means from file
pkl_filename = "./../files/model_objects/kmeans_model.pkl"
with open(pkl_filename, 'rb') as file:
    kmeans = pickle.load(file)

df['cluster_ids'] = kmeans.predict(df)

# save labels
df[['cluster_ids']].to_csv('./../files/kmeans_labels.csv', index=True)