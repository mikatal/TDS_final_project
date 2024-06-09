from sklearn.cluster import SpectralClustering,KMeans
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

def run_spectral_clustering(n_clusters,dataset_encoded,dataset):
    labels = SpectralClustering(n_clusters=n_clusters).fit_predict(dataset_encoded)
    clusters = {}
    for cluster_label in np.unique(labels):
        clusters[cluster_label] = dataset.iloc[labels == cluster_label]
    return clusters

def run_k_means(n_clusters,dataset_encoded,dataset):
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(dataset_encoded)
    clusters = {}
    for cluster_label in np.unique(labels):
        clusters[cluster_label] = dataset.iloc[labels == cluster_label]
    return clusters
def preprocess_4_apriori(dtf,very_numerical):
    df = dtf.copy()
    for c in very_numerical:
        try:
            df[c] = pd.qcut(dtf[c],5,labels=["very low", "low", "medium", "high","very high"])
        except:
            #sometimes for highly skewed data, we cannot perform qcut as most quantiles are equal
            df[c] = pd.cut(dtf[c],5,labels=["very low", "low", "medium", "high","very high"])
    return df

def convert_ds_to_transactions(dtf):
    records = dtf.to_dict(orient='records')
    transactions=[]
    for r in records:
        transactions.append(list(r.items()))
    return transactions
def one_hot_encoding(df,categorical_features,dtype ):
    df_encoded = pd.get_dummies(df, columns=categorical_features,dtype=dtype)
    return df_encoded

def is_interesting_rule():
    pass
def opt_kmeans(data, max_k):
    k = []
    inertias = []
    
    for i in range(1,max_k+1):
        kmeans = KMeans(n_clusters=i,n_init=i)
        kmeans.fit(data)
        
        k.append(i)
        inertias.append(kmeans.inertia_)
        
        
    sns.set_style('whitegrid')
    plt.figure(figsize=(20,8))
    sns.lineplot(x=k,y=inertias,marker='o',dashes=False)
def opt_spectral(data, max_k):
    k = []
    inertias = []
    
    for i in range(1,max_k+1):
        print(i)
        spectral_clustering = SpectralClustering(n_clusters=i)
        spectral_clustering.fit(data)
        
        k.append(i)
        inertias.append(spectral_clustering.inertia_)
        
        
    sns.set_style('whitegrid')
    plt.figure(figsize=(20,8))
    sns.lineplot(x=k,y=inertias,marker='o',dashes=False)

def label_encoding(df,categorical_features):
    label_encoder = preprocessing.LabelEncoder()
    for column in categorical_features:
        df[column]=label_encoder.fit_transform(df[column])
    return df
