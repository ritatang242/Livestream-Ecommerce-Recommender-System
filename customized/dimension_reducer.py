import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.cluster import KMeans

def pca_df(fit, cust_id, dim=2):
    pca_model = PCA(n_components=dim)
    pca_model.fit(fit)
    print(f"Explainess: {pca_model.explained_variance_ratio_}")
    print(f"Total explain: {sum(pca_model.explained_variance_ratio_)}")

    fit_df = pd.DataFrame(fit, index=cust_id, columns=['dim_'+str(i+1) for i in range(fit.shape[1])])
    pca = pca_model.transform(fit_df)
    pca_df = pd.DataFrame(pca, index=fit_df.index, columns=['Dim'+str(i+1) for i in range(dim)])
    return pca_df

def tsne_df(fit, cust_id, dim=2):
    tsne_model = TSNE(n_components=dim, perplexity=50, n_iter=300, random_state = 2022)
    tsne = tsne_model.fit_transform(fit)
    tsne_df = pd.DataFrame(tsne, index=cust_id, columns = ['Dim'+str(i+1) for i in range(tsne.shape[1])])
    return tsne_df

def umap_df(fit, cust_id, dim=2):
    reducer = umap.UMAP(n_components=dim, random_state=2022)
    embedding = reducer.fit_transform(fit)
    umap_df = pd.DataFrame(embedding, index=cust_id, columns = ['Dim'+str(i+1) for i in range(embedding.shape[1])])
    return umap_df

def kmeans_df(fit, cluster):
    kmeans_model = KMeans(n_clusters = cluster, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 2022)
    kmeans_model.fit(fit)
    print(f"wss: {kmeans_model.inertia_}")
    fit['Labels'] = kmeans_model.labels_
    return fit