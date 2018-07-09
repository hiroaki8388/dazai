import nltk
from nltk.cluster import KMeansClusterer
from scipy.cluster.hierarchy import dendrogram
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt


class KMeansTopics(BaseEstimator, TransformerMixin):
    def __init__(self, k=10):
        """
        :param k: クラスタ数　int
        """
        cosine = nltk.cluster.util.cosine_distance
        self.model = KMeansClusterer(
            k, distance=cosine, avoid_empty_clusters=True)

    def fit(self, sents):
        return self

    def transform(self, sents):
        self.model.cluster(sents)


class HierachicalTopics(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.model = AgglomerativeClustering()

    def fit(self):
        return self

    def transform(self, sents):
        clusters = self.model.fit_predict((sents))
        self.labels = self.model.labels_
        self.children = self.model.children_

        return clusters

    def plot_dendrogram(self, **kwargs):
        dis = np.arange(self.children.shape[0])
        pos = np.arange(self.children.shape[0])

        linkage_matrix = np.column_stack([
            self.children, dis, pos
        ]).astype(float)

        fig, ax = plt.subplots()
        ax = dendrogram(linkage_matrix, **kwargs, ax=ax)
        plt.show()
