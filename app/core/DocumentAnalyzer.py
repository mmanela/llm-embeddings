import numpy as np 
import os 
import hdbscan
from sklearn.cluster import OPTICS, MiniBatchKMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
import pandas as pd

from core.EmbeddingsModel import EmbeddingsModel

# CLUSTERING_ALGO= 'hdbscan'
CLUSTERING_ALGO = 'MiniBatchKMeans'
# CLUSTERING_ALGO= 'OPTICS'

SNIPPET_LENGTH = 100


class DocumentAnalyzer(object):
    def __init__(self, provider, fileName, wordDictName) -> None:
        self.provider = provider
        self.fileName = fileName
        self.name = os.path.splitext(fileName)[0]
        self.wordDictName = wordDictName
        self._load_models();
        self._build_clusters();
        self._categorize_cluster_centroids();

    def get_summary_data_frame(self):
        df = pd.DataFrame({
            "cluster": self.clusters.labels_,
            "probabilities": self.clusters.probabilities_,
            "hash_code": [embedding_object.hash_code for embedding_object in self.embedding_objects],
            "page_num": [embedding_object.metadata['page'] for embedding_object in self.embedding_objects],
            "snippet": [embedding_object.content[0:SNIPPET_LENGTH] for embedding_object in self.embedding_objects],
            "category": self.categories,
        })
        df = df.query("cluster != -1")  # Remove docs that are not in a cluster
        df = df.sort_values(by=['cluster', 'probabilities'])

        return df

    def render_histogram(self):
        color_palette = sns.color_palette('Paired', self.labelCount + 1)

        # Render histogram of the clusters
        labels = np.array(self.clusters.labels_)
        hist, bin_edges = np.histogram(
            labels, bins=range(-1, self.labelCount + 1))
        fig1 = plt.figure()
        f1 = fig1.add_subplot()
        f1.bar(bin_edges[:-1], hist, width=1, ec="black", color=[
            (0.5, 0.5, 0.5), *color_palette])
        plt.xlim(min(bin_edges), max(bin_edges))
        f1.set_xlabel('Cluster')
        f1.set_ylabel('Count')
        f1.grid(True)
        self._add_cluster_category_legend(
            self.clusters.labels_, self.clusterCategory, color_palette, f1)
        
    def render_cluster_chart(self):
        color_palette = sns.color_palette('Paired', self.labelCount + 1)
 
        # Scatter plot by first projecting the n-dimensional embeddings into 2D
        # then by using the cluster from the cluster to color the points
        cluster_colors = [color_palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in self.clusters.labels_]
        cluster_member_colors = [sns.desaturate(
            x, p) for x, p in zip(cluster_colors, self.clusters.probabilities_)]
        fig2 = plt.figure(figsize=(10, 10))
        projection = TSNE(n_components=3).fit_transform(self.embeddings_list)

        labelStrs = [x[0] if x else 'None' for x in self.categories]
        if (len(projection[0]) == 3):
            ax = fig2.add_subplot(projection='3d')
            ax.scatter(*projection.T, linewidth=0,
                       c=cluster_member_colors, alpha=1, label=labelStrs)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            self._add_cluster_category_legend(
                self.clusters.labels_, self.clusterCategory, color_palette, ax)

        else:
            ax = fig2.add_subplot(projection='2d')
            ax.scatter(*projection.T, s=50, linewidth=0,
                       c=cluster_member_colors, alpha=1, label=labelStrs)
            ax.legend()
            self._add_cluster_category_legend(
                self.clusters.labels_, self.clusterCategory, color_palette, ax)
        
    def show_charts(self):
        plt.show()



    def _load_models(self):
        docModel = EmbeddingsModel(self.provider, self.fileName)
        faiss_index, embedding_objects, embeddings_list = docModel.load_model()
        self.embeddings_list = embeddings_list
        self.embedding_objects = embedding_objects

        dictModel = EmbeddingsModel(self.provider, self.wordDictName)
        words_index, _, _ = dictModel.load_model(True)
        self.words_index = words_index

    def _build_clusters(self):
        self.clusters = None

        if CLUSTERING_ALGO == 'hdbscan':
            self.clusters = hdbscan.HDBSCAN(
                min_samples=1, min_cluster_size=3, metric='euclidean').fit(self.embeddings_list)
        elif CLUSTERING_ALGO == 'MiniBatchKMeans':
            self.clusters = MiniBatchKMeans().fit(self.embeddings_list)
            self.clusters.probabilities_ = [
                1 for i in range(len(self.embeddings_list))]
        elif CLUSTERING_ALGO == 'OPTICS':
            self.clusters = OPTICS(min_samples=3, min_cluster_size=3,
                                   metric='euclidean').fit(self.embeddings_list)
            self.clusters.probabilities_ = [
                1 for i in range(len(self.embeddings_list))]

        self.labelSet = set([x for x in self.clusters.labels_ if x != -1])
        self.labelCount = len(self.labelSet)
        print(f'Number of clusters: {self.labelCount}')

    def _categorize_cluster_centroids(self):
        print(f'Calculating centroids to categorize each with one word')

        self.clusterCategory: dict = {}
        for cluster in self.labelSet:
            print(f'Finding centroid for Cluster {cluster}')

            cluster_embeddings = np.array([x for i, x in enumerate(
                self.embeddings_list) if self.clusters.labels_[i] == cluster])
            mean = cluster_embeddings.mean(axis=0)
            print(f'Centroid for cluster {cluster} is {mean}')

            # Use centroid to find a single word to summarize the cluster
            word_matches = self.words_index.max_marginal_relevance_search_with_score_by_vector(
                embedding=mean, k=3)
            word_and_scores = [(x[0].page_content[0:SNIPPET_LENGTH], x[1])
                               for x in word_matches if x[0].page_content]
            print(f'Words for cluster {cluster} are {word_and_scores}')

            # Use tuple of top two matching words to categorize the groups
            self.clusterCategory[cluster] = f'{word_and_scores[0]},{word_and_scores[1]}'

        self.categories = [self.clusterCategory[x] if x in self.labelSet else None
                           for x in self.clusters.labels_]


    def _add_cluster_category_legend(self, labels, clusterCategory, color_palette, ax):
        handles = []
        for cluster in set(labels):
            if cluster >= 0:
                handles.append(
                    mpatches.Patch(color=color_palette[cluster], label=clusterCategory[cluster]))
            else:
                handles.append(
                    mpatches.Patch(color=(0.5, 0.5, 0.5), label='None'))
        ax.legend(handles=handles)
