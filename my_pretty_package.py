import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import DBSCAN

df = pd.read_csv('data_large.csv', index_col=0)
df = df[~pd.isna(df['vector'])]


def calculate_shannon_entropy(string):
    ent = 0.0
    if len(string) < 2:
        return ent
    size = float(len(string))
    for b in range(128):
        freq = string.count(chr(b))
        if freq > 0:
            freq = float(freq) / size
            ent = ent + freq * np.log(freq)
    return -ent


def filt(text):
    return 2 <= calculate_shannon_entropy(text) <= 4


df['vector'] = df['vector'].map(str.lower)

df = df[df['vector'].map(filt)]
df = df['vector']

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 7), max_features=1000)


vectorizer.fit(df)
X_vec = vectorizer.transform(df)





clustering = DBSCAN(n_jobs=14, eps=0.2, min_samples=100)

clustering.fit(X_vec)


def get_label(clustering, vector):
    core_points = clustering.components_.toarray()
    core_labels = clustering.labels_[clustering.core_sample_indices_]

    distances = euclidean_distances(core_points, [vector]).flatten()
    min_dist_idx = np.argmin(distances)
    min_dist = distances[min_dist_idx]
    if min_dist > clustering.eps:

        return -1
    else:
        return core_labels[min_dist_idx]






