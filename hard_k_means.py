from random import sample
from collections import defaultdict
import numpy as np


def k_means(K, dataset, max_iteration=30):
    centroids = [dataset[i] for i in sample(range(len(dataset)), K)]
    clusters = None
    for i in range(max_iteration):
        clusters = assignment(dataset, centroids)
        updated_centroids = set(map(lambda x: update_centroid(dataset, x), clusters.values()))
        original_centroids = set(clusters.keys())
        if original_centroids == updated_centroids:
            break
    return clusters


def update_centroid(vectors, members_indexes):
    component_count = len(vectors[0])
    result = [None] * component_count
    for component in range(component_count):
        result[component] = sum((vectors[i][component] for i in members_indexes)) / len(members_indexes)
    return tuple(result)


def assignment(dataset, centroids):
    clusters = defaultdict(list)
    for i, data in enumerate(dataset):
        min_dis = 99999987654321
        min_idx = -1
        for j, centroid in enumerate(centroids):
            dis = euclidean_dis(data, centroid)
            if min_dis > dis:
                min_dis = dis
                min_idx = j
        clusters[min_idx].append(i)
    return clusters


def euclidean_dis(data, centroid):
    return np.linalg.norm(data - centroid)