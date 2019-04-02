import random
import numpy as np


def soft_k_means(dataset, K, iter_num):
    centroids = initial_centroid(dataset,K)
    for _ in range(0,iter_num):
        r = assignment_responsibilities(centroids, dataset)
        centroids = expectation_centroids(dataset,r, K)
    return r, centroids


def assignment_responsibilities(centroids, dataset, beta=1):
    N, _ = dataset.shape
    K, D = centroids.shape
    R = np.zeros((N, K))
    for n in range(N):
        R[n] = np.exp(-beta * np.linalg.norm(centroids - dataset[n], 2, axis=1))
    R /= R.sum(axis=1, keepdims=True)
    return R


def expectation_centroids(dataset, r, K):
    N, D = dataset.shape
    centers = np.zeros((K, D))
    for k in range(K):
        centers[k] = r[:, k].dot(dataset) / r[:, k].sum()
    return centers


def initial_centroid(dataset, K):
    centroids = []
    for i in range(K):
        idx = random.randint(0,len(dataset))
        centroids.append(dataset[idx])
    return np.array(centroids)