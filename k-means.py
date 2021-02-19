import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import random
import math


# return initial point randomly
# this function define initial centroids for starting clustering
def initializer(k, dataSet_length):
    randomlist = []
    for i in range(0, k):
        n = random.randint(0, dataSet_length - 1)
        randomlist.append(n)
    return randomlist


# SSE calculate sse for the performance of clustering algorithm
def SSE(data_set, centroids, clusters, k):
    sse = 0
    for i in range(0, k):
        cluster_objects = []
        for l in range(len(clusters)):
            if clusters[l] == i:
                cluster_objects.append(data_set[l])

        ssec = 0
        for row in cluster_objects:
            ssec += sum(pow(row[j] - centroids[i][j], 2) for j in range(len(row)))
        sse += ssec

    return sse


# iterator do clustering and calculate final sse
# parameters:
# k: number of clusters
# data_set: the data set that must be clustered
# centroids: primary centroid
def iterator(k, data_set, centroids):
    clusters = []
    row_num = 0
    for row in data_set:
        points_list = []
        for j in range(0, k):
            point = centroids[j]
            distance = 0
            for i in range(len(row)):
                distance += pow((row[i] - point[i]), 2)

            euclidean_dis = math.sqrt(distance)
            points_list.append([j, euclidean_dis])

        # assign a class from j
        clusters.append(min(points_list, key=lambda x: x[1])[0])
        row_num += 1

    sse = SSE(data_set, centroids, clusters, k)

    # determine new centroid of each cluster
    new_centroids = []
    for i in range(0, k):
        cluster_objects = []
        for l in range(len(clusters)):
            if clusters[l] == i:
                cluster_objects.append(data_set[l])

        centroid = np.mean(cluster_objects, axis=0).tolist()
        centroid = [round(elem, 3) for elem in centroid]
        new_centroids.append(centroid)

    if new_centroids != centroids:
        return iterator(k, data_set, new_centroids)
    else:
        return clusters, sse


# running k-means algorithm
def run_k_means(k, data_set):
    dataSet_length = len(data_set)
    initial_points = initializer(k, dataSet_length)
    centroids = []
    for initial_point in initial_points:
        centroids.append(data_set[initial_point])

    clusters, sse = iterator(k, data_set, centroids)
    return clusters, sse


# Test k_means algorithm for iris data set 
iris = load_iris()
iris_list = []
for row in iris['data']:
    row_list = []
    for ele in row:
        row_list.append(ele)

    iris_list.append(row_list)

data_set = iris_list
k = 3
k_means_set = []
for i in range(0, 20):
    clusters, sse = run_k_means(3, iris_list)
    k_means_set.append([sse, clusters])

result = min(k_means_set, key=lambda x: x[0])
print(result[0])
print(result[1])
