#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 08:04:09 2023

@author: chloe
"""

""" ----------- Exercise 1 - K-means from scratch ----------- """

# Implementing K-means

import matplotlib.pyplot as plt
import copy
import numpy as np
from numpy.linalg import norm
from sklearn.cluster import KMeans


# 0. Generate dataset

def split_points(points: np.array, n_of_point_groups: int) -> np.array:
    changed_points = copy.copy(points)
    index = np.arange(len(points))
    groups_index = np.split(index, n_of_point_groups)
    
    for id_group,group_index in enumerate(groups_index):
        changed_points[group_index] = points[group_index] + 5*id_group
    
    return changed_points
    

#generate points

n_of_points = 60
points = np.random.rand(n_of_points,2) * 5 
points = split_points(points, 3)

plt.figure()
plt.scatter(points[:,0],points[:,1])

# 1. Specify number of clusters K

k = 3

# 2. initialize centroids

def initialize_clusters(points: np.array, k_clusters: int) -> np.array:
    vector_with_all_indexes = np.arange(points.shape[0])
    vector_with_all_indexes = np.random.permutation(vector_with_all_indexes)    
    required_indexes = vector_with_all_indexes[:k_clusters] 
    
    return points[required_indexes]

# 3. Calculate distance from centroids to all points in datasets

def calculate_metric(points: np.array, centroid: np.array) -> np.array:
    return np.square(norm(points-centroid, axis=1))

def compute_distances(points: np.array, centroids_points: np.array) -> np.array:
    return np.asarray([calculate_metric(points, centroid) for centroid in centroids_points])

# 4. Assign datapoints to the closest centroids

def assign_centroids(distances: np.array):
    return np.argmin(distances, axis = 1)

# 5. Calculate objective function

def calculate_objective(cluster_belongs: np.array, distances: np.array) -> np.array:
    distances = distances.T
    selected_min =  distances[np.arange(len(distances)), cluster_belongs]
    return np.sum(selected_min)

# 6. Compute new controids to the clusters by taking the average of the all data points that belong to each cluster

def calculate_new_centroids(points: np.array, clusters_belongs: np.array, n_of_clusters: int) -> np.array:
    new_clusters = []
    for cluster_id in range(n_of_clusters):
        j = np.where(clusters_belongs == cluster_id)
        points_sel = points[j]
        new_clusters.append(np.mean(points_sel, axis=0))

    return np.array(new_clusters)

# 7. repeat steps 2-6. until there is no change in objective function or until the maximum n of iterations is reached

def fit(points: np.array, n_of_centroids: int, n_of_oterations: int, error: float = 0.001) -> tuple:
    centroid_points = initialize_clusters(points, n_of_centroids)
    last_objective = 10000


    for n in range(n_of_oterations):
        distances = compute_distances(points, centroid_points)
        cluster_belongs = np.argmin(distances, axis=0)
        
        objective = calculate_objective(cluster_belongs, distances)
        
        if abs(last_objective - objective) < error:
            break
      
        last_objective = objective

        centroid_points = calculate_new_centroids(points, cluster_belongs, n_of_centroids)

    return centroid_points, last_objective


# All together

k = 4

n_of_points = 60
n_of_iterations = 200


points = np.random.rand(n_of_points,2) * 5 
points = split_points(points, 3)

def split_points(points: np.array, n_of_point_groups: int) -> np.array:
    changed_points = copy.copy(points)
    index = np.arange(len(points))
    groups_index = np.split(index, n_of_point_groups)
    
    for id_group,group_index in enumerate(groups_index):
        changed_points[group_index] = points[group_index] + 5*id_group
    
    return changed_points


def initialize_clusters(points: np.array, k_clusters: int) -> np.array:
    vector_with_all_indexes = np.arange(points.shape[0])
    vector_with_all_indexes = np.random.permutation(vector_with_all_indexes)    
    required_indexes = vector_with_all_indexes[:k_clusters] 
    
    return points[required_indexes]

def calculate_metric(points: np.array, centroid: np.array) -> np.array:
    return np.square(norm(points-centroid, axis=1))

def compute_distances(points: np.array, centroids_points: np.array) -> np.array:
    return np.asarray([calculate_metric(points, centroid) for centroid in centroids_points])

def assign_centroids(distances: np.array):
    return np.argmin(distances, axis = 1)


def calculate_objective(cluster_belongs: np.array, distances: np.array) -> np.array:
    distances = distances.T
    selected_min =  distances[np.arange(len(distances)), cluster_belongs]
    return np.sum(selected_min)
    

def calculate_new_centroids(points: np.array, clusters_belongs: np.array, n_of_clusters: int) -> np.array:
    new_clusters = []
    for cluster_id in range(n_of_clusters):
        j = np.where(clusters_belongs == cluster_id)
        points_sel = points[j]
        new_clusters.append(np.mean(points_sel, axis=0))

    return np.array(new_clusters)

def fit(points: np.array, n_of_centroids: int, n_of_oterations: int, error: float = 0.001) -> tuple:
    centroid_points = initialize_clusters(points, n_of_centroids)
    last_objective = 10000


    for n in range(n_of_oterations):
        distances = compute_distances(points, centroid_points)
        cluster_belongs = np.argmin(distances, axis=0)
        
        objective = calculate_objective(cluster_belongs, distances)
        
        if abs(last_objective - objective) < error:
            break
      
        last_objective = objective

        centroid_points = calculate_new_centroids(points, cluster_belongs, n_of_centroids)

    return centroid_points, last_objective



centroids, _ = fit(points, 3, n_of_iterations)


plt.figure()
plt.scatter(points[:,0],points[:,1])
plt.scatter(centroids[:,0].T,centroids[:,1].T)


""" ----------- Exercise 2 - Implementing an Elbow Method ----------- """

k_all = range(1, 15)
all_objective = []

for n_of_cluster in k_all:
    _, objective = fit(points, n_of_cluster, n_of_iterations) # we try each number of clusters
    all_objective.append(objective)
    
plt.figure()
plt.plot(k_all, all_objective)
plt.xlabel('K clusters')
plt.ylabel('Sum of squared distance')
plt.show()

print("We choose the k-number that can separate the values into clusters but we dont' want too much clusters.")

print("All values :", all_objective)


""" ----------- Exercise 3 - Image Compression ----------- """

from matplotlib.image import imread
from sklearn.cluster import KMeans
import numpy as np

# 1. load image

loaded_image = imread('/Users/chloe/Documents/BRNO/Machine Learning/TP/fish.jpg')

plt.imshow(loaded_image)
plt.show()

def compress_image(image: np.array, number_of_colours: int) -> np.array:
    original_shape = image.shape
    
    image_reshaped = np.reshape(image, (image.shape[0]*image.shape[1],image.shape[2]))
    kmeans = KMeans(n_clusters=number_of_colours, random_state=0).fit(image_reshaped)
    image_new = kmeans.predict(image_reshaped)  # we make clusters for each color
    

    cluster_labels = kmeans.labels_
    colors = kmeans.cluster_centers_.astype('uint8')
    
    new_image = colors[cluster_labels].reshape(original_shape)
    
    
    return new_image

img = compress_image(loaded_image, 10)  # we choose the number of color we use

plt.figure()
plt.imshow(img)
plt.show()
