import math
import random
import numpy as np
import copy

# IMPORTANT: DON'T CHANGE OR REMOVE THIS LINE
#            SO THAT YOUR RESULTS CAN BE VISUALLY SIMILAR
#            TO ONES GIVEN IN HOMEWORK FILES
random.seed(5710414)


def manhattan(a, b):
    return sum(abs(val1 - val2) for val1, val2 in zip(a, b))


def generate_random_number():
    return (
        int(random.uniform(0, 256)),
        int(random.uniform(0, 256)),
        int(random.uniform(0, 256)),
    )


class KMeans:
    def __init__(
        self,
        X,
        n_clusters,
        max_iterations=1000,
        epsilon=0.01,
        distance_metric="manhattan",
    ):
        self.X = X
        self.n_clusters = n_clusters
        self.distance_metric = distance_metric

        # for each cluster, init an empty list
        self.clusters = []
        # mean feature vector for each cluster
        self.cluster_centers = []
        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def fit(self):
        # first we need to pick random elements for our cluster centers
        for i in range(self.n_clusters):
            self.cluster_centers.append(generate_random_number())

        for i in range(self.max_iterations):
            print("KMeans iteration: " + str(i + 1))
            # Will contain a list of the points that are associated with that specific cluster
            self.clusters = [[] for _ in range(self.n_clusters)]

            # Loop through each point and check which is the closest cluster
            for point_idx, point in enumerate(self.X):
                idx = self.predict(tuple(point))
                self.clusters[idx].append(point_idx)

            previous_centres = copy.copy(self.cluster_centers)
            self.cluster_centers = []

            for idx, cluster in enumerate(self.clusters):
                if len(self.X[cluster]):
                    new_centroid = tuple(np.mean(self.X[cluster], axis=0))
                else:
                    new_centroid = (0, 0, 0)
                self.cluster_centers.append(new_centroid)

            is_optimal = True
            for idx, centroid in enumerate(self.cluster_centers):
                original_centroid = previous_centres[idx]
                curr = self.cluster_centers[idx]
                value = (
                    math.dist(curr, original_centroid)
                    if self.distance_metric == "euclidian"
                    else manhattan(curr, original_centroid)
                )
                if value > self.epsilon:
                    is_optimal = False

            # break out of the main loop if the results are optimal,
            # ie. the centroids don't change their positions much(more than our epsilon)
            if is_optimal:
                print("Epsilon boundary reached! Halting...")
                break
            if i == self.max_iterations - 1:
                print("Max iterations reached! Halting...")

    def predict(self, instance):
        distances = []
        for point in self.cluster_centers:
            if self.distance_metric == "euclidian":
                dst = math.dist(point, instance)
            else:
                dst = manhattan(point, instance)
            distances.append(dst)
        classification = distances.index(min(distances))
        return classification
