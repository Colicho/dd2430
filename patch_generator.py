import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

class PatchGenerator:
    def __init__(self, num_patches=8, attribute="transmitter"):
        assert attribute in ["transmitter", "receiver", "gain", "hash"]

        self.num_patches = num_patches
        self.attribute = attribute

    def generate_patches(self, paths):
        patches_index = self._generate_patches_index(paths)
        patches = [[] for _ in range(self.num_patches)]
        for i in range(len(patches_index)):
            patches[patches_index[i]].append(paths[i])
        return patches

    def _generate_patches_index(self, paths):
        attribute_getter = {
            "transmitter": lambda path: path.points[0],
            "receiver": lambda path: path.points[-1],
            "gain": lambda path: path.path_gain_db,
            "hash": lambda path: path.hash
        }[self.attribute]
        X = [attribute_getter(path) for path in paths]
        return self._equal_size_kmeans(X)

    def _equal_size_kmeans(self, X, max_iters=300):
        # Perform initial K-means clustering
        kmeans = KMeans(n_clusters=self.num_patches, max_iter=max_iters)
        kmeans.fit(X)
        labels = kmeans.labels_

        # Find the target size for each cluster
        target_size = int(np.ceil(len(X) / self.num_patches))

        # Create a list to keep track of cluster sizes
        cluster_sizes = [0] * self.num_patches

        # Initialize a list to store the points that need to be reassigned
        points_to_reassign = []

        # Calculate cluster sizes and identify points that need reassignment
        for i in range(len(X)):
            cluster_sizes[labels[i]] += 1
            if cluster_sizes[labels[i]] > target_size:
                points_to_reassign.append(i)

        # Iterate until all clusters have equal size
        while len(points_to_reassign) > 0:
            for point_index in points_to_reassign:
                # Find the cluster with the smallest current size
                min_size_cluster = np.argmin(cluster_sizes)

                # Find the nearest cluster center for the point
                nearest_cluster = pairwise_distances_argmin_min([X[point_index]], kmeans.cluster_centers_)[0][0]

                # Reassign the point to the cluster with the smallest current size
                labels[point_index] = min_size_cluster
                cluster_sizes[min_size_cluster] += 1
                cluster_sizes[nearest_cluster] -= 1

                # Remove the point from the list of points to reassign
                points_to_reassign.remove(point_index)

        return labels

