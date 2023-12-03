
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from point_cloud_handeling import get_xyz, add_dimension

import os
os.environ["_CCTRACE_"]="ON"          # only if you want debug traces from C++
import cloudComPy as cc  

def compute_cosine_deviation(normal1, normal2):
    return np.dot(normal1, normal2) / (np.linalg.norm(normal1) * np.linalg.norm(normal2))

def calculate_inconsistency_metric(normals, neighborhood_indices):
    inconsistency_values = []

    for i, neighbors in enumerate(neighborhood_indices):
        cosine_deviations = [compute_cosine_deviation(normals[i], normals[j]) for j in neighbors]
        mean_cosine_deviation = np.mean(cosine_deviations)
        inconsistency_values.append(np.std(cosine_deviations))

    return np.array(inconsistency_values)

def visualize_inconsistency(pc, inconsistency_values):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    scatter = ax.scatter(pc['X'], pc['Y'], pc['Z'], c=inconsistency_values, cmap='viridis', s=1)
    fig.colorbar(scatter, label='Inconsistency Metric')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def main():
    # Replace 'your_point_cloud.las' with the path to your LAS/LAZ file
    las_file_path = 'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder1_normals/boulder1.las'

    # Read the LAS/LAZ file
    las_file = pylas.read(las_file_path)
    point_cloud = las_file.points

    # Extract X, Y, Z coordinates and normals
    xyz = get_xyz(las_file)
    normals = np.vstack([las_file.Nx, las_file.Ny, las_file.Nz]).T

    # Define the number of neighbors for each point
    num_neighbors = 25

    # Use sklearn's NearestNeighbors to find neighbors
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='kd_tree').fit(xyz)
    _, neighborhood_indices = nbrs.kneighbors(xyz)

    # Calculate inconsistency metric
    inconsistency_values = calculate_inconsistency_metric(normals, neighborhood_indices)

    # Visualize the point cloud with color-coded inconsistency metric
    # visualize_inconsistency(point_cloud, inconsistency_values)

    add_dimension(las_file, 'normal_dev_{}'.format(num_neighbors), 'f8', 'blb', inconsistency_values)

    las_file.write(las_file_path.replace('.las', '_normdev_{}.las'.format(num_neighbors)))

if __name__ == "__main__":
    main()