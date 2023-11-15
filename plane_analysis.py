#!/bin/python
import pylas
import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.linear_model import RANSACRegressor
from sklearn.decomposition import PCA
import point_cloud_handeling as pc_h
import open3d
import math
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr, kendalltau
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# import pycircular


def main():
    boulder4_plane = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder_planar/boulder4_planar.las'
    boulder4_plane_rot = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder_planar/boulder4_planar_rot.las'

    # plane_new = plane_analysis(boulder4_plane)

    # test(boulder4_plane_rot)
    #modelling_residuals(boulder4_plane_rot)
    rose_hist_plot(boulder4_plane_rot)

    # plane_new.write(boulder4_plane_rot)

def plane_analysis(path):
    # load planes
    plane = pylas.read(path)

    plane_points = pc_h.get_xyz(plane)

    plane_points_shift = reposition_pc(plane_points)

    plane_new = plane
    plane_new.header.offsets = [0.,0.,0.]

    plane_new.x = plane_points_shift[:,0]
    plane_new.y = plane_points_shift[:,1]
    plane_new.z = -plane_points_shift[:,2]

    tile_labels = create_grid(plane_points_shift)

    plane_models, residuals = fit_planes(plane_points, tile_labels)

    pc_h.add_dimension(plane_new, 'tile', 'int32', 'tiles for plane analysis', tile_labels)
    pc_h.add_dimension(plane_new, 'residuals', 'f8', 'residuals', residuals)

    print(np.mean(residuals))

    return plane_new

def test(las):
    plane = pylas.read(las)
    factors = ['scan_distance', 'scan_angle', 'incidence_angle', 'angle_difference', 'traj_azimuth', 'scan_azimuth', 'local_azimuth', 'azimuth_difference']
    residuals = plane.residuals
    for factor in factors:
        factor_values = plane[factor]
        residuals_25 = residuals[factor_values > np.quantile(factor_values, 0.25)]
        residuals_75 = residuals[factor_values < np.quantile(factor_values, 0.75)]
        print('factor: {}, high 25 excluded: {}, low 25 excluded: {}'.format(factor, np.mean(residuals_75), np.mean(residuals_25)))


def modelling_residuals(las):
    plane = pylas.read(las)
    df = pd.DataFrame({
        'residuals': plane.residuals,
        'scan_distance': plane['scan_distance'],
        'scan_angle': plane['scan_angle'], 
        'incidence_angle': plane['incidence_angle'], 
        'angle_difference': plane['angle_difference'], 
        'traj_azimuth': plane['traj_azimuth'], 
        'scan_azimuth': plane['scan_azimuth'], 
        'local_azimuth': plane['local_azimuth'], 
        'azimuth_difference': plane['azimuth_difference']
    })
    normal_factors = ['scan_distance', 'scan_angle', 'incidence_angle']
    circular_factors = ['angle_difference', 'traj_azimuth', 'scan_azimuth', 'local_azimuth', 'azimuth_difference']

    for factor in normal_factors:
        pearson_corr = pearsonr(df['residuals'], df[factor])
        spearman_corr = spearmanr(df['residuals'], df[factor])
        kendall_corr = kendalltau(df['residuals'], df[factor])
        print('FACTOR: ', factor)
        print('Pearson: ', np.round(pearson_corr, 2))
        print('Spearman: ', np.round(spearman_corr, 2))
        print('Kendall: ', np.round(kendall_corr, 2))
        sns.set_theme(style="darkgrid")
        g = sns.jointplot(x="residuals", y=factor, data=df,
                  kind="reg", truncate=False,
                  xlim=(0, np.max(df['residuals'])), ylim=(np.min(df[factor]), np.max(df[factor])),
                  color="m", height=7)
    
    # sns.set_theme()
    # df_melt = pd.melt(df, id_vars=['residuals'], var_name='factor', value_name='degrees')
    # h = sns.FacetGrid(df_melt, col='factor', hue='factor', subplot_kws=dict(projection='polar'), height=4.5, sharex=False, sharey=False, despine=False)
    # h.map(sns.histplot, 'degrees')


def rose_hist_plot(las):
    plane = pylas.read(las)
    df = pd.DataFrame({
        'traj_azimuth': plane['traj_azimuth'], 
        'scan_azimuth': plane['scan_azimuth'], 
        'local_azimuth': plane['local_azimuth'], 
        'azimuth_difference': plane['azimuth_difference']
    })

    
    fig, axes = plt.subplots(2, 2, subplot_kw=dict(projection='polar'), dpi=300, layout='tight')
    print(np.concatenate(axes))
    for ax, factor in zip(np.concatenate(axes), list(df.columns)):
        circular_hist(ax, df, factor, gaps=True)



def circular_hist(ax, df, factor, bins=16, density=True, offset=np.pi/2, gaps=True):
    """
    Produce a circular histogram of angles on ax.
    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').
    x : array
        Angles to plot, expected in units of radians.
    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.
    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.
    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.
    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.
    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.
    bins : array
        The edges of the bins.
    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = np.deg2rad(df[factor])
    x = (x+np.pi) % (2*np.pi) - np.pi
    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)
    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)
    # Compute width of each bin
    widths = np.diff(bins)
    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n
    # Plot data on ax
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                     edgecolor='C0', fill=False, linewidth=1)
    # Set the direction of the zero angle
    #ax.set_theta_offset(offset)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(factor)
    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])
    return n, bins, patches


def create_grid(point_cloud, grid_size=1.0):
    min_bound = np.min(point_cloud, axis=0)
    max_bound = np.max(point_cloud, axis=0)
    bounding_box = max_bound - min_bound
    print(bounding_box)

    # voxel_size=gcd(bounding_box[0], bounding_box[1])
    x_min = min_bound[0]
    x_max = max_bound[0]
    y_min = min_bound[1]
    y_max = max_bound[1]

    tile_labels = []
    grid_labels = {}
    grid_label = 0

    for i in np.arange(x_min, x_max, grid_size):
        for j in np.arange(y_min, y_max, grid_size):
            grid_x = int((i - x_min) / grid_size)
            grid_y = int((j - y_min) / grid_size)
            grid_labels[(grid_x,grid_y)] = grid_label
            grid_label += 1

    print(grid_labels)

    for point in point_cloud:
        x, y, z = point
        grid_x = int((x - x_min) / grid_size)
        grid_y = int((y - y_min) / grid_size)
        grid_index = (grid_x, grid_y)
        tile_labels.append(grid_labels[grid_index])

    tile_labels = np.array(tile_labels)
    relable_tiles(point_cloud, tile_labels)

    return tile_labels


def relable_tiles(point_cloud, tile_labels, min_points=200):
    labels, counts =  np.unique(tile_labels, return_counts=True)
    while len(labels[counts<min_points]) != 0:
        for label in labels[counts<min_points]:
            tile = point_cloud[tile_labels == label]
            print(tile_labels == label)
            pc_rest = point_cloud[tile_labels != label]
            # print(pc_rest)
            kdtree = cKDTree(pc_rest)
            idx = kdtree.query(tile, 1)[1]
            print(idx)
            print(tile_labels[tile_labels == label])
            print(tile_labels[tile_labels != label][idx])
            tile_labels[tile_labels == label] = tile_labels[tile_labels != label][idx]

            labels, counts =  np.unique(tile_labels, return_counts=True)



def reposition_pc(point_cloud):
    # Compute PCA
    pca = PCA(n_components=3)
    pca.fit(point_cloud)

    # Get the principal axes (eigenvectors)
    principal_axes = pca.components_
    print(principal_axes)

    # Transform points
    centered_point_cloud = point_cloud - np.mean(point_cloud, axis=0)
    transformed_point_cloud = np.dot(centered_point_cloud, principal_axes.T)
    return transformed_point_cloud


def fit_planes(point_cloud, tile_labels):
    plane_models = []
    residuals = np.zeros(tile_labels.shape)

    for label in np.unique(tile_labels):
        tile = point_cloud[tile_labels == label]
        pca = PCA(n_components=3)
        pca.fit(tile)
        plane_model = pca.components_[2]
        residuals[tile_labels == label] = np.abs(np.dot(tile - np.mean(tile, axis=0), plane_model))
        print(residuals)

        plane_models.append(plane_model)

    return plane_models, residuals


def create_grid2(point_cloud, voxel_size):
    min_bound = np.min(point_cloud, axis=0)
    max_bound = np.max(point_cloud, axis=0)
    grid = []

    for x in np.arange(min_bound[0], max_bound[0], voxel_size):
        for y in np.arange(min_bound[1], max_bound[1], voxel_size):
            for z in np.arange(min_bound[2], max_bound[2], voxel_size):
                indices = np.where((point_cloud["X"] >= x) & (point_cloud["X"] < x + voxel_size) &
                                   (point_cloud["Y"] >= y) & (point_cloud["Y"] < y + voxel_size) &
                                   (point_cloud["Z"] >= z) & (point_cloud["Z"] < z + voxel_size))
                print(indices)

                if len(indices[0]) > 0:
                    voxel = point_cloud[indices]
                    grid.append(voxel)

    return grid

if __name__ == '__main__':
    main()