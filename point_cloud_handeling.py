#!/bin/python
import pdal
from osgeo import ogr
from shapely import Polygon, Point, LineString
import geopandas as gpd
import pandas as pd
import glob
import os
import sys
import pylas
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
import rasterio as rio
import scipy
import math as m
import matplotlib.pyplot as plt
import json
from math import pi
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.stats import ttest_ind, wilcoxon
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score, recall_score, balanced_accuracy_score
from scipy.stats import norm
import time
# from utils import *


def main():
    # path_raster = r"E:/NATUR_CUNI/_DP/data/LAZ/raster/ahk_uls_pdalmerge_aoi_first_raster.tif"
    # path_raster2 = r"E:/NATUR_CUNI/_DP/data/LAZ/raster/ahk_uls_pdalmerge_aoi_raster.tif"
    path_las = r'E:/NATUR_CUNI/_DP/data/LAZ/ahk_uls_pdalmerge_aoi.laz'
    # profil_shp = r'E:/NATUR_CUNI/_DP/data/profil.shp'
    profil_las = r'E:/NATUR_CUNI/_DP/data/LAZ/ahk_uls_pdalmerge_aoi_profil.laz'
    lower_rongue_las = r'E:/NATUR_CUNI/_DP/data/LAZ/lower_tongue/ahk_uls_pdalmerge_aoi_lower_tongue_split.laz'
    boulder1 = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/first/boulder1.laz'
    boulder2 = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/first/boulder2.laz'
    boulder3 = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/first/boulder3.laz'
    boulder4 = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/first/boulder4.laz'
    pentahedron = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/first/pentahedron.las'
    lower_tongue_shp = r'E:/NATUR_CUNI/_DP/data/lower_tongue.shp'
    strips_path = r'E:/NATUR_CUNI/_DP/data/LAZ/lower_tongue/ahk_uls_pdalmerge_aoi_lower_tongue_split_*.laz'
    striptest = r'E:/NATUR_CUNI/_DP/data/LAZ/strip_adjust/striptest_4_t.las'

    tile = r'E:/NATUR_CUNI/_DP/data/LAZ/tile_652583_5188541.laz'

    test_boulder = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/first/boulder1_filtered2.laz'

    boulder_noise = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder4_merged_full.laz'

    # boulder2_noise = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder2_noise.las'
    boulder2 = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder2/'

    dim_path = r'E:/NATUR_CUNI/_DP/data/DIM/2021_ahk_DIM_coreg.laz'

    # classify_7(boulder2_noise)

    #pc_merge(boulder2, 'boulder2_merge.las')

    # pc_fixstrip(lower_rongue_las)
    #pc_fixstrip(strips_path)
    #gps_time(striptest)

    #output_path = r'E:/NATUR_CUNI/_DP/data/LAZ/lower_tongue/stripAdjust/strips'
    #pc_merge(output_path, 'ahk_uls_pdalmerge_aoi_lower_tongue_split_adjusted.laz')

    # raster_to_histo(path_raster, output_path)
    # raster_to_histo(path_raster2, output_path)
    #pc_clip(dim_path, lower_tongue_shp)
    # pc_raster(path_las)
    # las_to_histo(path_las, output_path)
    # thinning(boulder1, _, _)¨
    
    # trajectory_path = r'E:/NATUR_CUNI/_DP/data/Trajectory/HEK_trajectory.csv'

    # pc_trajectory(trajectory_path)

    #_, _ = pick_scanner_position(boulder1, trajectory_path)

    # pc_angles(boulder3, trajectory_path, 50)

    # pc_normal(boulder1, 60)

    #pc_sor(boulder1, 50, 1.5, boulder1.replace('.laz', '_filtered.laz'))
    # pc_sor2(boulder1, 50, 1.5, boulder1.replace('.laz', '_filtered2.laz'))
    # train_sor(boulder_noise)

    # noise_evaluation(boulder3, trajectory_path, 0.15, 50, 1.0)
    # pc_visualize(boulder1)

    # process_point_cloud(boulder4)

    # generate_pentahedron_point_cloud(pentahedron, 100)

    # point1 = gpd.GeoSeries([Point(0., -1., -1.)])
    # point2 = gpd.GeoSeries([Point(0., 0., 0.)])
    # rot_angles = np.array([-pi/2, 0., 0.]).transpose()
    # , Point(0., 0., 1.)
    # , Point(0., 0., 1.)

    # noise_evaluation(pentahedron, trajectory_path, 0.5, 50, 1.0)

    # planes, planes_consist = normaldefinition_3D_real(boulder3, 50)
    #print(pd.Series(Point(0., 1., 0.)).x)
    # results, v2 = compute_scan_angle(point1, point2, rot_angles)

    # normalX = pd.Series([0.])
    # normalY = pd.Series([0.])
    # normalZ = pd.Series([1.])
    # normal = np.array([normalX, normalY, normalZ]).transpose()

    # incidence_angle = compute_incidence_angle(normal, -v2)

    # gps_time(boulder3)
    # pc_strip_split(lower_rongue_las)

    local_azimuth = np.arctan2(1., 0.)/m.pi*180
    deltaX = np.array(0 - 1)
    deltaY = np.array(0 - 0)
    scan_azimuth = np.arctan2(deltaX, deltaY)/m.pi*180
    print(abs((45+180) % 360. - 180.))


def add_gaussian_noise(las_path, amount=0.1, noise_mean=0, noise_std=0.1):
    las = pylas.read(las_path)
    
    total_points = len(las.points)
    num_points_to_move = int(0.1*total_points)
    idx = np.random.choice(total_points, num_points_to_move, replace=False)
    print(las.points)
    
    '''
    las.points['X'][idx] += np.random.normal(noise_mean, noise_std, num_points_to_move)
    las.points['Y'][idx] += np.random.normal(noise_mean, noise_std, num_points_to_move)
    las.points['Z'][idx] += np.random.normal(noise_mean, noise_std, num_points_to_move)

    las.points['classification'][idx] = 7


    output_file = las_path.replace('.las', 'noise_{}_{}_{}.las'.format(amount,noise_mean,noise_std))
    las.write(output_file)
    '''


def jaccard_similarity(set1, set2):
    # intersection of two sets
    intersection = len(set1.intersection(set2))
    # Unions of two sets
    union = len(set1.union(set2))
     
    return intersection / union

def circular_hist(ax, x, bins=16, density=True, offset=np.pi/2, gaps=True):
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
    ax.set_theta_offset(offset)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])
    return n, bins, patches

def get_xyz(las):
    return np.vstack((las.x, las.y, las.z)).T

def classify_7(las_path):
    las = pylas.read(las_path)
    las.classification[:] = 7
    las.write(las_path)
    print('done')
    
def train_sor(las_path):
    start = time.time()
    point_cloud = lp.read(las_path)
    # save classification 
    y_true = np.array(list(point_cloud.classification))
    y_true[y_true == 2] = 0
    y_true[y_true == 1] = 0
    y_true[y_true == 7] = 1
    print(y_true)
    # read point cloud
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)



    knn_range = range(5, 256, 50)
    std_range = np.arange(0.1, 5.2, 1.0)
    print(list(knn_range))
    print(std_range)
    cont_table = np.ndarray((len(knn_range), len(std_range)), np.float64)

    # print(cont_table)
    for i, knn in enumerate(knn_range):
        for j, std_ratio in enumerate(std_range):
            _, pc_indices = pcd.remove_statistical_outlier(nb_neighbors=knn, std_ratio=std_ratio)
            y_pred = np.ones(len(y_true), np.int8)
            y_pred[pc_indices] = 0
            # print(y_pred)
            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='binary')
            prec = precision_score(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            # print(knn, std_ratio, acc)
            cont_table[i,j] = f1
    print(cont_table)
    end = time.time()
    runtime = end-start
    print(runtime)

    min_acc = np.amin(cont_table)
    max_acc = np.amax(cont_table)
    fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    plt.clf()

    ax = fig.add_subplot(111)

    ax.set_aspect(1)

    res = sns.heatmap(cont_table, annot=True, fmt='.2f', cmap="YlGnBu", vmin=min_acc, vmax=max_acc)
    plt.xticks(np.arange(0.5, cont_table.shape[0] + 0.5), [round(i, 1) for i in std_range])
    plt.yticks(np.arange(0.5, cont_table.shape[1] + 0.5), list(knn_range))
    plt.xlabel('std_ratio')
    plt.ylabel('knn')

    plt.title('Statistical outlier filter parameters - recall',fontsize=12)
    plt.show()

def generate_pentahedron_point_cloud(filepath, num_points, height=1.0, width=1.0):
    # Define the coordinates of the vertices
    vertices = np.array([
        [0.0, 0.0, height],
        [width/2, width/2, 0.0],
        [width/2, -width/2, 0.0],
        [-width/2, -width/2, 0.0],
        [-width/2, width/2, 0.0]
    ])

    # Define the indices of the triangles that make up each facet
    facets = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [0, 4, 1],
    ])

    # Generate the point cloud by generating points within each triangle
    points = []
    for i in range(facets.shape[0]):
        # Get the coordinates of the vertices of the current triangle
        v1 = vertices[facets[i, 0]]
        v2 = vertices[facets[i, 1]]
        v3 = vertices[facets[i, 2]]
        # Generate points within the current triangle
        for j in range(num_points):
            for k in range(num_points-j):
                # Compute the barycentric coordinates of the current point
                u = float(j) / float(num_points-1)
                v = float(k) / float(num_points-1)
                w = 1.0 - u - v
                # Compute the coordinates of the current point
                point = u*v1 + v*v2 + w*v3
                points.append(point)

    # Flatten the points into a single array
    point_cloud = np.array(points).reshape(-1, 3)
    print(point_cloud)

    # Create a LAS file with the point cloud data
    new_hdr = lp.LasHeader(version="1.4", point_format=6)
    # You can set the scales and offsets to values tha suits your data
    new_las = lp.LasData(new_hdr)

    new_las.x = point_cloud[:, 0]
    new_las.y = point_cloud[:, 1]
    new_las.z = point_cloud[:, 2]

    if filepath is not None:
        new_las.write(filepath)
        

def noise_evaluation(las_path, trajectory_path, radius, knn, std_ratio):
    # las_path_final = las_path.replace('laz', 'las')
    las = lp.read(las_path)

    # calculate normals and sor
    pc_normal_sor(las, radius, knn, std_ratio)

    # add a position of the scanner to a point and calculate distance
    gdf = pick_scanner_position(las, trajectory_path)
   
    pc_angles_azimuths(las, gdf)

    gdf['Classification'] = pd.Series(np.array(las.classification).transpose())

    clean = gdf[gdf['Classification'] != 7]
    noise = gdf[gdf['Classification'] == 7]

    stats_clean_scan_angle = summary_stats(clean['scan_angle'])
    stats_noise_scan_angle = summary_stats(noise['scan_angle'])
    
    # print('stats', stats_clean_scan_angle)
    # print('stats', stats_noise_scan_angle)

    # Select the columns you want to use for PCA
    columns = ['scan_angle', 'incidence_angle', 'angle_difference', 'traj_azimuth', 'scan_azimuth', 'local_azimuth', 'azimuth_difference', 'scan_distance', 'intensity']

    noise_df = noise[columns]
    print(noise_df.shape)
    clean_df = clean[columns]
    print(clean_df.shape)
    # Extract the columns of interest

    # Perform PCA
    # pca_noise(clean_df, noise_df)

    # do statistical tests
    compare_datasets(clean_df, noise_df)

    # plot boxplots
    boxplots(clean_df, noise_df)

    ks_test(clean_df, noise_df)

    # scatter_plots(clean_df, noise_df)

    # lr_rf(clean_df, noise_df)

    las.write(las_path_final)


def pca_noise(clean, noise):

    columns = clean.columns
    # Perform PCA
    pca_noise = PCA()
    pca_noise.fit(noise)
    pca_clean = PCA()
    pca_clean.fit(clean)

    # Retrieve the loadings for each attribute on the first principal component
    loadings_PCA1_noise = pca_noise.components_[0]
    loadings_PCA1_clean = pca_clean.components_[0]
    loadings_PCA2_noise = pca_noise.components_[1]
    loadings_PCA2_clean = pca_clean.components_[1]

    # Print the loadings for each attribute
    print('PCA1 loadings')
    for i, column in enumerate(columns):
        print(f'{column}: {loadings_PCA1_noise[i]:.3f}, {loadings_PCA1_clean[i]:.3f}')

    # Print the loadings for each attribute
    print('PCA2 loadings')
    for i, column in enumerate(columns):
        print(f'{column}: {loadings_PCA2_noise[i]:.3f}, {loadings_PCA2_clean[i]:.3f}')


def boxplots(clean_df, noise_df):
    print('I am here')
    attributes = noise_df.columns
    print(attributes)
    noise_df['Group'] = 'Noise'
    clean_df['Group'] = 'Clean'
    combined_df = pd.concat([clean_df, noise_df], ignore_index=True)
    # print(combined_df)
    # melted_df = pd.melt(combined_df, var_name='Attribute', value_name='Value', ignore_index=True)
    # print(melted_df)

    n_cols = 3
    n_rows = int(np.ceil(len(attributes) / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15,5*n_rows))
    axs = axs.flatten()

    for i, attribute in enumerate(attributes):
        sns.boxplot(data=combined_df, y='Group', x=attribute, ax=axs[i])
        # sns.boxplot(data=noise_df, y='Group', x=attribute, ax=axs[i], color='red')
        axs[i].set_ylabel(attribute)
        axs[i].set_title(attribute)

    plt.tight_layout()
    plt.show()


def scatter_plots(clean_df, noise_df):
    combined_df = pd.concat([clean_df, noise_df], ignore_index=True)

    sns.jointplot(data=combined_df, x="scan_angle", y="incidence_angle", hue="Group", kind="hist")
    sns.jointplot(data=combined_df, x="scan_azimuth", y="local_azimuth", hue="Group", kind="hist")
    sns.jointplot(data=combined_df, x="angle_difference", y="azimuth_difference", hue="Group", kind="hist")
    sns.jointplot(data=combined_df, x="scan_distance", y="intensity", hue="Group", kind="hist")


def lr_rf(clean_df, noise_df):
    merge_df = pd.concat([clean_df, noise_df], ignore_index=True)
    print(merge_df.columns)
    labels = merge_df['Group']
    X = merge_df.loc[:, ~merge_df.columns.isin(['Group', 'traj_azimuth', 'scan_azimuth', 'local_azimuth'])]
    y = labels
    log_reg = LogisticRegression()
    X = sm.add_constant(X) # add a constant term for the intercept

    # check for linearity
    sns.regplot(x=X[:, 1], y=model.resid_response, lowess=True, line_kws={'color': 'red'})
    sns.regplot(x=X[:, 2], y=model.resid_response, lowess=True, line_kws={'color': 'blue'})

    # check for independence
    sns.residplot(model.predict(), model.resid_response, lowess=True, line_kws={'color': 'red'})

    # check for multicollinearity
    vif = [sm.OLS(X[:, i], X[:, np.arange(X.shape[1]) != i]).fit().rsquared for i in range(X.shape[1])]
    
    log_reg.fit(X, y)
    odds_ratios = pd.DataFrame({'Attribute': X.columns, 'Odds Ratio': log_reg.coef_[0]})
    print(odds_ratios)

    rf = RandomForestClassifier()
    rf.fit(X, y)
    feature_importances = pd.DataFrame({'Attribute': X.columns, 'Importance': rf.feature_importances_})
    print(feature_importances)

    # (Optional) Evaluate the performance of the model
    # y_pred = rf.predict(X)
    # accuracy = accuracy_score(y, y_pred)
    # confusion = confusion_matrix(y, y_pred)


def compare_datasets(clean_df, noise_df):
    
    """
    Compare two datasets using statistical tests and plot the data.
    
    Parameters:
        noise_df (DataFrame): A pandas DataFrame containing the noisy data.
        clean_df (DataFrame): A pandas DataFrame containing the clean data.
    
    Returns:
        None
    """
    # Check input shape
    if noise_df.shape[1] != clean_df.shape[1]:
        raise ValueError('The input datasets must have the same number of columns.')
    
    # Determine subplot layout
    n_cols = 3
    n_rows = int(np.ceil(noise_df.shape[1] / n_cols))

    # Perform statistical tests and plot data for each column
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    for i, col in enumerate(noise_df.columns):
        print('=====atribute {}====='.format(col))
        noise = noise_df[col]
        clean = clean_df[col]

        # Check normality using Shapiro-Wilk test
        _, noise_normality = stats.shapiro(noise)
        _, clean_normality = stats.shapiro(clean)
        print('Normality test (Shapiro-Wilk):')
        print(f'Noise: p = {noise_normality:.4f}')
        print(f'Clean: p = {clean_normality:.4f}')

        # Check homogeneity of variance using Levene's test
        _, variance_homogeneity = stats.levene(noise, clean)
        print('Homogeneity of variance test (Levene):')
        print(f'p = {variance_homogeneity:.4f}')

        if noise_normality < 0.05 or clean_normality < 0.05:
            print('Data may not be normally distributed')
            if variance_homogeneity < 0.05:
                print('Variance may not be homogeneous')
                test_name = 'Wilcoxon rank-sum test'
                # stat, pval = stats.wilcoxon(noise, clean)
                stat, pval = stats.mannwhitneyu(clean, noise, alternative="two-sided")
            else:
                test_name = 'Welch\'s t-test'
                stat, pval = stats.ttest_ind(clean, noise, equal_var=False)
        else:
            print('Data is normally distributed')
            if variance_homogeneity < 0.05:
                print('Variance may not be homogeneous')
                test_name = 'Welch\'s t-test'
                stat, pval = stats.ttest_ind(clean, noise, equal_var=False)
            else:
                print('Variance is homogeneous')
                test_name = 'Student\'s t-test'
                stat, pval = stats.ttest_ind(noise, clean)
        
        # Compute effect size using Cohen's d
        n1, n2 = len(noise), len(clean)
        s1, s2 = np.var(noise, ddof=1), np.var(clean, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        d = (np.mean(noise) - np.mean(clean)) / pooled_std

        # Compute power of the test
        z_alpha = norm.ppf(0.975) # Two-tailed 95% significance level
        z_beta = (abs(stat) / np.sqrt(n1 + n2)) - z_alpha
        power = norm.cdf(z_beta)
        
        print('Performing {}:'.format(test_name))
        print(f'Test statistic = {stat:.4f}')
        print(f'p-value = {pval:.4f}')
        print(f'Effect size (Cohen\'s d) = {d:.4f}')
        print(f'Power of the test = {power:.4f}')
        effect_size = cohen_eval(d)
          

        # Plot the data
        sns.histplot(data=clean_df, x=col, color='blue', alpha=0.5, label='Clean', ax=axes[i])
        sns.histplot(data=noise_df, x=col, color='red', alpha=0.5, label='Noise', ax=axes[i])
        # Print statistical test results
        axes[i].set_title(f'{col} ({test_name} p-value: {pval:.4f})')
        if pval < 0.05:
            axes[i].set_xlabel('Statistically different {}'.format(effect_size))
        else:
            axes[i].set_xlabel('Statistically not different {}'.format(effect_size))
        axes[i].legend()

    # Hide any unused subplots
    for i in range(noise_df.shape[1], n_rows*n_cols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def cohen_eval(d):
    if d < 0.0:
        dir = 'negative'
    else:
        dir = 'positive'
    if d >= 0.8:
        comment = f'(d = {d:.4f} - very large {dir} effect size)'
    elif 0.8 < abs(d) >= 0.5:
        comment = f'(d = {d:.4f} - large {dir} effect size)'
    elif 0.5 > abs(d) >= 0.2:
        comment = f'(d = {d:.4f} - medium {dir} effect size)'
    else:
        comment = f'(d = {d:.4f} - small {dir} effect size)'
    return comment
    

def ks_test(clean_df, noise_df):
    for i, col in enumerate(noise_df.columns):
        stat, pval = stats.kstest(clean_df[col], noise_df[col])
        print(col)
        print(stat, pval)
        if pval < 0.05:
            print('yes')


def pc_angles_azimuths(las, gdf):

    # calculate angles
    pc_angles(gdf, las)

    gdf['traj_azimuth'] = gdf['Yaw[deg]']  

    # trajectory azimuth
    add_dimension(las, 'traj_azimuth', 'Azimuth of trajectory', gdf['Yaw[deg]'])

    # scan azimuth
    scan_azimuth = pc_scan_azimuth(gdf['geometry_y'], gdf['geometry_x'])

    gdf['scan_azimuth'] = scan_azimuth

    add_dimension(las, 'scan_azimuth', 'Azimuth between point/scanner', gdf['scan_azimuth'])

    local_azimuth = pc_local_azimuth(gdf['NormalY'], gdf['NormalX'])

    gdf['local_azimuth'] = local_azimuth

    add_dimension(las, 'local_azimuth', 'Local azimuth od surface', gdf['local_azimuth'])

    gdf['azimuth_difference'] = abs((gdf['scan_azimuth']+gdf['local_azimuth']) % 360. - 180.)

    add_dimension(las, 'azimuth_difference', 'difference of azimuths', gdf['azimuth_difference'])


def pc_local_azimuth(normalY, normalX):
    degrees = np.arctan2(normalX, normalY)/m.pi*180
    # degrees[degrees<0] += 360
    return degrees


def pc_scan_azimuth(destination, origin):
    
    deltaX = np.array(destination.x - origin.x)
    deltaY = np.array(destination.y - origin.y)
    degrees = np.arctan2(deltaX, deltaY)/m.pi*180
    # degrees[degrees<0] += 360.


    return degrees


def pick_scanner_position(las, trajectory_path):
    df_trj = pd.read_csv(trajectory_path)
    gdf_trj = gpd.GeoDataFrame(df_trj, geometry=gpd.points_from_xy(df_trj['Easting[m]'], df_trj['Northing[m]'], df_trj['Height[m]']))
    las_gps = np.array(las.gps_time)
    las_intensity = np.array(las.intensity)
    gdf_las = gpd.GeoDataFrame({'gps_time': las_gps, 'intensity': las_intensity}, geometry=gpd.points_from_xy(las.x, las.y, las.z))
    for dim in ['NormalX', 'NormalY', 'NormalZ']:
        gdf_las[dim] = las[dim]
    gdf_las['las_index'] = gdf_las.index
    merge_gdf = pd.merge_asof(gdf_las.sort_values('gps_time'), gdf_trj, left_on='gps_time', right_on='Time[s]', direction='nearest')
    
    # geometry_x = point cloud, geometry_y = trajectory
    merge_gdf['scan_distance'] = merge_gdf['geometry_x'].distance(merge_gdf['geometry_y'])
    merge_gdf.index = merge_gdf['las_index']
    sort_gdf = merge_gdf.sort_index()
    # print('geometry_x ', sort_gdf['geometry_x'])
    add_dimension(las, 'scan_distance', "Distance from scanner", merge_gdf['scan_distance'])

    return sort_gdf

def pc_fixstrip(strips_path):
    strips_list = glob.glob(strips_path)
    print(strips_list)
    for las_path in strips_list:
        las_new3 = lp.read(las_path)
        strip = int(os.path.split(las_path)[1][-6:-4].replace('_',''))
        print(strip)

        # las_new = las[las['point_source_id'] != 19]
        # print(np.unique(las['point_source_id']))
        # las_new2 = las_new[las_new['point_source_id'] != 26]
        # print(np.unique(las_new2['point_source_id']))
        # las_new3 = las_new2[las_new2['point_source_id'] != 30]
        # print(np.unique(las_new3['point_source_id']))


        # for i in [18, 17, 16, 15, 14, 13]:
        #    print(i)
        #    las_new3['point_source_id'][las_new3['point_source_id'] == i] += 1

        #for i in range(27,30):
        #    print(i)
        #    las_new3['point_source_id'][las_new3['point_source_id'] == i] -= 1
        # print(np.unique(las_new3['point_source_id']))
        # las_new3['point_source_id'][las_new3['point_source_id'] == 31] -= 2
        # print(np.unique(las_new3['point_source_id']))
        las_new3['point_source_id'] == strip

        
        las_new3.write(las_path)


def pc_strip_split(las_path):
    output_path = las_path.replace('.laz', '_split.laz')
    # Load the point cloud from a LAS file
    las = lp.read(las_path)
    print(list(las.point_format.dimension_names))
    gps_time = las['gps_time']
    sorted_ind = np.argsort(gps_time)
    gps_time = np.sort(gps_time)
    las.points = las.points[sorted_ind]
    print(gps_time)
    
    low_index = 0
    strip = 1
    flight = 100
    for i in range(len(gps_time)-1):
        d_gps = gps_time[i+1]-gps_time[i]
        if d_gps > 2.:
            # if d_gps > 500.:
                # print(d_gps)
                # flight += 100
                # strip = 1
            # code = int(strip + flight)
            # print(gps_time[i], gps_time[i+1])
            # print(gps_time[i+1]-gps_time[i])
            # print(code)
            las['point_source_id'][low_index:i+1] = strip
            print(strip, las['point_source_id'][i])
            strip += 1
            low_index = i + 1
    las['point_source_id'][low_index:i+2] = strip
    print(strip, las['point_source_id'][i])
    
    print(las['point_source_id'])

    las.write(output_path)

    pc_groupby(output_path)

def pc_groupby(las_path):

    output_path = las_path.replace('.las', '_#.las')
    pipeline = pdal.Pipeline(json.dumps({
        "pipeline": [
            {
                "type": "readers.las",
                "filename": las_path
            }
            ,
            {
                "type": "filters.groupby",
                "dimension": "PointSourceId"
            },
            {
                "type": "writers.las",
                "compression": "laszip",
                "filename": output_path,
                "extra_dims": "all"
            }
        ]
    }))

    pipeline.execute()

    print("done")
    

def add_dimension(las, dim_name, dim_type, dim_description, dim_values):
    try:
        las[dim_name] = dim_values
    except ValueError:
        try:
            las.add_extra_dim(
                    name=dim_name,
                    type=dim_type,
                    description=dim_description
            )
        except ValueError:
            print('dimension {} already exists'.format(dim_name))
        else:
            las[dim_name] = dim_values
    finally:
        print(list(las.point_format.dimension_names))


def pc_sor2(las_path, knn, std_ratio, outfile):
    start = time.time()
    point_cloud = lp.read(las_path)
    
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals for the filtered point cloud
    # pcd.estimate_normals(
    #    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))
    
    # Orient normals to a consistent direction based on the tangent plane
    # pcd.orient_normals_consistent_tangent_plane(k=200)

    # Apply a statistical outlier removal filter to flag points as noise
    _, pc_indices = pcd.remove_statistical_outlier(nb_neighbors=knn, std_ratio=std_ratio)

    end = time.time()
    runtime = end-start
    print(runtime)

    # print(list(pcd.normals))
    # NormalX, NormalY, NormalZ = zip(*pcd.normals)
    # normal_columns = ['NormalX', 'NormalY', 'NormalZ']
    # normals = pd.DataFrame(pcd.normals, columns=normal_columns)
    #print(pcd_filtered)
    #print(noise_indices)

    # for dim in normal_columns:
    #    add_dimension(point_cloud, dim, 'Normal vector', normals[dim])

    #noise_indices = list(set(range(len(point_cloud.x)))-set(pc_indices))
    #noise_indices = [item for item in range(len(point_cloud.x)) if item not in pc_indices]
    #point_cloud.classification[point_cloud.classification == 7] = 1
    #point_cloud.classification[noise_indices] = 7
    y_pred = np.full(len(point_cloud.classification), 7, np.int8)
    y_pred[pc_indices] = 1
    point_cloud.classification = y_pred
    point_cloud.write(outfile)


def pc_sor(path, knn, multiplier, output_path):
    start = time.time()
    pipeline = pdal.Pipeline(json.dumps({
        "pipeline": [
            {
                "type": "readers.las",
                "filename": path
            }
            ,
            {
                "type": "filters.outlier",
                "method": "statistical",
                "mean_k": knn,
                "multiplier": multiplier
            },
            {
                "type": "writers.las",
                "compression": "laszip",
                "filename": output_path,
                "extra_dims": "all"
            }
        ]
    }))

    pipeline.execute()
    end = time.time()
    print("done")
    runtime = end-start
    print(runtime)

def gps_time(path):
    # trajectory = pd.read_csv(trajectory_path)
    # gps_time_traj = trajectory['Time[s]']

    # Load the point cloud from a LAS file
    inFile = lp.read(path)

    # Get the GPSTime attribute as a numpy array
    gps_time = inFile.gps_time
    gps_time.sort()
    print(gps_time)

    print(len(gps_time))
    count=0
    for i in range(len(gps_time)-1):
        if gps_time[i+1]-gps_time[i] > 2:
            count += 1
            # print(gps_time[i], gps_time[i+1])
            print(gps_time[i+1]-gps_time[i])

    # print(count)

    # Create a simple line graph of the GPSTime attribute
    # plt.plot(gps_time)
    # plt.plot(gps_time_traj)
    # plt.xlabel("Point Index")
    # plt.ylabel("GPSTime")
    # plt.show()


def pc_greedyprojection(path):
    pipeline = pdal.Pipeline(json.dumps({
        "pipeline": [
            {
                "type": "readers.las",
                "filename": path
            },
            {
                "type": "filters.greedyprojection",
                "multiplier": 2,
                "radius": 1.0
            },
            {
                "type": "writers.ply",
                "faces": True,
                "filename": path.replace(".las", ".ply")
            }
        ]

    }))

    pipeline.execute()

    print("done")


def pc_angles(gdf, las):

    # calculate scan angle
    roll = gdf['Roll[deg]'].apply(m.radians)
    pitch = gdf['Pitch[deg]'].apply(m.radians)
    yaw = gdf['Yaw[deg]'].apply(m.radians)
    rot_angles = np.array([roll, pitch, yaw]).transpose()
    scan_angle, v2 = compute_scan_angle(gdf['geometry_x'], gdf['geometry_y'], rot_angles)

    # save scan angle to gdf and to las
    gdf['scan_angle'] = scan_angle

    add_dimension(las, 'scan_angle', "Scanning angle", gdf['scan_angle'])

    # print(gdf.columns)
           
    normals = np.array([gdf['NormalX'], gdf['NormalY'], gdf['NormalZ']]).transpose()

    incidence_angle = compute_incidence_angle_new(normals, -v2)
    gdf['incidence_angle'] = incidence_angle

    angle_difference = scan_angle - incidence_angle

    gdf['angle_difference'] = angle_difference

    add_dimension(las, 'incidence_angle', 'Incidence angle of given point', incidence_angle)
    add_dimension(las, 'angle_difference', 'Diff of scan and incidence angle', angle_difference)


def pc_angles_new(gdf, las):

    # calculate scan angle
    roll = gdf['Roll[deg]'].apply(m.radians)
    pitch = gdf['Pitch[deg]'].apply(m.radians)
    yaw = gdf['Yaw[deg]'].apply(m.radians)
    rot_angles = np.array([roll, pitch, yaw]).transpose()
    scan_angle, v2 = compute_scan_angle_new(gdf['geometry_x'], gdf['geometry_y'], rot_angles)

    # save scan angle to gdf and to las
    gdf['scan_angle'] = scan_angle

    add_dimension(las, 'scan_angle', "Scanning angle", gdf['scan_angle'])

    # print(gdf.columns)
           
    normals = np.array([gdf['NormalX'], gdf['NormalY'], gdf['NormalZ']]).transpose()

    incidence_angle = compute_incidence_angle_new(normals, -v2)
    gdf['incidence_angle'] = incidence_angle

    angle_difference = scan_angle - incidence_angle

    gdf['angle_difference'] = angle_difference

    add_dimension(las, 'incidence_angle', 'Incidence angle of given point', incidence_angle)
    add_dimension(las, 'angle_difference', 'Diff of scan and incidence angle', angle_difference)


def compute_incidence_angle(v1, v2):
    result = np.degrees([np.arccos(np.clip(np.dot(e1, e2), -1.0, 1.0)) for e1, e2 in zip(v1, v2)])
    # i = np.argmax(result)
    # visualize_vectors(v1[i],v2[i])
    print(result)
    return result

def compute_incidence_angle_new(laser_beams, normals):
    result = np.degrees(np.arccos(np.einsum('ij,ij->i', laser_beams, normals)/(np.linalg.norm(laser_beams, axis=1)*np.linalg.norm(normals, axis=1))))
    return result

def directional_vector_angles(rot_angles):
    rot_matrix = scipy.spatial.transform.Rotation.from_rotvec(rot_angles)
    rot_vectors = rot_matrix.apply(np.array([0, 0, -1]))
    return rot_vectors
    # return unit_vector(np.array(x, y, z))


def directional_vector_points(pointA, pointB):
    xyz_pointA = np.array([pointA.x, pointA.y, pointA.z]).transpose()
    xyz_pointB = np.array([pointB.x, pointB.y, pointB.z]).transpose()

    return unit_vector(xyz_pointA-xyz_pointB)


def unit_vector(vector):
    return (vector / np.linalg.norm(vector, axis=1)[:, None])


def compute_scan_angle(pointA, pointB, rot_angles):
    v1 = directional_vector_angles(rot_angles)
    v2 = directional_vector_points(pointA, pointB)
    result = np.degrees([np.arccos(np.clip(np.dot(e1, e2), -1.0, 1.0)) for e1, e2 in zip(v1, v2)])
    # i = np.argmax(result)
    # visualize_vectors(v1[i],v2[i])
    return result, v2

def compute_scan_angle_new(pointA, pointB, rot_angles):
    v1 = directional_vector_angles(rot_angles)
    v2 = directional_vector_points(pointA, pointB)
    result = np.degrees(np.arccos(np.einsum('ij,ij->i', v1, v2)/(np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1))))
    i = np.argmax(result)
    visualize_vectors(v1[i],v2[i])
    return result, v2

def visualize_vectors(v1, v2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(0, 0, 0, v1[0],v1[1],v1[2], color='red')
    ax.quiver(0, 0, 0, v2[0],v2[1],v2[2])
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(0,0,0)
    plt.show()
    

def pc_normal(path, knn):
    file = path2str(path, None)
    output = path2str(path, '_normals_{}.las'.format(knn))
    x="""
    {{
        "pipeline": [
            {{
                "filename": "{}",
                "spatialreference": "EPSG:32632"
            }},
            {{
                "type":"filters.normal",
                "knn":{}
            }},
            {{
                "filename": "{}",
                "extra_dims": "all"
            }}
        ]
    }}""".format(file, knn, output)

    print(x)

    pipeline = pdal.Pipeline(x)
    execute = pipeline.execute()
    print('success')
    return output

        
def thinning(path, output, method):
    orig_las = pylas.read(path)
    print(list(orig_las.point_format.dimension_names))
    # orig_las.gps_time.sort()
    print(orig_las.gps_time)
    

def las_to_histo(path, output_path):
    las = pylas.read(path)
    dim_list = list(las.point_format.dimension_names)
    split_path = os.path.split(path)
    print(dim_list[:4])
    for name in dim_list[:4]:
        if name in ['X', 'Y', 'Z']:
            dim = np.array(las[name])/100
        else:
            dim = np.array(las[name])
        output = os.path.join(output_path, 'histo', split_path[1][:-4], name + '.jpg')
        histogram(dim, name, output)


def raster_to_histo(path, output_path):
    raster = rio.open(path, mode="r+")
    print(type(raster.descriptions))
    # masked_band = np.ma.masked_array(raster.read(1), raster.nodata)
    bands = raster.read(masked=True)
    mask = bands[0].mask
    split_path = os.path.split(path)
    for band, desc in zip(bands, raster.descriptions):
        band.mask = mask
        list_band = []
        for row in band:
            for value in row:
                if type(value) != np.ma.core.MaskedConstant:
                    list_band.append(value)
        output = os.path.join(output_path, 'histo', split_path[1][:-4], desc + '.jpg')
        histogram(np.array(list_band), desc, output)

     
def histogram_old(band):
    summary_stats(np.array(band))


    n, bins, patches = plt.hist(x=band, bins='auto',
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('My Very Own Histogram')
    plt.text(1, 1, r'$\mu=15, b=3$', transform=ax.transAxes)
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 1000) * 1000 if maxfreq % 1000 else maxfreq + 1000)
    plt.show()


def histogram(band, desc, output):
    stats = summary_stats(np.array(band))
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(x=band, bins = 20)
    plt.title(desc)
    i=0.95
    for stat in stats.iteritems():
        text = '{}: {}'.format(stat[0], round(stat[1],3))
        ax.text(1.05,i, text, transform=ax.transAxes)
        i-=0.05
    plt.savefig(output,  bbox_inches='tight')
    plt.show()


def summary_stats(band):
    band_series = pd.Series(band)
    stats1 = band_series.describe()
    stats2 = scipy.stats.describe(band)
    stats1['median'] = np.median(band)
    stats1['mode'] = scipy.stats.mode(band).mode[0]
    stats1['mad'] = np.median(abs(band-np.median(band)))
    stats1['variance'] = stats2.variance
    stats1['range'] = np.ptp(band)
    stats1['rms'] = np.sqrt(np.mean(band**2))
    stats1['skewness'] = stats2.skewness
    stats1['kurtosis'] = stats2.kurtosis
    return stats1


def pc_raster(path):
    file = path2str(path, None)
    output = path2str(path, '_first_raster_01.tif')

    x="""
    {{
        "pipeline": [
            {{
                "filename": "{}",
                "spatialreference": "EPSG:32632"
            }},
            {{
                "filename": "{}",
                "resolution": 0.25,
                "where": "ReturnNumber == 1"
            }}
        ]
    }}""".format(file, output)

    print(x)

    pipeline = pdal.Pipeline(x)
    execute = pipeline.execute()
    print('success')


def pc_point_density(path, mode, radius):
    file = path2str(path, None)
    output_pc = path2str(path, '_pd.las')
    output_raster = path2str(path, '_pd.tif')

    x="""
    {{
        "pipeline": [
            {{
                "filename": "{}",
                "spatialreference": "EPSG:32632"
            }},
            {{
                "type": "filters.radialdensity",
                "radius": "{}"
            }},
            {{
                "type": "writers.las",
                "filename": {output_pc},
                "output_dims":
            }}
        ]
    }}""".format(file, output_pc)


def pc_clip(path, shapefile_path):
    shapefile_name = os.path.split(shapefile_path)[1].replace('.shp', '.laz')
    output_path = path.replace('.laz', '_' + shapefile_name)
    print(output_path)
    polygon = str(get_geometry_from_shp(shapefile_path))
    print(polygon)

    pipeline = pdal.Pipeline(json.dumps({
        "pipeline": [
            {
                "type": "readers.las",
                "filename": path
            }
            ,
            {
                "type": "filters.crop",
                "polygon": polygon
            },
            {
                "type": "writers.las",
                "compression": "laszip",
                "filename": output_path,
            }
        ]
    }))

    pipeline.execute()

    print("done")
    
    

def pc_merge(path, filename):
    output = os.path.join(path, filename)
    print(output)
    pipeline = pdal.Pipeline(json.dumps({
        "pipeline": [
            {
                "type": "readers.las",
                "filename": path + "/*"
            },
            {
                "type": "filters.merge"
            },
            {
                "type": "writers.las",
                "filename": output,
                
            }
        ]

    }))

    pipeline.execute()

    print(output)
    print("success")


def pc_visualize(path):
    point_cloud = pylas.read(path)
    print("point cloud read")
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
    colors = np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors/65535)
    print("let's visualize")

    o3d.visualization.draw_geometries([pcd])


def pc_trajectory(path):
    file_list = get_files(path)
    

    d_lines = {'start': [], 'end': [], 'strip': [], 'geometry': [] }

    for file in file_list:
        split_path = os.path.split(file)
        shp_point_path = os.path.join(split_path[0], 'points', split_path[1][:-4] + '_p.shp')
        print(shp_point_path)
        
        df = pd.read_csv(file)
        df['geometry'] = df.apply(lambda row: Point(row[['Easting[m]', 'Northing[m]', 'Height[m]']]), axis=1)
        gdf_points = gpd.GeoDataFrame(df, crs="EPSG:32632")

        gps_time = gdf_points['Time[s]']
        
        low_index = 0
        strip = 1

        for i in range(len(gps_time)-1):
            d_gps = gps_time[i+1]-gps_time[i]
            if d_gps > 2.:
                gdf_strip = gdf_points[low_index:i+1]
                print(d_gps)

                start = gdf_strip['Time[s]'].iat[0]
                end = gdf_strip['Time[s]'].iat[-1]
                line = LineString(gdf_strip['geometry'])
                d_lines['start'] += [start]
                d_lines['end'] += [end]
                d_lines['strip'] += [strip]
                d_lines['geometry'] += [line]

                strip += 1
                low_index = i + 1
        # las['point_source_id'][low_index:i+2] = strip


        # gdf_points.to_file(shp_point_path)

    shp_line_path = os.path.join(split_path[0], 'lines', 'trajectory.shp')

    geodf_lines = gpd.GeoDataFrame(d_lines, crs="EPSG:32632")
    geodf_lines.to_file(shp_line_path)


def create_boundary_shp(path):
    print(os.path.split(path)[0])
    shp_path = os.path.join(os.path.split(path)[0], 'boundary/boundary.shp')
    print(shp_path)
    
    file_list = get_files(path)

    filenames, geometries = create_lists(file_list)

    create_shapefile(shp_path, filenames, geometries)


def get_pipeline(file, type, output):
    file = path2str(file, None)
    output = path2str(output, None)
    x = """
    {{
        "pipeline": [
            "{}",
            {{
                "type": {}
            }},
            "{}"
        ]
    }}""".format(file, str(type), output)
    print(x)
    return pdal.Pipeline(x)


def boundary(metadata):
    coord = metadata['metadata']['filters.stats']['bbox']['native']['boundary']['coordinates'][0]
    return Polygon(coord)


def create_lists(file_list):

    filenames = []
    geometries = []

    for file in file_list:
        filename = os.path.split(file)[-1]
        filenames.append(filename)
        pipeline = pdal.Pipeline(get_pipeline(file))
        execution = pipeline.execute()
        metadata = pipeline.metadata
        geom = boundary(metadata)
        geometries.append(geom)

    print(filenames)
    print(geometries)

    return filenames, geometries


def get_files(path):
    return glob.glob(path)


def create_shapefile(shp_path, filenames, geometries):
    d = {'filename': filenames, 'geometry': geometries}
    gdf = gpd.GeoDataFrame(d, crs="EPSG:32632")
    gdf.to_file(shp_path)


def path2str(path, end):
    if end != None:
        return str(path).replace('\\', '/')[:-4] + '{}'.format(end)
    else:
        return str(path).replace('\\', '/')
    

def get_geometry_from_shp(path):
    data = gpd.read_file(path)
    return data['geometry'][0]


def test():
    print('start')
    # pipeline = pdal.Reader(filename=r'E:/NATUR_CUNI/_DP/data/LAZ/tile_652783_5188541.laz') | pdal.Filters.info()
    pipeline = pdal.Pipeline(x)
    print('success')
    count = pipeline.execute()
    print('megasuccess')
    arrays = pipeline.arrays
    metadata = pipeline.metadata
    log = pipeline.log
    print(metadata['metadata']['filters.stats']['bbox']['native']['boundary'])

if __name__ == '__main__':
    main()
