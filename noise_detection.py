#!/bin/python
import pylas
import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import eigh
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, roc_auc_score, recall_score, balanced_accuracy_score, jaccard_score
import time
import matplotlib.pyplot as plt
import seaborn as sns
import laspy as lp
import open3d as o3d
import json
import pdal
import pandas as pd
import os
from multiprocessing import Pool, cpu_count
import point_cloud_handeling as pc_h

def main():
    threads = cpu_count()-1
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    classify = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder4_opals.las'
    boulder = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder4_opals_mrr.las'
    latest = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder4_pdal_oknn2_noise.las'
    boulder2 = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder2_merge.las'
    boulder4 = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder4_new.las'
    boulder4_oKNN = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder4_new_oknn2.las'
    boulder4_smoothed = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder4_smoothed.las'
    boulder4_synthetic = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder4_smoothed_noise_0.1_0_0.1.las'
    boulder4_synthetic_r = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder4_smoothed_random_0.1_0.2.las'
    boulder4_synthetic2 = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder4_smoothed_noise_0.1_0_0.05.las'
    boulder4_synthetic_gr = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder4_smoothed_gauss_0.05_0.0_0.1-random_0.05_0.2.las'
    boulder4_synthetic_gr2 = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder4_smoothed_gauss_0.1_0.0_0.1-random_0.1_0.2.las'
    boulder4_synthetic_oKNN = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder4_smoothed_gauss_0.05_0.0_0.1-random_0.05_0.2_oknn2.las'

    boulder2_smoothed = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder2_smoothed.las'
    boulder2_synthetic_gr = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder2_smoothed_gauss_0.05_0.0_0.1-random_0.05_0.2.las'
    boulder2_synthetic_gr2 = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder2_smoothed_gauss_0.08_0.0_0.08-random_0.05_0.15.las'
    boulder2_synthetic_oKNN = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder2_smoothed_gauss_0.05_0.0_0.1-random_0.05_0.2_oknn2.las'
    boulder2_synthetic_oKNN2 = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder2_smoothed_gauss_0.08_0.0_0.08-random_0.05_0.15_oknn2.las'

    boulder4_fixed_normals = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder4_smoothed_N.las'
    boulder4_fixed_normals_noise = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder4_smoothed_N_gauss_0.05_0.0_0.05-random_0.05_0.2.las'
    boulder4_fixed_normals_noise2 = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder4_smoothed_N_gauss_0.1_0.0_0.05-random_0.1_0.2.las'
    boulder4_fixed_normals_noise3 = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder4_smoothed_N_gauss_0.05_0.0_0.1-random_0.05_0.3.las'
    boulder4_fixed_normals_noise4 = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder4_smoothed_N_gauss_0.05_0.0_0.03-random_0.05_0.08.las'


    boulder4_f_normals = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder4_smoothed_N.las'

    # add_gaussian_noise(boulder4_smoothed, amount=0.1, noise_std=0.05)
    # add_random_noise(boulder4_smoothed, amount=0.1, scale=0.2)
    # add_noise(boulder2_smoothed, amount_g=0.08, amount_r=0.05, g_std=0.08, r_scale=0.15)

    # add_noise_normal(boulder4_fixed_normals, amount_g = 0.05, amount_r=0.05, g_std=0.1, r_scale=0.2)
    add_noise_normal(boulder2_smoothed, amount_g = 0.05, amount_r=0.05, g_std=0.1, r_scale=0.2)
    # mrecor_train(boulder4_fixed_normals_noise4, max_k=50, step=1)
    # mrecsor_train(boulder4_fixed_normals_noise2, min_knn=5, max_knn=50, step=1, max_nsigma=3.0)
    # sor_train(boulder4_fixed_normals_noise4, min_k=2, max_k=25, step_k=4, step_nsigma=0.5)


    # sor_train(boulder4, min_k=2, max_k=25, step_k=4, step_nsigma=0.5)
    # sor_per_strip_train(boulder4)
    # msor_train(boulder4)
    # osor_train(boulder4_oKNN)
    #ror_train(boulder4)
    # recor_train(boulder4)
    # mrecor_train(boulder4)
    # orecor_train(boulder4_oKNN)
    # mrecsor_train(boulder4)
    # rsor_train(boulder4, min_k=3, max_k=55, min_nsigma=0.5, max_nsigma=5.0)

    # sor_train(boulder4_synthetic_gr2, min_k=2, max_k=25, step_k=4, step_nsigma=0.5)
    # sor_train(boulder4, min_k=2, max_k=25, step_k=4, step_nsigma=0.5)
    # sor_per_strip_train(boulder4)
    # msor_train(boulder4_synthetic_gr, step=1, max_k=50)
    # osor_train(boulder4_synthetic_oKNN)
    # ror_train(boulder4_synthetic_gr)
    # recor_train(boulder4_synthetic_gr)
    # mrecor_train(boulder4_synthetic_gr, max_k=50, step=1)
    # orecor_train(boulder4_synthetic_oKNN)
    # mrecsor_train(boulder4_synthetic_gr, min_knn=5, max_knn=50, step=1, max_nsigma=3.0)
    # mrecsor_detection(boulder4)
    # sor_train(boulder4_synthetic_gr, min_k=2, max_k=55, min_nsigma=0.5, max_nsigma=5.0, step_nsigma=0.5)
    # rsor_train(boulder4_synthetic_gr, min_k=2, max_k=55, min_nsigma=0.5, max_nsigma=5.0)

    # sor_train(boulder2_synthetic_gr, min_k=2, max_k=25, step_k=4, step_nsigma=0.5)
    # msor_train(boulder2_synthetic_gr, step=1, max_k=50)
    # osor_train(boulder2_synthetic_oKNN2)
    # mrecor_train(boulder4_synthetic_gr, max_k=50, step=1)
    # orecor_train(boulder2_synthetic_oKNN)
    # optimal_knn2(boulder4_synthetic_gr, min_k=4, max_k=50)
    # optimal_knn2(boulder2_synthetic_gr2, min_k=4, max_k=50)

    # mrecsor_train(boulder2_synthetic_gr, min_knn=5, max_knn=50, step=1, max_nsigma=3.0)
    # mrecsor_precise_train(boulder2_synthetic_gr)
    # mrecsor_detection(boulder4_fixed_normals)

def add_random_noise(las_path, amount=0.1, scale=0.1):
    las = pylas.read(las_path)

    total_points = len(las.points)
    num_points_to_move = int(amount*total_points)
    idx = np.random.choice(total_points, num_points_to_move, replace=False)

    las_new = las

    new_x = las.x
    new_y = las.y
    new_z = las.z

    new_x[idx] += np.random.uniform(-1., 1., num_points_to_move)*scale
    new_y[idx] += np.random.uniform(-1., 1., num_points_to_move)*scale
    new_z[idx] += np.random.uniform(-1., 1., num_points_to_move)*scale

    las_new.x = new_x
    las_new.y = new_y
    las_new.z = new_z

    las_new.classification[idx] = 7

    output_file = las_path.replace('.las', '_random_{}_{}.las'.format(amount,scale))
    las_new.write(output_file)


def add_gaussian_noise(las_path, amount=0.1, noise_mean=0, noise_std=0.1):
    las = pylas.read(las_path)
    
    total_points = len(las.points)
    num_points_to_move = int(amount*total_points)
    idx = np.random.choice(total_points, num_points_to_move, replace=False)

    las_new = las
    #print(las.header.scale)

    print(las.x[idx])
    print(las.X[idx])

    new_x = las.x
    new_y = las.y
    new_z = las.z

    new_x[idx] += np.random.normal(noise_mean, noise_std, num_points_to_move)
    new_y[idx] += np.random.normal(noise_mean, noise_std, num_points_to_move)
    new_z[idx] += np.random.normal(noise_mean, noise_std, num_points_to_move)
    
    
    las_new.x = new_x
    las_new.y = new_y
    las_new.z = new_z
    print(las_new.x[idx])
    las_new.classification[idx] = 7


    output_file = las_path.replace('.las', '_noise_{}_{}_{}.las'.format(amount,noise_mean,noise_std))
    las_new.write(output_file)

def add_noise(las_path, amount_g=0.05, amount_r=0.05, g_mean=0., g_std=0.05, r_scale=0.2):
    las = pylas.read(las_path)

    total_points = len(las.points)
    num_points_to_move = int((amount_g+amount_r)*total_points)
    print(num_points_to_move)
    idx = np.random.choice(total_points, num_points_to_move, replace=False)
    split_id = int(total_points*((amount_g+amount_r)*amount_g))
    g_idx = idx[:split_id]
    r_idx = idx[split_id:]

    las_new = las

    new_x = las.x
    new_y = las.y
    new_z = las.z

    new_x[g_idx] += np.random.normal(g_mean, g_std, len(g_idx))
    new_y[g_idx] += np.random.normal(g_mean, g_std, len(g_idx))
    new_z[g_idx] += np.random.normal(g_mean, g_std, len(g_idx))

    new_x[r_idx] += np.random.uniform(-1., 1., len(r_idx))*r_scale
    new_y[r_idx] += np.random.uniform(-1., 1., len(r_idx))*r_scale
    new_z[r_idx] += np.random.uniform(-1., 1., len(r_idx))*r_scale

    las_new.x = new_x
    las_new.y = new_y
    las_new.z = new_z
    las_new.classification[idx] = 7
    print(las_new.classification[idx])

    output_file = las_path.replace('.las', '_gauss_{}_{}_{}-random_{}_{}.las'.format(amount_g, g_mean, g_std, amount_r, r_scale))
    las_new.write(output_file)


def add_noise_normal(las_path, amount_g=0.05, amount_r=0.05, g_mean=0., g_std=0.05, r_scale=0.2):
    las = pylas.read(las_path)

    total_points = len(las.points)
    num_points_to_move = int((amount_g+amount_r)*total_points)
    print(num_points_to_move)
    idx = np.random.choice(total_points, num_points_to_move, replace=False)
    split_id = int(total_points*((amount_g+amount_r)*amount_g))
    g_idx = idx[:split_id]
    r_idx = idx[split_id:]

    las_new = las

    new_x = las.x
    new_y = las.y
    new_z = las.z
    Nx = las.Nx
    Ny = las.Ny
    Nz = las.Nz

    gaus_dev = np.random.normal(g_mean, g_std, len(g_idx))
    print(max(gaus_dev))
    # uni_dev = np.random.uniform(-1., 1., len(r_idx))*r_scale

    new_x[g_idx] += Nx[g_idx]*gaus_dev
    new_y[g_idx] += Ny[g_idx]*gaus_dev
    new_z[g_idx] += Nz[g_idx]*gaus_dev

    new_x[r_idx] += np.random.uniform(-1., 1., len(r_idx))*r_scale
    new_y[r_idx] += np.random.uniform(-1., 1., len(r_idx))*r_scale
    new_z[r_idx] += np.random.uniform(-1., 1., len(r_idx))*r_scale

    las_new.x = new_x
    las_new.y = new_y
    las_new.z = new_z
    # las_new.clasification = np.zeros(len(las_new.x), np.int8)
    las_new.classification[idx] = 7
    print(las_new.classification)

    output_file = las_path.replace('.las', '_gauss_{}_{}_{}-random_{}_{}.las'.format(amount_g, g_mean, g_std, amount_r, r_scale))
    las_new.write(output_file)

def add_noise_normal2(las_path, amount_g=0.05, amount_r=0.05, g_mean=0., g_std=0.05, r_scale=0.2):
    las = pylas.read(las_path)

    total_points = len(las.points)
    num_points_to_move = int((amount_g+amount_r)*total_points)
    print(num_points_to_move)
    idx = np.random.choice(total_points, num_points_to_move, replace=False)
    split_id = int(total_points*((amount_g+amount_r)*amount_g))
    g_idx = idx[:split_id]
    r_idx = idx[split_id:]

    las_new = las

    new_x = las.x
    new_y = las.y
    new_z = las.z
    Nx = las.Nx
    Ny = las.Ny
    Nz = las.Nz

    gaus_dev = np.random.normal(g_mean, g_std, len(g_idx))
    print(max(gaus_dev))
    uni_dev = np.random.uniform(-1., 1., len(r_idx))*r_scale

    new_x[g_idx] += Nx[g_idx]*gaus_dev
    new_y[g_idx] += Ny[g_idx]*gaus_dev
    new_z[g_idx] += Nz[g_idx]*gaus_dev

    new_x[r_idx] += Nx[r_idx]*uni_dev
    new_y[r_idx] += Ny[r_idx]*uni_dev
    new_z[r_idx] += Nz[r_idx]*uni_dev

    las_new.x = new_x
    las_new.y = new_y
    las_new.z = new_z
    # las_new.clasification = np.zeros(len(las_new.x), np.int8)
    las_new.classification[idx] = 7
    print(las_new.classification)

    output_file = las_path.replace('.las', '_gauss_{}_{}_{}-random_{}_{}_.las'.format(amount_g, g_mean, g_std, amount_r, r_scale))
    las_new.write(output_file)

"""
outlier detection
"""
def mrecsor_precise_train(las_path, min_knn=5, max_knn=50, min_nsigma=0.05, max_nsigma=0.4, min_r=0.7, max_r=0.9, step=1):
    start = time.time()
    
    las = pylas.read(las_path)  # Read LAS file using pylas
    threads = cpu_count()-1
    
    y_true = np.array(list(las.classification))
    y_true[y_true == 2] = 0
    y_true[y_true == 1] = 0
    y_true[y_true == 7] = 1

    xyz_points = pc_h.get_xyz(las)

    tree = cKDTree(xyz_points)  # Creating KDTree for efficient nearest neighbor search
    distances, indices = tree.query(xyz_points, k=max_knn + 1, workers = threads)  # Query k+1 nearest neighbors to exclude the point itself
    indices = indices[:,1:][indices[:,1:]]
    distances = distances[:,1:]

    
    knn_range = range(min_knn, max_knn+1, step)
    nsigma_range = np.arange(min_nsigma, max_nsigma, 0.05)
    min_r_range = np.arange(min_r,max_r,0.025)

    reciprocity_list = []
    knn_distances_list = []

    cont_table_f1 = np.ndarray((len(nsigma_range), len(min_r_range)), np.float64)
    cont_table_jaccard = cont_table_f1.copy()
    cont_table_recall = cont_table_f1.copy()
    cont_table_precision = cont_table_f1.copy()
    cont_table_outlier_percentage = cont_table_f1.copy()
    
    comparison = indices == np.arange(indices.shape[0])[:, None, None]
    
    for knn in knn_range:
        reciprocal_counts = np.count_nonzero(comparison[:, :knn, :knn], axis=(1,2))
        reciprocity = 1 - (reciprocal_counts / knn)
        reciprocity_list.append(reciprocity)
        knn_distances = np.median(distances[:,:knn], axis=1)
        knn_distances_list.append(knn_distances)

    max_reciprocity = np.max(np.array(reciprocity_list), axis=0)
    min_knn_distances = np.min(np.array(knn_distances_list), axis=0)
    max_avg_distance = np.mean(min_knn_distances)
    median_distance = np.median(min_knn_distances)
    max_std_distance = np.std(min_knn_distances)

    abs_diff = np.abs(knn_distances - median_distance)
    mad = np.median(abs_diff)
    nmad_distance = 1.4826 * mad

    dice_max = 0.0
    nsigma_top = 0.0
    min_r_top = 0.0
    for i, nsigma in enumerate(nsigma_range):
        # threshold = max_avg_distance + nsigma * max_std_distance
        threshold = median_distance + nsigma * nmad_distance

        for j, min_r in enumerate(min_r_range):
            #print(min_r)
            y_pred = np.zeros(len(y_true), np.int8)

            noise_mask = (max_reciprocity > min_r) & (knn_distances >= threshold)
            y_pred[noise_mask] = 1

            f1 = f1_score(y_true, y_pred, average='binary')
            jaccard = jaccard_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            outlier_percentage = np.sum(y_pred)/len(y_pred)
            if dice_max < f1:
                dice_max=f1
                nsigma_top=nsigma
                min_r_top=min_r
            cont_table_f1[i,j] = f1
            cont_table_jaccard[i,j] = jaccard
            cont_table_precision[i,j] = prec
            cont_table_recall[i,j] = rec
            cont_table_outlier_percentage[i,j] = outlier_percentage
    


    create_table(cont_table_f1, 'Multiscale RecSOR - Dice', np.round(min_r_range, decimals=2), np.round(nsigma_range, decimals=2), 'reciprocity threshold', 'nsigma')
    create_table(cont_table_jaccard, 'Multiscale RecSOR - Jaccard', np.round(min_r_range, decimals=2), np.round(nsigma_range, decimals=2), 'reciprocity threshold', 'nsigma')
    create_table(cont_table_recall, 'Multiscale RecSOR - recall', np.round(min_r_range, decimals=2),  np.round(nsigma_range, decimals=2), 'reciprocity threshold', 'nsigma')
    create_table(cont_table_precision, 'Multiscale RecSOR - precision', np.round(min_r_range, decimals=2),  np.round(nsigma_range, decimals=2), 'reciprocity threshold', 'nsigma')
    create_table(cont_table_outlier_percentage, 'Multiscale RecSOR - outlier percentage', np.round(min_r_range, decimals=2),  np.round(nsigma_range, decimals=2), 'reciprocity threshold', 'nsigma')

    print(nsigma_top, min_r_top, dice_max)
    end = time.time()
    runtime = end-start
    print('============ TOTAL RUNTIME: {} s ============'.format(runtime))

def mrecsor_detection(las_path, min_knn=5, max_knn=50, nsigma=0.3, min_r=0.7, step=1):
    start = time.time()
    
    las = pylas.read(las_path)  # Read LAS file using pylas
    threads = cpu_count()-1
    
    y_true = np.array(list(las.classification))
    y_true[y_true == 2] = 0
    y_true[y_true == 1] = 0
    y_true[y_true == 7] = 1

    xyz_points = pc_h.get_xyz(las)

    tree = cKDTree(xyz_points)  # Creating KDTree for efficient nearest neighbor search
    distances, indices = tree.query(xyz_points, k=max_knn + 1, workers = threads)  # Query k+1 nearest neighbors to exclude the point itself
    indices = indices[:,1:][indices[:,1:]]
    distances = distances[:,1:]

    
    knn_range = range(min_knn, max_knn+1, step)

    reciprocity_list = []
    knn_distances_list = []
    
    comparison = indices == np.arange(indices.shape[0])[:, None, None]
    
    for knn in knn_range:
        reciprocal_counts = np.count_nonzero(comparison[:, :knn, :knn], axis=(1,2))
        reciprocity = 1 - (reciprocal_counts / knn)
        reciprocity_list.append(reciprocity)
        knn_distances = np.median(distances[:,:knn], axis=1)
        knn_distances_list.append(knn_distances)

    max_reciprocity = np.max(np.array(reciprocity_list), axis=0)
    min_knn_distances = np.min(np.array(knn_distances_list), axis=0)
    max_avg_distance = np.mean(min_knn_distances)
    median_distance = np.median(min_knn_distances)
    max_std_distance = np.std(min_knn_distances)

    abs_diff = np.abs(knn_distances - median_distance)
    mad = np.median(abs_diff)
    nmad_distance = 1.4826 * mad

    threshold = median_distance + nsigma * nmad_distance
    y_pred = np.zeros(len(y_true), np.int8)

    noise_mask = (max_reciprocity > min_r) & (knn_distances >= threshold)
    y_pred[noise_mask] = 1  

    crossvalid = (y_true-y_pred).astype(np.int8)
    print(type(y_pred[0]))
    print(crossvalid)

    pc_h.add_dimension(las=las, dim_name='crossvalid', dim_type='int8', dim_description='crossvalid of outlier detection', dim_values=crossvalid)
    pc_h.add_dimension(las=las, dim_name='outliers', dim_type='int8', dim_description='outliers classified by mrecsor', dim_values=y_pred)

    output_file = las_path.replace('.las', 'mrecsor_{}_{}.las'.format(nsigma, min_r))
    las.write(output_file)

    end = time.time()
    runtime = end-start
    print('============ TOTAL RUNTIME: {} s ============'.format(runtime))

"""
training of outlier detection algorithms
"""

def sor_train(las_path, min_k=5, max_k=105, min_nsigma=0.1, max_nsigma=3.0, step_k=10, step_nsigma=0.2):
    start = time.time()

    las = pylas.read(las_path)  # Read LAS file using pylas
    
    y_true = np.array(list(las.classification))
    y_true[y_true == 2] = 0
    y_true[y_true == 1] = 0
    y_true[y_true == 7] = 1

    xyz_points = pc_h.get_xyz(las)

    tree = cKDTree(xyz_points)

    knn_range = np.arange(min_k, max_k+1, step_k)
    nsigma_range = np.arange(min_nsigma, max_nsigma+0.1, step_nsigma)

    distances = tree.query(xyz_points,k=max_k+1)[0][:,1:]
    print('distances computed')

    cont_table_f1 = np.ndarray((len(knn_range), len(nsigma_range)), np.float64)
    cont_table_recall = cont_table_f1.copy()
    cont_table_precision = cont_table_f1.copy()
    cont_table_outlier_percentage = cont_table_f1.copy()
    cont_table_jaccard = cont_table_f1.copy()

    x = 'nsigma'
    y = 'knn'
    cont_df = pd.DataFrame(columns=['nsigma', 'knn', 'metric', 'values'])

    for i, knn in enumerate(knn_range):
        print(knn)
        knn_distances = np.mean(distances[:,:knn], axis=1)
        avg_distance = np.mean(knn_distances)
        std_distance = np.std(knn_distances)

        for j, nsigma in enumerate(nsigma_range):
            y_pred = np.zeros(len(y_true), np.int8)

            max_distance = avg_distance + nsigma * std_distance

            y_pred[knn_distances >= max_distance] = 1

            f1 = f1_score(y_true, y_pred, average='binary')
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            outlier_percentage = np.sum(y_pred)/len(y_pred)
            jaccard = jaccard_score(y_true, y_pred)

            cont_table_f1[i,j] = f1
            cont_table_precision[i,j] = prec
            cont_table_recall[i,j] = rec
            cont_table_outlier_percentage[i,j] = outlier_percentage
            cont_table_jaccard[i,j] = jaccard

            cont_df.loc[len(cont_df)] = {x: nsigma, y: knn, 'metric': 'dice', 'values': f1}
            # cont_df.loc[len(cont_df)] = {'x': nsigma, 'y': knn, 'metric': 'jaccard', 'values': jaccard}
            cont_df.loc[len(cont_df)] = {x: nsigma, y: knn, 'metric': 'precision', 'values': prec}
            cont_df.loc[len(cont_df)] = {x: nsigma, y: knn, 'metric': 'recall', 'values': rec}
            cont_df.loc[len(cont_df)] = {x: nsigma, y: knn, 'metric': 'outlier percentage', 'values': outlier_percentage}

    # print(cont_df)


    # create_table(cont_table_f1, 'SOR - Dice', np.round(nsigma_range, decimals=1), knn_range.astype(int), 'nsigma', 'knn')
    # create_table(cont_table_jaccard, 'SOR - Jaccard similarity', np.round(nsigma_range, decimals=1), knn_range.astype(int), 'nsigma', 'knn')
    # create_table(cont_table_recall, 'SOR - recall', np.round(nsigma_range, decimals=1), knn_range.astype(int), 'nsigma', 'knn')
    # create_table(cont_table_precision, 'SOR - precision', np.round(nsigma_range, decimals=1), knn_range.astype(int), 'nsigma', 'knn')
    # create_table(cont_table_outlier_percentage, 'SOR - outlier percentage', np.round(nsigma_range, decimals=1), knn_range.astype(int), 'nsigma', 'knn')
    create_tables(cont_df, x, y, 'SOR')
    end = time.time()
    runtime = end-start
    print('============ TOTAL RUNTIME: {} s ============'.format(runtime))


def sor_per_strip_train(las_path, noise_dec=False, min_k=5, max_k=105, min_nsigma=0.1, max_nsigma=3.0):
    start = time.time()

    # load point cloud
    las = pylas.read(las_path)

    y_true = las.classification
    y_true[y_true == 2] = 0
    y_true[y_true == 1] = 0
    y_true[y_true == 7] = 1

    las.add_extra_dim('noise', type='int8')
    las.add_extra_dim('distance', type='f8')
    las.noise = np.zeros(len(las.points))

    # iterate for different combinations of parameters
    knn_range = np.arange(min_k, max_k+1, 10)
    nsigma_range = np.arange(min_nsigma, max_nsigma+0.1, 0.2)

    cont_table_f1 = np.ndarray((len(knn_range), len(nsigma_range)), np.float64)
    cont_table_recall = cont_table_f1.copy()
    cont_table_precision = cont_table_f1.copy()
    cont_table_outlier_percentage = cont_table_f1.copy()
    cont_table_jaccard = cont_table_f1.copy()

    strip_distances = {}

    for strip in np.unique(las.point_source_id):
        if noise_dec is False:
            pc_strip = np.array(list(zip(*[las.x[las.point_source_id == strip], 
                                las.y[las.point_source_id == strip],
                                las.z[las.point_source_id == strip]])))
            
            pc_rest = np.array(list(zip(*[las.x[las.point_source_id != strip], 
                                las.y[las.point_source_id != strip], 
                                las.z[las.point_source_id != strip]])))
        else:
            pc_strip = np.array(list(zip(*[las.x[(las.point_source_id == strip) & (las.noise != 1)], 
                                las.y[(las.point_source_id == strip) & (las.noise != 1)],
                                las.z[(las.point_source_id == strip) & (las.noise != 1)]])))
            
            pc_rest = np.array(list(zip(*[las.x[(las.point_source_id != strip) & (las.noise != 1)], 
                                las.y[(las.point_source_id != strip) & (las.noise != 1)], 
                                las.z[(las.point_source_id != strip) & (las.noise != 1)]])))
        
        kdtree = cKDTree(pc_rest)
        # kdtree_strip = cKDTree(pc_strip)

        strip_distances[strip] = kdtree.query(pc_strip, max_k+1)[0][:,1:]

    for i, knn in enumerate(knn_range):
        for j, nsigma in enumerate(nsigma_range):
            for strip in np.unique(las.point_source_id):
                strip_d = strip_distances[strip][:,:knn]

                if knn > 1:
                    distances = np.mean(strip_d, axis=1)

                if noise_dec is False:
                    las.distance[las.point_source_id == strip] = distances
                else:
                    las.distance[(las.point_source_id == strip) & (las.noise != 1)] = distances
            avg_distance = np.mean(las.distance)
            std_distance = np.std(las.distance)
            max_distance = avg_distance + nsigma * std_distance
            las.noise[las.distance >= max_distance] = 1

            y_pred = las.noise
            print(y_pred)
            f1 = f1_score(y_true, y_pred, average='binary')
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            outlier_percentage = np.sum(y_pred)/len(y_pred)
            jaccard = jaccard_score(y_true, y_pred)

            cont_table_f1[i,j] = f1
            cont_table_precision[i,j] = prec
            cont_table_recall[i,j] = rec
            cont_table_outlier_percentage[i,j] = outlier_percentage
            cont_table_jaccard[i,j] = jaccard

            las.noise = np.zeros(len(las.points))
            las.distances = np.zeros(len(las.points))
            
    create_table(cont_table_f1, 'SOR-per-strip - f1', np.round(nsigma_range, decimals=1), knn_range.astype(int), 'nsigma', 'knn')
    create_table(cont_table_jaccard, 'SOR - jaccard similarity', np.round(nsigma_range, decimals=1), knn_range.astype(int), 'nsigma', 'knn')
    create_table(cont_table_recall, 'SOR-per-strip - recall', np.round(nsigma_range, decimals=1), knn_range.astype(int), 'nsigma', 'knn')
    create_table(cont_table_precision, 'SOR-per-strip - precision', np.round(nsigma_range, decimals=1), knn_range.astype(int), 'nsigma', 'knn')
    create_table(cont_table_outlier_percentage, 'SOR - outlier percentage', np.round(nsigma_range, decimals=1), knn_range.astype(int), 'nsigma', 'knn')

    end = time.time()
    runtime = end-start

    print('============ TOTAL RUNTIME: {} s ============'.format(runtime))


def msor_train(las_path, min_k=5, max_k=50, min_nsigma=0.1, max_nsigma=2.5, step=1):
    start = time.time()

    las = pylas.read(las_path)  # Read LAS file using pylas
    
    y_true = np.array(list(las.classification))
    y_true[y_true == 2] = 0
    y_true[y_true == 1] = 0
    y_true[y_true == 7] = 1

    xyz_points = pc_h.get_xyz(las)

    knn_range = np.arange(min_k, max_k+1, 5)
    nsigma_range = np.arange(min_nsigma, max_nsigma+0.1, 0.1)

    tree = cKDTree(xyz_points)  # Creating KDTree for efficient nearest neighbor search
    distances = tree.query(xyz_points, k=max_k + 1)[0][:,1:]  # Query k+1 nearest neighbors to exclude the point itself
    
    f1_list = []
    jaccard_list = []
    rec_list = []
    prec_list = []
    outp_list = []

    for sd in nsigma_range:
        y_pred = np.zeros(len(y_true), np.int8)
        for knn in knn_range:
            knn_distances = np.mean(distances[:,:knn], axis=1)
            avg_distance = np.mean(knn_distances)
            std_distance = np.std(knn_distances)
        
            max_distance = avg_distance + sd * std_distance

            y_pred[knn_distances >= max_distance] = 1

        f1_list.append(f1_score(y_true, y_pred, average='binary'))
        jaccard_list.append(jaccard_score(y_true, y_pred))
        prec_list.append(precision_score(y_true, y_pred))
        rec_list.append(recall_score(y_true, y_pred))
        outp_list.append(round(np.sum(y_pred)/len(y_pred),2))

    df = pd.DataFrame(data={'nsigma': nsigma_range, 
                            'dice': f1_list,
                            'jaccard': jaccard_list,
                            'recall': rec_list,
                            'precision': prec_list,
                            'outlier_percentage': outp_list})
    
    create_lineplot(df, 'Multiscale SOR', 'nsigma')

    end = time.time()
    runtime = end-start

    print('============ TOTAL RUNTIME: {} s ============'.format(runtime))


def osor_train(las_path, min_nsigma=0.1, max_nsigma=3.0):
    start = time.time()
    # load file
    las = pylas.read(las_path)

    # add dimensions and prepare y_true
    las.noise = np.zeros(len(las.points))
    y_true = np.array(list(las.classification))
    y_true[y_true == 2] = 0
    y_true[y_true == 1] = 0
    y_true[y_true == 7] = 1

    # knn and sd ranges
    knn_range = las['oKNN']
    nsigma_range = np.arange(min_nsigma, max_nsigma, 0.1)

    xyz_points = pc_h.get_xyz(las)

    kdtree = cKDTree(xyz_points)
    pool_size = cpu_count()-1
    distances = np.array([np.mean(kdtree.query(xyz_points[i], knn, workers=pool_size)[0][1:]) for i, knn in enumerate(knn_range)])
    las.distance = distances
    print(distances)

    avg_distance = np.mean(las.distance)
    std_distance = np.std(las.distance)

    f1_list = []
    jaccard_list = []
    rec_list = []
    prec_list = []
    outp_list = []

    for nsigma in nsigma_range:
        max_distance = avg_distance + nsigma * std_distance

        las.noise[las.distance >= max_distance] = 1

        y_pred = las.noise
    
        f1_list.append(f1_score(y_true, y_pred, average='binary'))
        jaccard_list.append(jaccard_score(y_true, y_pred))
        prec_list.append(precision_score(y_true, y_pred))
        rec_list.append(recall_score(y_true, y_pred))
        outp_list.append(round(np.sum(y_pred)/len(y_pred),2))

        las.noise = np.zeros(len(las.points))

    df = pd.DataFrame(data={'nsigma': nsigma_range, 
                            'dice': f1_list,
                            'jaccard': jaccard_list,
                            'recall': rec_list,
                            'precision': prec_list,
                            'outlier_percentage': outp_list})
    
    create_lineplot(df, 'OptimalKNN SOR', 'nsigma')

    end = time.time()
    runtime = end-start

    print('============ TOTAL RUNTIME: {} s ============'.format(runtime))

def rsor_train(las_path, min_k=5, max_k=105, min_nsigma=0.1, max_nsigma=3.0):
    start = time.time()

    las = pylas.read(las_path)  # Read LAS file using pylas
    
    y_true = np.array(list(las.classification))
    y_true[y_true == 2] = 0
    y_true[y_true == 1] = 0
    y_true[y_true == 7] = 1

    xyz_points = pc_h.get_xyz(las)

    tree = cKDTree(xyz_points)

    knn_range = np.arange(min_k, max_k+1, 5)
    nsigma_range = np.arange(min_nsigma, max_nsigma+0.1, 0.5)

    distances = tree.query(xyz_points,k=max_k+1)[0][:,1:]
    print('distances computed')

    cont_table_f1 = np.ndarray((len(knn_range), len(nsigma_range)), np.float64)
    cont_table_jaccard = cont_table_f1.copy()
    cont_table_recall = cont_table_f1.copy()
    cont_table_precision = cont_table_f1.copy()
    cont_table_outlier_percentage = cont_table_f1.copy()

    for i, knn in enumerate(knn_range):
        print(knn)
        knn_distances = np.median(distances[:,:knn], axis=1)
        median_distance = np.median(knn_distances)

        abs_diff = np.abs(knn_distances - median_distance)
        mad = np.median(abs_diff)
        # nmad_distance = 1.4826 * mad

        for j, nsigma in enumerate(nsigma_range):
            y_pred = np.zeros(len(y_true), np.int8)

            max_distance = median_distance + nsigma * mad

            y_pred[knn_distances >= max_distance] = 1

            f1 = f1_score(y_true, y_pred, average='binary')
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            outlier_percentage = np.sum(y_pred)/len(y_pred)
            jaccard = jaccard_score(y_true, y_pred)

            cont_table_f1[i,j] = f1
            cont_table_jaccard[i,j] = jaccard
            cont_table_precision[i,j] = prec
            cont_table_recall[i,j] = rec
            cont_table_outlier_percentage[i,j] = outlier_percentage

    
    create_table(cont_table_f1, 'SOR - Dice', np.round(nsigma_range, decimals=1), knn_range.astype(int), 'nsigma', 'knn')
    create_table(cont_table_jaccard, 'SOR - Jaccard', np.round(nsigma_range, decimals=1), knn_range.astype(int), 'nsigma', 'knn')
    create_table(cont_table_recall, 'SOR - recall', np.round(nsigma_range, decimals=1), knn_range.astype(int), 'nsigma', 'knn')
    create_table(cont_table_precision, 'SOR - precision', np.round(nsigma_range, decimals=1), knn_range.astype(int), 'nsigma', 'knn')
    create_table(cont_table_outlier_percentage, 'SOR - outlier percentage', np.round(nsigma_range, decimals=1), knn_range.astype(int), 'nsigma', 'knn')
    end = time.time()
    runtime = end-start

    print('============ TOTAL RUNTIME: {} s ============'.format(runtime))

def ror_train(las_path, r_min=0.04, r_max=0.15, k_min=1, k_max=15):
    start = time.time()
    
    las = pylas.read(las_path)  # Read LAS file using pylas
    threads = cpu_count()-1
    
    y_true = np.array(list(las.classification))
    y_true[y_true == 2] = 0
    y_true[y_true == 1] = 0
    y_true[y_true == 7] = 1

    xyz_points = pc_h.get_xyz(las)

    r_range = np.arange(r_min, r_max+0.01, 0.01)
    k_range = np.arange(k_min, k_max+1, 1)

    cont_table_f1 = np.ndarray((len(r_range), len(k_range)), np.float64)
    cont_table_jaccard = cont_table_f1.copy()
    cont_table_recall = cont_table_f1.copy()
    cont_table_precision = cont_table_f1.copy()
    cont_table_outlier_percentage = cont_table_f1.copy()

    tree = cKDTree(xyz_points)  # Creating KDTree for efficient nearest neighbor search
    
    for i, r in enumerate(r_range):
        indices = tree.query(xyz_points, k=k_max + 1, workers = threads, distance_upper_bound=r)[1][:,1:]
        # print(indices)
        #print(indices)
        for j, k in enumerate(k_range):  
            y_pred = np.zeros(len(y_true), np.int8)
            idx = indices[:,:k]
            # print(idx)
            #print(idx)
            #print(np.max(idx, axis=1))
            mask = np.max(idx, axis=1) == len(y_pred)
            #print(mask)
            y_pred[mask] = 1

            f1 = f1_score(y_true, y_pred, average='binary')
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            outlier_percentage = round((np.sum(y_pred)/len(y_pred))*100,2)
            jaccard = jaccard_score(y_true, y_pred)

            cont_table_f1[i,j] = f1
            cont_table_jaccard[i,j] = jaccard
            cont_table_precision[i,j] = prec
            cont_table_recall[i,j] = rec
            cont_table_outlier_percentage[i,j] = outlier_percentage

    create_table(cont_table_f1, 'ROR - Dice', k_range.astype(int), np.round(r_range, decimals=2), 'knn', 'radius')
    create_table(cont_table_jaccard, 'ROR - Jaccard', k_range.astype(int), np.round(r_range, decimals=2), 'knn', 'radius')
    create_table(cont_table_recall, 'ROR - recall', k_range.astype(int), np.round(r_range, decimals=2), 'knn', 'radius')
    create_table(cont_table_precision, 'ROR - precision', k_range.astype(int), np.round(r_range, decimals=2), 'knn', 'radius')
    create_table(cont_table_outlier_percentage, 'ROR - outlier percentage', k_range.astype(int), np.round(r_range, decimals=2), 'knn', 'radius')
    
    end = time.time()
    runtime = end-start
    print('============ TOTAL RUNTIME: {} s ============'.format(runtime))

def recor_train(las_path, min_k=5, max_k=55, rec_min=0.1, rec_max=0.8):
    start = time.time()

    las = pylas.read(las_path)
    # save classification 
    y_true = np.array(list(las.classification))
    y_true[y_true == 2] = 0
    y_true[y_true == 1] = 0
    y_true[y_true == 7] = 1

    # read point cloud
    xyz_points = pc_h.get_xyz(las)

    knn_range = np.arange(min_k, max_k+1, 5)
    minr_range = np.arange(rec_min, rec_max+0.01, 0.05)

    cont_table_f1 = np.ndarray((len(knn_range), len(minr_range)), np.float64)
    cont_table_jaccard = cont_table_f1.copy()
    cont_table_recall = cont_table_f1.copy()
    cont_table_precision = cont_table_f1.copy()
    cont_table_outlier_percentage = cont_table_f1.copy()

    tree = cKDTree(xyz_points)
    indices = tree.query(xyz_points, k=max_k+1)[1][:,1:]

    # print(cont_table)
    for i, knn in enumerate(knn_range):  
        idx = indices[:,:knn][indices[:,:knn]]

        reciprocal_counts = np.count_nonzero(idx == np.arange(idx.shape[0])[:, None, None], axis=(1, 2)) 

        reciprocity = 1-(reciprocal_counts / knn)
        
        for j, minr in enumerate(minr_range):
            y_pred = np.zeros(len(y_true), np.int8)
            noise_mask = reciprocity > minr
            y_pred[noise_mask] = 1

            f1 = f1_score(y_true, y_pred, average='binary')
            jaccard = jaccard_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            outlier_percentage = np.sum(y_pred)/len(y_pred)

            cont_table_f1[i,j] = f1
            cont_table_jaccard[i,j] = jaccard
            cont_table_precision[i,j] = prec
            cont_table_recall[i,j] = rec
            cont_table_outlier_percentage[i,j] = outlier_percentage

    create_table(cont_table_f1, 'Reciprocity Outlier Removal - Dice', np.round(minr_range, decimals=2),  knn_range.astype(int), 'reciprocity threshold', 'knn')
    create_table(cont_table_jaccard, 'Reciprocity Outlier Removal - Jaccard', np.round(minr_range, decimals=2),  knn_range.astype(int), 'reciprocity threshold', 'knn')
    create_table(cont_table_recall, 'Reciprocity Outlier Removal - recall', np.round(minr_range, decimals=2),  knn_range.astype(int), 'reciprocity threshold', 'knn')
    create_table(cont_table_precision, 'Reciprocity Outlier Removal - precision',np.round(minr_range, decimals=2),  knn_range.astype(int), 'reciprocity threshold', 'knn')
    create_table(cont_table_outlier_percentage, 'Reciprocity Outlier Removal - outlier percentage', np.round(minr_range, decimals=2),  knn_range.astype(int), 'reciprocity threshold', 'knn')
    
    end = time.time()
    runtime = end-start

    print('============ TOTAL RUNTIME: {} s ============'.format(runtime))

def mrecor_train(las_path, min_k=5, max_k=50, rec_min=0.3, rec_max=0.95, step=1):
    start = time.time()

    las = pylas.read(las_path)  # Read LAS file using pylas
    threads = cpu_count()-1
    5
    y_true = np.array(list(las.classification))
    y_true[y_true == 2] = 0
    y_true[y_true == 1] = 0
    y_true[y_true == 7] = 1

    xyz_points = pc_h.get_xyz(las)

    knn_range = np.arange(min_k, max_k+1, step)
    minr_range = np.arange(rec_min, rec_max+0.01, 0.05)

    tree = cKDTree(xyz_points)  # Creating KDTree for efficient nearest neighbor search
    indices = tree.query(xyz_points, k=max_k + 1, workers = threads)[1][:,1:]  # Query k+1 nearest neighbors to exclude the point itself
    indices = indices[:,:][indices[:,:]]

    reciprocity_list = []
    f1_list = []
    jaccard_list = []
    rec_list = []
    prec_list = []
    outp_list = []

    # print(np.arange(idx.shape[0])[:, None, None])
    comparison = indices == np.arange(indices.shape[0])[:, None, None]
    
    for knn in knn_range:
        reciprocal_counts = np.count_nonzero(comparison[:, :knn, :knn], axis=(1,2))    
        reciprocity = 1 - (reciprocal_counts / knn)
        reciprocity_list.append(reciprocity)

    print(np.array(reciprocity_list))

    max_reciprocity = np.max(np.array(reciprocity_list), axis=0)
    print(max_reciprocity)

    for min_r in minr_range:
        #print(min_r)
        y_pred = np.zeros(len(y_true), np.int8)
        noise_mask = max_reciprocity > min_r
        #print(noise_mask)
        y_pred[noise_mask] = 1

        f1_list.append(f1_score(y_true, y_pred, average='binary'))
        jaccard_list.append(jaccard_score(y_true, y_pred))
        prec_list.append(precision_score(y_true, y_pred))
        rec_list.append(recall_score(y_true, y_pred))
        outp_list.append(round(np.sum(y_pred)/len(y_pred),2))

    df = pd.DataFrame(data={'reciprocity threshold': minr_range, 
                            'dice': f1_list,
                            'jaccard': jaccard_list,
                            'recall': rec_list,
                            'precision': prec_list,
                            'outlier_percentage': outp_list})
    
    create_lineplot(df, 'Multiscale RecOR', 'reciprocity threshold')

    end = time.time()
    runtime = end-start

    print('============ TOTAL RUNTIME: {} s ============'.format(runtime))


def orecor_train(las_path, min_k=10, max_k=100, min_r=0.1, max_r=0.8):
    start = time.time()
    pool_size = cpu_count()-1

    las = pylas.read(las_path)
    # save classification 
    y_true = np.array(list(las.classification))
    y_true[y_true == 2] = 0
    y_true[y_true == 1] = 0
    y_true[y_true == 7] = 1
    
    # read point cloud
    xyz_points = pc_h.get_xyz(las)

    knn_range = np.arange(min_k, max_k+1, 1)
    minr_range = np.arange(min_r,max_r+0.01, 0.05)

    kdtree = cKDTree(xyz_points)
    knn_indices = kdtree.query(xyz_points, max_k+1, workers=pool_size)[1][:,1:]

    try:
        optimal_knn = las.oKNN
        print('it works')
    except ValueError:
        arguments = [xyz_points[knn_indices[:, :knn]] for knn in knn_range]

        with Pool(pool_size) as pool:
            results = np.array(pool.map(process_knn, arguments)).T
            eigen_knn = pd.DataFrame(results)

        eigen_knn.columns = knn_range

        optimal_knn = eigen_knn.idxmin(axis=1)
    print('yes, indeed')

    # Create a mask for filtering optimal neighbors
    mask = np.arange(knn_indices.shape[1]) < optimal_knn[:, np.newaxis]
    mask2 = np.broadcast_to(mask[: , :, np.newaxis], (mask.shape[0], mask.shape[1], mask.shape[1]))

    masked_indices = np.ma.MaskedArray(knn_indices, ~mask)
    
    # print(masked_indices)

    # reciprocity = np.equal(masked_indices[:, :, np.newaxis], masked_indices[:, np.newaxis, :])
    # Count reciprocal neighbors
    # reciprocal_counts = np.sum(masked_indices[:, :, np.newaxis] == masked_indices[:, np.newaxis, :], axis=2)
    idx = masked_indices[masked_indices]
    masked_idx = np.ma.MaskedArray(idx, ~mask2)

    reciprocal_counts = np.count_nonzero(masked_idx == np.arange(masked_idx.shape[0])[:, None, None], axis=(1, 2)) 

    reciprocity = 1 - (reciprocal_counts / np.array(optimal_knn))

    f1_list = []
    jaccard_list = []
    rec_list = []
    prec_list = []
    outp_list = []

    for minr in minr_range:
        y_pred = np.zeros(len(y_true), np.int8)
        noise_mask = reciprocity > minr
        y_pred[noise_mask] = 1

        f1_list.append(f1_score(y_true, y_pred, average='binary'))
        jaccard_list.append(jaccard_score(y_true, y_pred))
        prec_list.append(precision_score(y_true, y_pred))
        rec_list.append(recall_score(y_true, y_pred))
        outp_list.append(round(np.sum(y_pred)/len(y_pred),2))

    df = pd.DataFrame(data={'reciprocity threshold': minr_range, 
                            'dice': f1_list,
                            'jaccard': jaccard_list,
                            'recall': rec_list,
                            'precision': prec_list,
                            'outlier_percentage': outp_list})

    create_lineplot(df, 'OptimalKNN RecOR', 'reciprocity threshold')

    end = time.time()
    runtime = end-start

    print('============ TOTAL RUNTIME: {} s ============'.format(runtime))

def mrecsor_train(las_path, min_knn=5, max_knn=100, min_nsigma=0.1, max_nsigma=4.4, step=10):
    start = time.time()
    
    las = pylas.read(las_path)  # Read LAS file using pylas
    threads = cpu_count()-1
    
    y_true = np.array(list(las.classification))
    y_true[y_true == 2] = 0
    y_true[y_true == 1] = 0
    y_true[y_true == 7] = 1

    xyz_points = pc_h.get_xyz(las)

    tree = cKDTree(xyz_points)  # Creating KDTree for efficient nearest neighbor search
    distances, indices = tree.query(xyz_points, k=max_knn + 1, workers = threads)  # Query k+1 nearest neighbors to exclude the point itself
    indices = indices[:,1:][indices[:,1:]]
    distances = distances[:,1:]

    
    knn_range = range(min_knn, max_knn+1, step)
    nsigma_range = np.arange(min_nsigma, max_nsigma, 0.3)
    min_r_range = np.arange(0.1,1.0,0.1)

    reciprocity_list = []
    knn_distances_list = []

    cont_table_f1 = np.ndarray((len(nsigma_range), len(min_r_range)), np.float64)
    cont_table_jaccard = cont_table_f1.copy()
    cont_table_recall = cont_table_f1.copy()
    cont_table_precision = cont_table_f1.copy()
    cont_table_outlier_percentage = cont_table_f1.copy()
    
    comparison = indices == np.arange(indices.shape[0])[:, None, None]
    
    for knn in knn_range:
        reciprocal_counts = np.count_nonzero(comparison[:, :knn, :knn], axis=(1,2))
        reciprocity = 1 - (reciprocal_counts / knn)
        reciprocity_list.append(reciprocity)
        knn_distances = np.median(distances[:,:knn], axis=1)
        knn_distances_list.append(knn_distances)

    max_reciprocity = np.max(np.array(reciprocity_list), axis=0)
    min_knn_distances = np.min(np.array(knn_distances_list), axis=0)
    max_avg_distance = np.mean(min_knn_distances)
    median_distance = np.median(min_knn_distances)
    max_std_distance = np.std(min_knn_distances)

    abs_diff = np.abs(knn_distances - median_distance)
    mad = np.median(abs_diff)
    nmad_distance = 1.4826 * mad

    for i, nsigma in enumerate(nsigma_range):
        # threshold = max_avg_distance + nsigma * max_std_distance
        threshold = median_distance + nsigma * nmad_distance

        for j, min_r in enumerate(min_r_range):
            #print(min_r)
            y_pred = np.zeros(len(y_true), np.int8)

            noise_mask = (max_reciprocity > min_r) & (knn_distances >= threshold)
            y_pred[noise_mask] = 1

            f1 = f1_score(y_true, y_pred, average='binary')
            jaccard = jaccard_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            outlier_percentage = np.sum(y_pred)/len(y_pred)

            cont_table_f1[i,j] = f1
            cont_table_jaccard[i,j] = jaccard
            cont_table_precision[i,j] = prec
            cont_table_recall[i,j] = rec
            cont_table_outlier_percentage[i,j] = outlier_percentage
    
    create_table(cont_table_f1, 'Multiscale RecSOR - Dice', np.round(min_r_range, decimals=2), np.round(nsigma_range, decimals=2), 'reciprocity threshold', 'nsigma')
    create_table(cont_table_jaccard, 'Multiscale RecSOR - Jaccard', np.round(min_r_range, decimals=2), np.round(nsigma_range, decimals=2), 'reciprocity threshold', 'nsigma')
    create_table(cont_table_recall, 'Multiscale RecSOR - recall', np.round(min_r_range, decimals=2),  np.round(nsigma_range, decimals=2), 'reciprocity threshold', 'nsigma')
    create_table(cont_table_precision, 'Multiscale RecSOR - precision', np.round(min_r_range, decimals=2),  np.round(nsigma_range, decimals=2), 'reciprocity threshold', 'nsigma')
    create_table(cont_table_outlier_percentage, 'Multiscale RecSOR - outlier percentage', np.round(min_r_range, decimals=2),  np.round(nsigma_range, decimals=2), 'reciprocity threshold', 'nsigma')

    end = time.time()
    runtime = end-start
    print('============ TOTAL RUNTIME: {} s ============'.format(runtime))


def nf_train(las_path, r_min=0.04, r_max=0.2, k_min=4, k_max=25, m_min=0.4, m_max=3, relative=True):
    start = time.time()
    print('hi')
    las = pylas.read(las_path)  # Read LAS file using pylas
    threads = cpu_count()-1
    
    y_true = np.array(list(las.classification))
    y_true[y_true == 2] = 0
    y_true[y_true == 1] = 0
    y_true[y_true == 7] = 1

    xyz_points = pc_h.get_xyz(las)

    r_range = np.arange(r_min, r_max+0.01, 0.02)
    k_range = np.arange(k_min, k_max+1, 2)
    m_range = np.arange(m_min, m_max+0.1, 0.2)

    cont_table_r = np.ndarray((len(r_range), len(m_range)), np.float64)
    cont_table_k = np.ndarray((len(k_range), len(m_range)), np.float64)

    tree = cKDTree(xyz_points)  # Creating KDTree for efficient nearest neighbor search

    for i, r in enumerate(r_range):
        indices = tree.query_ball_point(xyz_points, workers = threads, r=r)
        points = np.array([xyz_points[idx[1:]] for idx in indices])
        less_than_3 = np.array([point.shape[0] <= 3 for point in points])
        # print(points)
        # print(less_than_3)
        neighborhoods = points[~less_than_3]
        plane_parameters = [fit_plane(neighborhood) for neighborhood in neighborhoods]
        print(plane_parameters)
        
"""
Utils functions
"""

def compute_eigen(neighborhoods):
    # Compute the covariance matrices for all neighborhoods
    covariance_matrices = np.array([np.cov(neighborhood.T) for neighborhood in neighborhoods])

    # Compute the eigenvalues for all covariance matrices
    _, eigenvalues, _ = np.linalg.svd(covariance_matrices, full_matrices=False)

    # Compute the eigenentropies
    eigenentropies = [entropy(eigvals) for eigvals in eigenvalues]
    return eigenentropies, eigenvalues

def get_knn(df, ev_knn):
    minima_index = df.idxmin(axis='columns')
    print(minima_index)
    #optimal_nearest_neighbors = df.columns[minima_index]  # Retrieve the column name corresponding to the minimum eigenentropy
    #return optimal_nearest_neighbors

def write_classification(las_path1, las_path2):
    las1 = pylas.read(las_path1)
    las2 = pylas.read(las_path2)
    las1.classification = las2.classification
    las1.write(las_path1)

def evaluate_noise_detection(las_path):
    las = pylas.read(las_path)

    y_true = las.classification
    y_true[y_true == 2] = 0
    y_true[y_true == 1] = 0
    y_true[y_true == 7] = 1
    y_pred = las.Noise

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print('Accuracy: ', acc)
    print('Balanced accuracy: ', bal_acc)
    print('Prec: ', prec)
    print('Recall: ', rec)
    print('F1 score: ', f1)

def fit_plane(points):
    """
    Fit a plane to the input points using the least-squares method.

    Parameters:
        points (numpy.ndarray): Nx3 array representing the input 3D points.

    Returns:
        numpy.ndarray: 4-element array representing the plane parameters (a, b, c, d) in ax + by + cz + d = 0.
    """

    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    _, _, v = np.linalg.svd(points_centered)
    normal = v[2, :]
    d = -np.dot(normal, centroid)
    plane_params = np.append(normal, d)
    return plane_params

def test(las_path):
    las = pylas.read(las_path)  # Read LAS file using pylas

    xyz_points = pc_h.get_xyz(las)

    tree = cKDTree(xyz_points)  # Creating KDTree for efficient nearest neighbor search
    distances, indices = tree.query(xyz_points, k=6)  # Query k+1 nearest neighbors to exclude the point itself

    idx = indices[:,1:][indices[:,1:]]
    print(idx)

    indices = np.arange(idx.shape[0])

    counts = np.count_nonzero(idx == indices[:, None, None], axis=(1, 2)) / 5
    print(counts)

def create_table(cont_table, title, x_ticks, y_ticks, x_label, y_label):
    min_acc = np.amin(cont_table)
    max_acc = np.amax(cont_table)
    fig = plt.figure(num=None, figsize=(12, 10), dpi=250, facecolor='w', edgecolor='k')

    plt.clf()

    ax = fig.add_subplot(111)

    ax.set_aspect(1)

    res = sns.heatmap(cont_table, annot=True, fmt='.2f', cmap="YlGnBu", vmin=min_acc, vmax=max_acc)
    plt.xticks(np.arange(0.5, cont_table.shape[1] + 0.5), x_ticks)
    plt.yticks(np.arange(0.5, cont_table.shape[0] + 0.5), y_ticks)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title(title,fontsize=16)
    plt.show()

def create_tables(cont_df, x_label, y_label, title):
    g = sns.FacetGrid(cont_df, col='metric', height=6)
    g.map_dataframe(draw_heatmap, x_label, y_label, 'values', annot=True, fmt='.2f', cmap="YlGnBu")
    # g.set_ylabel=y_label
    # for ax in g.axes[0]:
    #     ax.set_xlabel=x_label
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(title, fontsize=20, fontweight=800)


def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, **kwargs)

def create_lineplot(df, title, x_label):
    df.set_index(x_label)

    plt.figure(num=None, figsize=(12, 10), dpi=250, facecolor='w', edgecolor='k')
    plt.grid(True)
    plt.plot(x_label, 'dice', data=df, label='dice')
    plt.plot(x_label, 'jaccard', data=df, label='jaccard')
    plt.plot(x_label, 'recall', data=df, label='recall')
    plt.plot(x_label, 'precision', data=df, label='precision')
    plt.plot(x_label, 'outlier_percentage', data=df, label='outlier percentage', c='m')
    annot_max(df, x_label)
    plt.xlabel(x_label)
    plt.legend()
    plt.title(title,fontsize=16)
    plt.xlim(0.2)
    plt.ylim(0)
    plt.show()

def annot_max(df, x_label):
    print(df)
    max_index = df['dice'].idxmax()
    row = df.iloc[max_index]
    print(row)
    text = '{}={:.2f}\n Dice={:.2f}\n Jaccard={:.2f}\n precision={:.2f}\n recall={:.2f}\n outlier percentage={:.2f}'.format(x_label, 
                                             row[x_label], row['dice'], row['jaccard'], row['precision'],
                                             row['recall'], row['outlier_percentage'])
    
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=120")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props,  ha="right", va="top")
    plt.annotate(text, xy=(row[x_label], row['dice']), xytext=(0.3,0.98), **kw)


def strip_pop(las, strip, noise_dec = False):
    if noise_dec is False:
        pc_strip = np.array(list(zip(*[las.x[las.point_source_id == strip], 
                            las.y[las.point_source_id == strip],
                            las.z[las.point_source_id == strip]])))
        
        pc_rest = np.array(list(zip(*[las.x[las.point_source_id != strip], 
                            las.y[las.point_source_id != strip], 
                            las.z[las.point_source_id != strip]])))
    else:
        pc_strip = np.array(list(zip(*[las.x[(las.point_source_id == strip) & (las.noise != 1)], 
                            las.y[(las.point_source_id == strip) & (las.noise != 1)],
                            las.z[(las.point_source_id == strip) & (las.noise != 1)]])))
        
        pc_rest = np.array(list(zip(*[las.x[(las.point_source_id != strip) & (las.noise != 1)], 
                            las.y[(las.point_source_id != strip) & (las.noise != 1)], 
                            las.z[(las.point_source_id != strip) & (las.noise != 1)]])))
        
    return pc_strip, pc_rest

def compute_eigen2(neighborhoods):
    # Compute the scatter matrices for all neighborhoods
    scatter_matrices = np.matmul(neighborhoods.transpose(0, 2, 1), neighborhoods)

    # Compute the eigenvalues for all scatter matrices
    eigenvalues = np.linalg.eigvalsh(scatter_matrices)

    # Compute the eigenentropies
    eigenentropies = entropy(eigenvalues)

    return eigenentropies, eigenvalues

def process_knn(args):
    points_knn = args
    eigenentropy, _ = compute_eigen(points_knn)
    return eigenentropy

def pdal_optimal_knn(las_path, min_k, max_k):
    start = time.time()
    pipeline = pdal.Pipeline(json.dumps({
        "pipeline": [
            {
                "type": "readers.las",
                "filename": las_path
            }
            ,
            {
                "type": "filters.optimalneighborhood",
                "min_k": min_k,
                "max_k": max_k
            },
            {
                "type": "writers.las",
                "compression": "laszip",
                "filename": las_path.replace('.las', '_optimalKNN.las'),
                "extra_dims": "all",
                "forward": "all",
            }
        ]
    }))

    pipeline.execute()
    end = time.time()
    print("done")
    runtime = end-start
    print(runtime)


def pdal_lof(las_path):
    start = time.time()
    pipeline = pdal.Pipeline(json.dumps({
        "pipeline": [
            {
                "type": "readers.las",
                "filename": las_path
            }
            ,
            {
                "type": "filters.lof",
                "minpts": 20
            },
            {
                "type": "writers.las",
                "compression": "laszip",
                "filename": las_path.replace('.las', '_lof.las'),
                "extra_dims": "all",
                "forward": "all",
            }
        ]
    }))

    pipeline.execute()
    end = time.time()
    print("done")
    runtime = end-start
    print(runtime)


def optimal_knn(las_path, min_k, max_k):
    start = time.time()

    # load point cloud
    las = pylas.read(las_path)

    # add extra dimensions
    las.add_extra_dim('oKNN', type='int8')
    las.add_extra_dim('oRadius', type='f8')
    las.add_extra_dim('min_eigentropy', type='f8')
    las.add_extra_dim('min_egen1', type='f8')
    las.add_extra_dim('min_egen2', type='f8')
    las.add_extra_dim('min_egen3', type='f8')

    # range of knn
    knn_range = range(min_k, max_k+2, 1)

    points_xyz = np.array(list(zip(*[las.x, las.y, las.z])))

    kdtree = cKDTree(points_xyz)

    _, knn_indices = kdtree.query(points_xyz, max_k+1, workers = 7)

    eigen_knn = pd.DataFrame()
    
    for knn in knn_range:
        points_knn = points_xyz[knn_indices[:, 1:knn+1]]          
        eigenentropy, eigenvalues = compute_eigen(points_knn)
        eigen_knn[knn] = eigenentropy

    print(eigen_knn)

    optimalNN = eigen_knn.idxmin(axis='columns')
    print(optimalNN)
    # list_eigentropy.append(eigenentropy)
    # list_eigenvalues.append(eigenvalues)
    # list_radius.append(max(distances))

    # knn_index = np.array(list_eigentropy).argmin()

    # optimal_knn = knn_range[knn_index]
    # min_eigentropy = min(list_eigentropy)
    # min_eigenvalues = list_eigenvalues[knn_index]
    # optimal_radius = list_radius[knn_index]
    print('now')
    end = time.time()
    runtime = end - start
    print('============ TOTAL RUNTIME: {} s ============'.format(runtime))
    

def optimal_knn2(las_path, min_k, max_k):
    start = time.time()
    pool_size = cpu_count()-1

    # Load point cloud
    las = pylas.read(las_path)

    # Add extra dimensions
    las.add_extra_dim('oKNN', type='int8')
    las.add_extra_dim('oRadius', type='f8')
    las.add_extra_dim('min_eigentropy', type='f8')
    las.add_extra_dim('min_egen1', type='f8')
    las.add_extra_dim('min_egen2', type='f8')
    las.add_extra_dim('min_egen3', type='f8')
    
    knn_range = np.arange(min_k, max_k + 1)

    points_xyz = np.array(list(zip(*[las.x, las.y, las.z])))


    kdtree = cKDTree(points_xyz)
    knn_indices = kdtree.query(points_xyz, max_k, workers=pool_size)[1]

    knn = knn_range[0]

    arguments = [points_xyz[knn_indices[:, 0:knn]] for knn in knn_range]
    
    with Pool(pool_size) as pool:
        results = np.array(pool.map(process_knn, arguments)).T
        eigen_knn = pd.DataFrame(results)

    eigen_knn.columns = knn_range

    print(eigen_knn)

    optimal_knn = eigen_knn.idxmin(axis=1)
    print(optimal_knn)
    print('now')
    las['oKNN'] = optimal_knn
    las.write(las_path.replace('.las', '_oknn2.las'))
    end = time.time()
    runtime = end - start
    print('============ TOTAL RUNTIME: {} s ============'.format(runtime))

def optimal_knn_perstrip(las_path, min_k, max_k):
    start = time.time()
    pool_size = cpu_count()-1

    # Load point cloud
    las = pylas.read(las_path)

    # Add extra dimensions
    las.add_extra_dim('oKNN_strip', type='int8')
    #las.add_extra_dim('oRadius', type='f8')
    #las.add_extra_dim('min_eigentropy', type='f8')
    #las.add_extra_dim('min_egen1', type='f8')
    #las.add_extra_dim('min_egen2', type='f8')
    #las.add_extra_dim('min_egen3', type='f8')
    
    knn_range = np.arange(min_k, max_k + 1)

    # points_xyz = np.array(list(zip(*[las.x, las.y, las.z])))

    for strip in np.unique(las.point_source_id):
        print(strip)
        pc_strip, pc_rest = strip_pop(las, strip)

        kdtree = cKDTree(pc_rest)
        knn_indices = kdtree.query(pc_strip, max_k, workers=pool_size)[1]

        arguments = [pc_rest[knn_indices[:, 0:knn]] for knn in knn_range]

        with Pool(pool_size) as pool:
            results = np.array(pool.map(process_knn, arguments)).T
            eigen_knn = pd.DataFrame(results)

        eigen_knn.columns = knn_range

        optimal_knn = eigen_knn.idxmin(axis=1)
        las.oKNN_strip[las.point_source_id == strip] = optimal_knn
    las.write(las_path.replace('.las', '_oknn_strip.las'))
    end = time.time()
    runtime = end - start
    print('============ TOTAL RUNTIME: {} s ============'.format(runtime))


"""
Garbage
"""

def sor_train_pcd(las_path):
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



    knn_range = range(5, 50, 5)
    std_range = np.arange(0.1, 2.6, 0.5)
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

    plt.title('Statistical outlier filter parameters - f1',fontsize=12)
    plt.show()


def denoising_strips_train(strips_path):
    start = time.time()

    # load point cloud
    las = pylas.read(strips_path)
    las.add_extra_dim('noise', type='int8')
    las.add_extra_dim('distance', type='f8')
    las.noise = np.zeros(len(las.points))
    y_true = las.classification
    print(y_true)
    y_true[y_true == 2] = 0
    y_true[y_true == 1] = 0
    y_true[y_true == 7] = 1
    print(y_true)

    # iterate for different combinations of parameters
    k_range = range(1, 3, 1)
    dist_range = np.arange(0.01, 0.11, 0.01)
    cont_table = np.ndarray((len(k_range), len(dist_range)), np.float64)


    for i, k in enumerate(k_range):
        for j, dist in enumerate(dist_range):
            # initialize controls
            pc_strip = [0]
            noise_values = [0,1]
            iter = 0
            print('===== parameters k: {}, dist: {} ====='.format(k, dist))
            while (noise_values[-1] != noise_values[-2]) and (len(pc_strip) != 0):
                try:
                    # for every strip in point cloud
                    strip_len = []
                    for strip in np.unique(las.point_source_id):
                        #print(strip)
                        # split point cloud into one strip and the rest
                        
                        pc_strip = np.array(list(zip(*[las.x[(las.point_source_id == strip) & (las.noise != 1)], 
                                            las.y[(las.point_source_id == strip) & (las.noise != 1)],
                                            las.z[(las.point_source_id == strip) & (las.noise != 1)]])))
                        
                        pc_rest = np.array(list(zip(*[las.x[(las.point_source_id != strip) & (las.noise != 1)], 
                                            las.y[(las.point_source_id != strip) & (las.noise != 1)], 
                                            las.z[(las.point_source_id != strip) & (las.noise != 1)]])))
                        print(len(pc_strip))
                        strip_len.append(len(pc_strip))
                        
                        # compute strip2cloud distance to NN
                        kdtree = cKDTree(pc_rest)
                        distances, _ = kdtree.query(pc_strip, k)
                        if k > 1:
                            distances = np.mean(distances, axis=1)

                        # asign distances to 
                        las.distance[(las.point_source_id == strip) & (las.noise != 1)] = distances
                        las.noise[las.distance >= dist] = 1
                        # print(len(las.points[las.noise == 1]))
                    noise_values.append(len(las.points[las.noise == 1]))
                    iter += 1
                    if iter >= 1:
                        break
                except ValueError:
                    break
            y_pred = las.noise
            print(y_pred)
            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='binary')
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            cont_table[i,j] = f1

            las.noise = np.zeros(len(las.points))
            las.distances = np.zeros(len(las.points))
    
    end = time.time()
    runtime = end-start
    print('============ TOTAL RUNTIME: {} s ============'.format(runtime))

    min_acc = np.amin(cont_table)
    max_acc = np.amax(cont_table)
    fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    plt.clf()

    ax = fig.add_subplot(111)

    ax.set_aspect(1)

    res = sns.heatmap(cont_table, annot=True, fmt='.2f', cmap="YlGnBu", vmin=min_acc, vmax=max_acc)
    plt.xticks(np.arange(0.5, cont_table.shape[0] + 0.5), [round(i, 2) for i in dist_range])
    plt.yticks(np.arange(0.5, cont_table.shape[1] + 0.5), list(k_range))
    plt.xlabel('distance_treshold')
    plt.ylabel('knn')

    plt.title('Statistical outlier filter parameters - f1',fontsize=12)
    plt.show()


def idontknow_strip(strips_path, radius):
    start = time.time()

    # load point cloud and add dimensions
    las = pylas.read(strips_path)
    las.add_extra_dim('noise', type='int8')
    las.add_extra_dim('distance', type='f8')
    las.add_extra_dim('density', type='f8')
    las.add_extra_dim('outlier_coef', type='f8')

    # iterate for different combinations of parameters
    r_range = np.arange(0.05, 1.5, 0)
    dist_range = np.arange(0.01, 0.11, 0.01)
    cont_table = np.ndarray((len(r_range), len(dist_range)), np.float64)


    for i, r in enumerate(r_range):
        for j, dist in enumerate(dist_range):
            # initialize controls
            pc_strip = [0]
            noise_values = [0,1]
            iter = 0
            print('===== parameters k: {}, dist: {} ====='.format(k, dist))
            while (noise_values[-1] != noise_values[-2]) and (len(pc_strip) != 0):
                try:
                    # for every strip in point cloud
                    strip_len = []
                    for strip in np.unique(las.point_source_id):
                        #print(strip)
                        # split point cloud into one strip and the rest
                        
                        pc_strip = np.array(list(zip(*[las.x[(las.point_source_id == strip) & (las.noise != 1)], 
                                            las.y[(las.point_source_id == strip) & (las.noise != 1)],
                                            las.z[(las.point_source_id == strip) & (las.noise != 1)]])))
                        
                        pc_rest = np.array(list(zip(*[las.x[(las.point_source_id != strip) & (las.noise != 1)], 
                                            las.y[(las.point_source_id != strip) & (las.noise != 1)], 
                                            las.z[(las.point_source_id != strip) & (las.noise != 1)]])))
                        print(len(pc_strip))
                        strip_len.append(len(pc_strip))
                        
                        # compute strip2cloud distance to NN
                        kdtree = cKDTree(pc_rest)
                        distances, _ = kdtree.query_ball_point(pc_strip, r)

                        if k > 1:
                            distances = np.mean(distances, axis=1)

                        # asign distances to 
                        las.distance[(las.point_source_id == strip) & (las.noise != 1)] = distances
                        las.noise[las.distance >= dist] = 1
                        # print(len(las.points[las.noise == 1]))
                    noise_values.append(len(las.points[las.noise == 1]))
                    iter += 1
                    if iter >= 1:
                        break
                except ValueError:
                    break
            y_pred = las.noise
            print(y_pred)
            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='binary')
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            cont_table[i,j] = f1

            las.noise = np.zeros(len(las.points))
            las.distances = np.zeros(len(las.points))
    
    end = time.time()
    runtime = end-start
    print('============ TOTAL RUNTIME: {} s ============'.format(runtime))

    min_acc = np.amin(cont_table)
    max_acc = np.amax(cont_table)
    fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    plt.clf()

    ax = fig.add_subplot(111)

    ax.set_aspect(1)

    res = sns.heatmap(cont_table, annot=True, fmt='.2f', cmap="YlGnBu", vmin=min_acc, vmax=max_acc)
    plt.xticks(np.arange(0.5, cont_table.shape[0] + 0.5), [round(i, 2) for i in dist_range])
    plt.yticks(np.arange(0.5, cont_table.shape[1] + 0.5), list(k_range))
    plt.xlabel('distance_treshold')
    plt.ylabel('knn')

    plt.title('Statistical outlier filter parameters - f1',fontsize=12)
    plt.show()


    # split strip and rest

    # calculate average 

def multiscale_reciprocity_removal(las_path):
    print('start')
    output_path = las_path.replace('.las', '_covf_mrr.las')
    pipeline = [
            {
                "type": "readers.las",
                "filename": las_path
            } 
            ]
    for knn in range(5,101,5):
        step1 = {
            "type":"filters.reciprocity",
                "knn": knn
               }
        pipeline.append(step1)
        step2 = {
            "type":"filters.range",
            "limits":"Reciprocity[:80.0]"
                }
        pipeline.append(step2)

    pipeline.append({
                "type": "writers.las",
                "compression": "laszip",
                "filename": output_path,
                "extra_dims": "all",
                "forward": "all"
            })


    pipeline = pdal.Pipeline(json.dumps({
        "pipeline": pipeline
    }))
    print(pipeline)

    pipeline.execute()

    print("done")


def mrr_train_old(las_path, max_knn=100, step=15):
    start = time.time()
    pool_size = cpu_count()-1
    las = pylas.read(las_path)  # Read LAS file using pylas
    
    y_true = np.array(list(las.classification))
    y_true[y_true == 2] = 0
    y_true[y_true == 1] = 0
    y_true[y_true == 7] = 1

    xyz_points = pc_h.get_xyz(las)

    tree = cKDTree(xyz_points)  # Creating KDTree for efficient nearest neighbor search
    _, indices = tree.query(xyz_points, k=max_knn + 1)  # Query k+1 nearest neighbors to exclude the point itself
    idx_full = indices[:,1:][indices[:,1:]]

    metrics_list = []

    for min_r in np.arange(0.1,1.0,0.05):
        print(min_r)
        y_pred = np.zeros(len(y_true), np.int8)
        for knn in range(5, max_knn + 1, step):
            
            idx = idx_full[:,:knn,:knn]

            reciprocal_counts = np.count_nonzero(idx == np.arange(idx.shape[0])[:, None, None], axis=(1, 2)) 
        
            reciprocity = reciprocal_counts / knn
            # print(reciprocity)
            noise_mask = reciprocity < min_r
            y_pred[noise_mask] = 1

        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='binary')
        prec = precision_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)

        metrics_list.append(f1)

    print(metrics_list)
    print(max(metrics_list))

    end = time.time()
    runtime = end - start
    print('============ TOTAL RUNTIME: {} s ============'.format(runtime))

if __name__ == '__main__':
    main()