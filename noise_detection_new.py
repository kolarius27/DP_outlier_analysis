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
from PyNomaly import loop


def main():
    boulder1 = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder4_smoothed_N_gauss_0.05_0.0_0.1-random_0.05_0.2.las'
    # boulder1 = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder4_smoothed_N_gauss_0.05_0.0_0.05-random_0.05_0.2.las'
    boulder1_smooth = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder4_smoothed_N.las'

    # boulder2 = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder2_smoothed_gauss_0.07_0.0_0.08-random_0.07_0.15.las'
    boulder2_smooth = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/synthetic/data/boulder2_smoothed.las'

    
    # train_outlier_detection(boulder1, max_k=50)
    # train_outlier_detection(boulder2, max_k=50)
    test_robustness(boulder1_smooth)



def train_outlier_detection(las_path, max_k):
    start_total = time.time()
    threads = cpu_count()-1

    _, xyz_points, _, y_true  = prepare_las(las_path)

    tree = cKDTree(xyz_points)

    tree_query = tree.query(xyz_points, k = max_k+1, workers = threads)
    distances = tree_query[0][:,1:]
    indices = tree_query[1][:,1:]
    idx = indices[indices]

    sor(y_true, distances, 'normal')
    sor(y_true, distances, 'robust', min_nsigma = 1.5, max_nsigma = 5.5, step_nsigma=1.0)
    ror(y_true, distances)
    recor(y_true, idx)
    #mrecor2(y_true, idx, step_k=5, max_k=20)



    end_total = time.time()
    runtime = end_total-start_total
    print('============ TOTAL RUNTIME: {} s ============'.format(runtime))

def test_robustness(las_path):
    threads = cpu_count()-1
    # prepare las
    _, xyz_points, normals, _  = prepare_las(las_path)

    xyz_points_n = normalize_coord(xyz_points)

    cont_df = pd.DataFrame(columns=['noise', 'algo', 'std', 'amount', 'metric', 'values'])

    std_range = np.arange(0.05, 0.201, 0.01)
    scale_range = np.arange(0.05, 0.4, 0.025)
    amount_range = np.arange(0.01, 0.201, 0.01)


    for amount in amount_range:
        print(amount)
        for std in std_range:
            xyz_points_gauss, y_true = add_gaussian_noise(xyz_points_n, normals, amount, 0.0, std)

            tree = cKDTree(xyz_points_gauss)

            tree_query = tree.query(xyz_points_gauss, k = 50+1, workers = threads)
            distances = tree_query[0][:,1:]
            indices = tree_query[1][:,1:]
            idx = indices[indices]

            sor_test(cont_df, 'gauss', y_true, distances, 'normal', std, amount, knn=2, nsigma=1.1)
            sor_test(cont_df, 'gauss', y_true, distances, 'robust', std, amount, knn=2, nsigma=4.0)
            ror_test(cont_df, 'gauss', y_true, distances, std, amount, k=2, r=0.06)
            recor_test(cont_df, 'gauss', y_true, idx, std, amount, knn=15, minrec=0.55)
            mrecor_test(cont_df, 'gauss', y_true, idx, std, amount, minrec=0.7)

        for scale in scale_range:
            xyz_points_gauss, y_true = add_random_noise(xyz_points_n, amount, scale)
            tree = cKDTree(xyz_points_gauss)

            tree_query = tree.query(xyz_points_gauss, k = 50+1, workers = threads)
            distances = tree_query[0][:,1:]
            indices = tree_query[1][:,1:]
            idx = indices[indices]

            sor_test(cont_df, 'random', y_true, distances, 'normal', scale, amount, knn=2, nsigma=1.1)
            sor_test(cont_df, 'random', y_true, distances, 'robust', scale, amount, knn=2, nsigma=4.0)
            ror_test(cont_df, 'random', y_true, distances, scale, amount, k=2, r=0.06)
            recor_test(cont_df, 'random', y_true, idx, scale, amount, knn=15, minrec=0.55)
            mrecor_test(cont_df, 'random', y_true, idx, scale, amount, minrec=0.7)

    
    create_robustness_plots(cont_df, _)
    print(cont_df)


"""
ALGORITHMS to test robustness
"""
def sor_test(cont_df, ntype, y_true, distances, version, std, amount, knn=2, nsigma=1.1):
    if version=='normal':
            knn_distances = np.mean(distances[:,:knn], axis=1)
            avg_distance = np.mean(knn_distances)
            std_distance = np.std(knn_distances)
    elif version=='robust':
            knn_distances = np.median(distances[:,:knn], axis=1)
            avg_distance = np.median(knn_distances)
            abs_diff = np.abs(knn_distances - avg_distance)
            std_distance = np.median(abs_diff)
    
    y_pred = np.zeros(len(y_true), np.int8)

    max_distance = avg_distance + nsigma * std_distance

    y_pred[knn_distances >= max_distance] = 1

    if version=='normal':
        algo = 'SOR'
    elif version=='robust':
        algo = 'rSOR'

    add_to_df3(y_true, y_pred, cont_df, ntype, algo, std, amount)

def ror_test(cont_df, ntype, y_true, distances, std, amount, k=2, r=0.06):
    mask = distances[:,:k] < r
    count_k = np.sum(mask, axis = 1)

    y_pred = np.zeros(len(y_true), np.int8)
    pred_mask = count_k < k
    y_pred[pred_mask] = 1

    add_to_df3(y_true, y_pred, cont_df, ntype, 'ROR', std, amount)

def recor_test(cont_df, ntype, y_true, idx, std, amount, knn=15, minrec=0.55):
    idx_knn = idx[:, :int(knn), :int(knn)]
    reciprocal_counts = np.count_nonzero(idx_knn == np.arange(idx_knn.shape[0])[:, None, None], axis=(1, 2)) 
    reciprocity = 1-(reciprocal_counts / knn)

    y_pred = np.zeros(len(y_true), np.int8)
    y_pred[reciprocity > minrec] = 1

    add_to_df3(y_true, y_pred, cont_df, ntype, 'RecOR', std, amount)

def mrecor_test(cont_df, ntype, y_true, idx, std, amount, minrec):
    reciprocity_list = []
    knn_range = np.arange(5, 50+1, 1)

    for knn in knn_range:
        idx_knn = idx[:, :int(knn), :int(knn)]
        reciprocal_counts = np.count_nonzero(idx_knn == np.arange(idx_knn.shape[0])[:, None, None], axis=(1, 2)) 
        reciprocity = 1-(reciprocal_counts / knn)
        reciprocity_list.append(reciprocity)
    
    final_reciprocity = np.max(reciprocity_list, axis=0)

    y_pred = np.zeros(len(y_true), np.int8)
    y_pred[final_reciprocity > minrec] = 1

    add_to_df3(y_true, y_pred, cont_df, ntype, 'mRecOR', std, amount)

    
    

"""
ALGORITHMS to test parameters
"""
def sor(y_true, distances, version, min_k = 2, max_k = 22, min_nsigma = 0.1, max_nsigma = 3.0, step_k = 5, step_nsigma = 0.6):
    x = 'nsigma'
    y = 'knn'
    cont_df = pd.DataFrame(columns=[x, y, 'metric', 'values'])
    
    knn_range = np.arange(min_k, max_k+1, step_k)
    nsigma_range = np.arange(min_nsigma, max_nsigma+0.1, step_nsigma)

    for knn in knn_range:

        if version=='normal':
            knn_distances = np.mean(distances[:,:knn], axis=1)
            avg_distance = np.mean(knn_distances)
            std_distance = np.std(knn_distances)
        elif version=='robust':
            knn_distances = np.median(distances[:,:knn], axis=1)
            avg_distance = np.median(knn_distances)
            abs_diff = np.abs(knn_distances - avg_distance)
            std_distance = np.median(abs_diff)

        for nsigma in nsigma_range:

            y_pred = np.zeros(len(y_true), np.int8)

            max_distance = avg_distance + nsigma * std_distance

            y_pred[knn_distances >= max_distance] = 1

            add_to_df(y_true, y_pred, cont_df, x, y, nsigma, knn)

    if version=='normal':
        title = 'SOR'
    elif version=='robust':
        title = 'rSOR'
        
    create_tables(cont_df, x, y, title)

    print('{} completed'.format(title))


def ror(y_true, distances, min_k = 1, max_k = 5, min_r = 0.02, max_r = 0.1, step_k = 1, step_r = 0.02):
    x='r'
    y='k'
    cont_df = pd.DataFrame(columns=[x, y, 'metric', 'values'])

    r_range = np.arange(min_r, max_r+0.01, step_r)
    k_range = np.arange(min_k, max_k+1, step_k)

    for r in r_range:
        mask = distances < r
        count_k = np.sum(mask, axis=1)
        for k in k_range:
            y_pred = np.zeros(len(y_true), np.int8)
            pred_mask = count_k < k
            y_pred[pred_mask] = 1

            add_to_df(y_true, y_pred, cont_df, x, y, r, k)

    create_tables(cont_df, x, y, 'ROR')

    print('ROR completed')


def recor(y_true, idx, min_rec = 0.25, max_rec = 0.85, min_k = 5, max_k = 45, step_rec = 0.15, step_k = 10):
    x = 'minrec'
    y = 'knn'
    cont_df = pd.DataFrame(columns=[x, y, 'metric', 'values'])

    rec_range = np.arange(min_rec, max_rec+0.01, step_rec)
    knn_range = np.arange(min_k, max_k+1, step_k)

    for knn in knn_range:
        idx_knn = idx[:, :int(knn), :int(knn)]
        reciprocal_counts = np.count_nonzero(idx_knn == np.arange(idx_knn.shape[0])[:, None, None], axis=(1, 2)) 
        reciprocity = 1-(reciprocal_counts / knn)

        for minrec in rec_range:
            y_pred = np.zeros(len(y_true), np.int8)
            y_pred[reciprocity > minrec] = 1

            add_to_df(y_true, y_pred, cont_df, x, y, round(minrec,2), knn)

    create_tables(cont_df, x, y, 'RecOR')

    print('RecOR completed')


def mrecor(y_true, idx, min_rec = 0.0, max_rec = 1.0, min_k = 5, max_k = 50, step_rec = 0.01, step_k = 1):
    knn_range = np.arange(min_k, max_k+1, step_k)
    rec_range = np.arange(min_rec, max_rec+0.01, step_rec)

    reciprocity_list = []
    f1_list = []
    rec_list = []
    prec_list = []
    outp_list = []

    for knn in knn_range:
        idx_knn = idx[:, :int(knn), :int(knn)]
        reciprocal_counts = np.count_nonzero(idx_knn == np.arange(idx_knn.shape[0])[:, None, None], axis=(1, 2)) 
        reciprocity = 1-(reciprocal_counts / knn)

        reciprocity_list.append(reciprocity)

    print(reciprocity_list[0])
    print(reciprocity_list[-1])
    final_reciprocity = np.max(reciprocity_list, axis=0)
    print(final_reciprocity)
    
    f1_list = []
    for minrec in rec_range:
        y_pred = np.zeros(len(y_true), np.int8)
        
        y_pred[final_reciprocity > minrec] = 1
        f1_list.append(f1_score(y_true, y_pred, average='binary'))
        prec_list.append(precision_score(y_true, y_pred))
        rec_list.append(recall_score(y_true, y_pred))
        outp_list.append(round(np.sum(y_pred)/len(y_pred),2))

    df = pd.DataFrame(data={'reciprocity threshold': rec_range, 
                            'F1': f1_list,
                            'recall': rec_list,
                            'precision': prec_list,
                            'outlier_percentage': outp_list})
    
    create_lineplot(df, 'mRecOR', 'reciprocity threshold')

def mrecor2(y_true, idx, min_rec = 0.0, max_rec = 1.0, min_k = 5, max_k = 50, step_rec = 0.01, step_k = 1):
    knn_range = np.arange(min_k, max_k+1, step_k)
    rec_range = np.arange(min_rec, max_rec+0.01, step_rec)

    x = 'minrec'

    cont_df = pd.DataFrame(columns=[x, 'metric', 'values'])

    reciprocity_list = []
   
    for knn in knn_range:
        idx_knn = idx[:, :int(knn), :int(knn)]
        reciprocal_counts = np.count_nonzero(idx_knn == np.arange(idx_knn.shape[0])[:, None, None], axis=(1, 2)) 
        reciprocity = 1-(reciprocal_counts / knn)

        reciprocity_list.append(reciprocity)

    print(reciprocity_list[0])
    print(reciprocity_list[-1])
    final_reciprocity = np.max(reciprocity_list, axis=0)
    print(final_reciprocity)
    
    for minrec in rec_range:
        y_pred = np.zeros(len(y_true), np.int8)
        
        y_pred[final_reciprocity > minrec] = 1
        add_to_df2(y_true, y_pred, cont_df, x, round(minrec,2))
    
    create_lineplot_new(cont_df, x, 'mRecOR')

"""
UTILS
"""
def add_gaussian_noise(xyz_points, normals, amount, mean, std):
    total_points = len(xyz_points)
    num_points_to_move = int(amount*total_points)
    idx = np.random.choice(total_points, num_points_to_move, replace=False)

    noise = np.vstack(np.random.normal(mean, std, num_points_to_move))
    xyz_points_new = xyz_points.copy()
    xyz_points_new[idx] = xyz_points[idx] + normals[idx] * noise

    y_true = np.zeros(total_points, np.int8)
    y_true[idx] = 1

    return xyz_points_new, y_true

def add_random_noise(xyz_points, amount, scale):
    total_points = len(xyz_points)
    num_points_to_move = int(amount*total_points)
    idx = np.random.choice(total_points, num_points_to_move, replace=False)
    
    noise = np.vstack([np.random.uniform(-1., 1., num_points_to_move),
                        np.random.uniform(-1., 1., num_points_to_move),
                        np.random.uniform(-1., 1., num_points_to_move)]).T*scale
    
    xyz_points_new = xyz_points.copy()
    xyz_points_new[idx] += noise

    y_true = np.zeros(total_points, np.int8)
    y_true[idx] = 1

    return xyz_points_new, y_true
    



def prepare_synthetic_dataset(las_path, amount_g=0.05, amount_r=0.05, g_mean=0., g_std=0.1, r_scale=0.2):
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

def normalize_coord(xyz_points):
    t_xyz = xyz_points.T
    x_offset = min(np.max(t_xyz[0]), np.min(t_xyz[0]))
    y_offset = min(np.max(t_xyz[1]), np.min(t_xyz[1]))
    z_offset = min(np.max(t_xyz[2]), np.min(t_xyz[2]))
    offsets = np.array([x_offset, y_offset, z_offset])
    fix_coords = xyz_points - offsets
    return fix_coords
    

def prepare_las(las_path):
    # load dataset
    las = pylas.read(las_path)

    # prepare y_true
    y_true = np.array(list(las.classification))
    y_true[y_true == 2] = 0
    y_true[y_true == 1] = 0
    y_true[y_true == 7] = 1

    xyz_points = pc_h.get_xyz(las)

    normals = pc_h.get_normals(las)

    return las, xyz_points, normals, y_true

def add_to_df(y_true, y_pred, cont_df, x, y, x_val, y_val):
    f1 = f1_score(y_true, y_pred, average='binary')
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    outlier_percentage = np.sum(y_pred)/len(y_pred)

    cont_df.loc[len(cont_df)] = {x: x_val, y: y_val, 'metric': 'F1', 'values': f1}
    cont_df.loc[len(cont_df)] = {x: x_val, y: y_val, 'metric': 'precision', 'values': prec}
    cont_df.loc[len(cont_df)] = {x: x_val, y: y_val, 'metric': 'recall', 'values': rec}
    cont_df.loc[len(cont_df)] = {x: x_val, y: y_val, 'metric': 'outlier percentage', 'values': outlier_percentage}

def add_to_df2(y_true, y_pred, cont_df, x,x_val):
    f1 = f1_score(y_true, y_pred, average='binary')
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    outlier_percentage = np.sum(y_pred)/len(y_pred)

    cont_df.loc[len(cont_df)] = {x: x_val, 'metric': 'F1', 'values': f1}
    cont_df.loc[len(cont_df)] = {x: x_val, 'metric': 'precision', 'values': prec}
    cont_df.loc[len(cont_df)] = {x: x_val, 'metric': 'recall', 'values': rec}
    cont_df.loc[len(cont_df)] = {x: x_val, 'metric': 'outlier percentage', 'values': outlier_percentage}

def add_to_df3(y_true, y_pred, cont_df, ntype, algo, std, amount):
    f1 = f1_score(y_true, y_pred, average='binary')
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    detection_rate = ((np.sum(y_pred)/len(y_pred))/amount)-1

    cont_df.loc[len(cont_df)] = {'noise': ntype, 'algo': algo, 'std': std, 'amount': amount, 'metric': 'F1', 'values': f1}
    cont_df.loc[len(cont_df)] = {'noise': ntype, 'algo': algo, 'std': std, 'amount': amount, 'metric': 'precision', 'values': prec}
    cont_df.loc[len(cont_df)] = {'noise': ntype, 'algo': algo, 'std': std, 'amount': amount, 'metric': 'recall', 'values': rec}
    cont_df.loc[len(cont_df)] = {'noise': ntype, 'algo': algo, 'std': std, 'amount': amount, 'metric': 'detection rate', 'values': detection_rate}

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
    plt.plot(x_label, 'F1', data=df, label='F1')
    plt.plot(x_label, 'recall', data=df, label='recall')
    plt.plot(x_label, 'precision', data=df, label='precision')
    plt.plot(x_label, 'outlier_percentage', data=df, label='outlier percentage', c='m')
    annot_max(df, x_label)
    plt.xlabel(x_label)
    plt.legend()
    plt.title(title,fontsize=16)
    plt.xlim(0.0)
    plt.ylim(0)
    plt.show()



def create_robustness_plots(df, title):
    sns.relplot(
    data=df, x='std', y='values', col='metric', row='noise',
    hue='algo', kind="line", facet_kws=dict(sharey=False, sharex=False)
    )


def annot_max(df, x_label):
    # print(df)
    max_index = df['F1'].idxmax()
    row = df.iloc[max_index]
    print(row)
    text = '{}={:.2f}\n F1={:.2f}\n precision={:.2f}\n recall={:.2f}\n outlier percentage={:.2f}'.format(x_label, 
                                             row[x_label], row['F1'], row['precision'],
                                             row['recall'], row['outlier_percentage'])
    
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=120")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props,  ha="right", va="top")
    plt.annotate(text, xy=(row[x_label], row['F1']), xytext=(0.3,0.98), **kw)


def create_lineplot_new(df, x, name):
    sns.set_theme()
    g = sns.lineplot(data=df, x=x, y='values', hue = 'metric')
    g.set_xlim(0.0, 1.0)
    g.set_ylim(0.0, 1.0)

    g.set_title(name, fontsize=20, fontweight=800)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    

    # Find the row with the maximum 'values' for 'metric' = 'F1'
    max_f1_row = df[df['metric'] == 'F1'].nlargest(1, 'values')

    # Get the corresponding 'minrec' value
    max_minrec_value = max_f1_row[x].values[0].copy()

    # Find all rows with the same 'minrec'
    rows = df[df[x] == max_minrec_value]
    rows.iloc[0] = {'metric': x, 'values': max_minrec_value}   

    # g.text(1.05, 0.5, txt)

    plt.axvline(max_minrec_value, linestyle=':')
    g.text(max_minrec_value-0.02, -0.065, max_minrec_value, fontsize=10, fontweight=600)

    table = plt.table(cellText = rows['values'], rowLabels=rows['metric'])

"""
garbage
"""

def mrecor_old(y_true, idx, min_rec = 0.25, max_rec = 0.85, min_k = 5, max_k = 50, step_rec = 0.05, step_k = 10):
    knn_range = np.arange(min_k, max_k+1, step_k)
    rec_range = np.arange(min_rec, max_rec+0.01, step_rec)

    y_pred_final = np.zeros(len(y_true), np.int8)
    for knn in knn_range:

        idx_knn = idx[:, :int(knn), :int(knn)]
        reciprocal_counts = np.count_nonzero(idx_knn == np.arange(idx_knn.shape[0])[:, None, None], axis=(1, 2)) 
        reciprocity = 1-(reciprocal_counts / knn)

        minrec_list = []
        for minrec in rec_range:
            y_pred_test = np.zeros(len(y_true), np.int8)
            y_pred_test[reciprocity > minrec] = 1
            minrec_list.append(f1_score(y_true, y_pred_test))

        minrec_index = minrec_list.index(max(minrec_list))
        minrec_best = list(rec_range)[minrec_index]
        print(minrec_best)

        y_pred_final[reciprocity > minrec_best] += 1
    
    print(max(y_pred_final))

    f1_list = []
    for m in np.arange(1,len(knn_range),1):
        y_pred = np.zeros(len(y_true), np.int8)
        y_pred[y_pred_final > m] = 1
        f1_list.append(f1_score(y_true, y_pred))

    print(max(f1_list))


def mrecsor(y_true, distances, idx, min_k = 2, max_k = 55, min_nsigma = 0.1, max_nsigma = 5.0, min_rec = 0.1, max_rec = 0.9, step_k=1, step_nsigma=0.1, step_rec = 0.05):
    knn_range = np.arange(min_k, max_k+1, step_k)
    nsigma_range = np.arange(min_nsigma, max_nsigma+0.1, step_nsigma)
    rec_range = np.arange(min_rec, max_rec+0.01, step_rec)

    f1_list = []

    for knn in knn_range:
        y_pred_final = np.zeros(len(y_true), np.int8)

        #knn_distances = np.mean(distances[:,:knn], axis=1)
        #avg_distance = np.mean(knn_distances)
        #std_distance = np.std(knn_distances)

        knn_distances = np.median(distances[:,:knn], axis=1)
        avg_distance = np.median(knn_distances)
        abs_diff = np.abs(knn_distances - avg_distance)
        std_distance = np.median(abs_diff)

        nsigma_list = []
        for nsigma in nsigma_range:
            y_pred_test = np.zeros(len(y_true), np.int8)
            max_distance = avg_distance + std_distance * nsigma
            y_pred_test[knn_distances >= max_distance] = 1

            nsigma_list.append(f1_score(y_true, y_pred_test))

        nsigma_index = nsigma_list.index(max(nsigma_list))
        nsigma_best = list(nsigma_range)[nsigma_index]
        
        max_distance = avg_distance + std_distance * nsigma_best

        idx_knn = idx[:, :int(knn), :int(knn)]
        reciprocal_counts = np.count_nonzero(idx_knn == np.arange(idx_knn.shape[0])[:, None, None], axis=(1, 2)) 
        reciprocity = 1-(reciprocal_counts / knn)

        minrec_list = []
        for minrec in rec_range:
            y_pred_test = np.zeros(len(y_true), np.int8)
            y_pred_test[reciprocity > minrec] = 1
            minrec_list.append(f1_score(y_true, y_pred_test))

        minrec_index = minrec_list.index(max(minrec_list))
        minrec_best = list(rec_range)[minrec_index]

        y_pred_final[(knn_distances >= max_distance) & (reciprocity > minrec_best)] = 1

        f1_list.append(f1_score(y_true, y_pred_final))

    print(max(f1_list))

def mrecsor2(y_true, distances, idx, min_k = 2, max_k = 55, min_nsigma = 0.1, max_nsigma = 5.0, min_rec = 0.1, max_rec = 0.9, step_k=1, step_nsigma=0.1, step_rec = 0.05): 
    
    knn_range = np.arange(min_k, max_k+1, step_k)
    nsigma_range = np.arange(min_nsigma, max_nsigma+0.1, step_nsigma)
    rec_range = np.arange(min_rec, max_rec+0.01, step_rec)

    y_pred_sor_list = []
    f1_sor_list = []
    reciprocity_list = []
    y_pred_rec_list = []
    f1_reciprocity_list = []
    y_pred_final = np.zeros(len(y_true), np.int8)
    for knn in knn_range:
        print(knn)

        knn_distances = np.mean(distances[:,:knn], axis=1)
        avg_distance = np.mean(knn_distances)
        std_distance = np.std(knn_distances)

        # knn_distances = np.median(distances[:,:knn], axis=1)
        # avg_distance = np.median(knn_distances)
        # abs_diff = np.abs(knn_distances - avg_distance)
        # std_distance = np.median(abs_diff)

        for nsigma in nsigma_range:
            y_pred_test = np.zeros(len(y_true), np.int8)
            max_distance = avg_distance + std_distance * nsigma
            y_pred_test[knn_distances >= max_distance] = 1
            y_pred_sor_list.append(y_pred_test)
            f1_sor_list.append(f1_score(y_true, y_pred_test))

        idx_knn = idx[:, :int(knn), :int(knn)]
        reciprocal_counts = np.count_nonzero(idx_knn == np.arange(idx_knn.shape[0])[:, None, None], axis=(1, 2)) 
        reciprocity = 1-(reciprocal_counts / knn)
        reciprocity_list.append(reciprocity)


    max_reciprocity = np.max(reciprocity_list, axis=0)

    for minrec in rec_range:
        y_pred_test = np.zeros(len(y_true), np.int8)
        y_pred_test[max_reciprocity > minrec] = 1
        y_pred_rec_list.append(y_pred_test)
        f1_reciprocity_list.append(f1_score(y_true, y_pred_test))

    print(f1_reciprocity_list)

    best_minrec_index =f1_reciprocity_list.index(max(f1_reciprocity_list))
    y_pred_rec = y_pred_rec_list[best_minrec_index]

    print(y_pred_rec)

    print(max(f1_sor_list))
    best_sor_index = f1_sor_list.index(max(f1_sor_list))
    y_pred_sor = y_pred_sor_list[best_sor_index]

    print(y_pred_sor)

    y_pred_final = y_pred_rec & y_pred_sor

    print(y_pred_final)
    final_f1_score = f1_score(y_true, y_pred_final)

    print(final_f1_score)

def mrecsor3(y_true, distances, idx, min_k = 2, max_k = 55, min_nsigma = 0.1, max_nsigma = 5.0, min_rec = 0.1, max_rec = 0.9, step_k=1, step_nsigma=0.1, step_rec = 0.05):
    x = 'nsigma'
    y = 'minrec'
    z = 'knn'
    
    knn_range = np.arange(min_k, max_k+1, step_k)
    nsigma_range = np.arange(min_nsigma, max_nsigma+0.1, step_nsigma)
    rec_range = np.arange(min_rec, max_rec+0.01, step_rec)

    cont_df = pd.DataFrame(columns=[x, y, z, 'metric', 'values'])

    for knn in knn_range:
        print(knn)
        knn_distances = np.mean(distances[:,:knn], axis=1)
        avg_distance = np.mean(knn_distances)
        std_distance = np.std(knn_distances)

        # knn_distances = np.median(distances[:,:knn], axis=1)
        # avg_distance = np.median(knn_distances)
        # abs_diff = np.abs(knn_distances - avg_distance)
        # std_distance = np.median(abs_diff)

        idx_knn = idx[:, :int(knn), :int(knn)]
        reciprocal_counts = np.count_nonzero(idx_knn == np.arange(idx_knn.shape[0])[:, None, None], axis=(1, 2)) 
        reciprocity = 1-(reciprocal_counts / knn)

        for nsigma in nsigma_range:
            max_distance = avg_distance + std_distance * nsigma
            for minrec in rec_range:
                y_pred = np.zeros(len(y_true), np.int8)
                y_pred[(reciprocity > minrec) & (knn_distances >= max_distance)] = 1
                f1 = f1_score(y_true, y_pred, average='binary')
                cont_df.loc[len(cont_df)] = {x: nsigma, y: minrec, z: knn, 'metric': 'F1', 'values': f1}
        
    max_value_indices = cont_df.groupby('knn')['values'].idxmax()

    # Select the rows with the maximum 'values' for each 'knn'
    result_df = cont_df.loc[max_value_indices]

    # Display the result
    print(result_df)
    print(np.max(result_df['values']))


def mrecsor4(y_true, distances, idx, min_k = 5, max_k = 5, min_nsigma = 0.1, max_nsigma = 1.0, min_rec = 0.1, max_rec = 0.9, step_k=1, step_nsigma=0.2, step_rec = 0.2):
    x = 'nsigma'
    y = 'minrec'
    
    knn_range = np.arange(min_k, max_k+1, step_k)
    nsigma_range = np.arange(min_nsigma, max_nsigma+0.1, step_nsigma)
    rec_range = np.arange(min_rec, max_rec+0.01, step_rec)

    cont_df = pd.DataFrame(columns=[x, y, 'metric', 'values'])

    min_knn_distances = np.mean(distances[:,:2], axis=1)
    mean_knn_distance = np.mean(min_knn_distances)
    std_knn_distance = np.std(min_knn_distances)
    #abs_diff = np.abs(min_knn_distances - mean_knn_distance)
    # std_knn_distance = np.median(abs_diff)

    reciprocity_list = []
    for knn in knn_range:
        idx_knn = idx[:, :int(knn), :int(knn)]
        reciprocal_counts = np.count_nonzero(idx_knn == np.arange(idx_knn.shape[0])[:, None, None], axis=(1, 2)) 
        reciprocity = 1-(reciprocal_counts / knn)
        reciprocity_list.append(reciprocity)

    max_reciprocity = np.max(reciprocity_list, axis=0)

    for nsigma in nsigma_range:
        max_distance = mean_knn_distance + std_knn_distance * nsigma
        for minrec in rec_range:
            y_pred = np.zeros(len(y_true), np.int8)
            y_pred[(max_reciprocity > minrec) & (min_knn_distances >= max_distance)] = 1
            add_to_df(y_true, y_pred, cont_df, x, y, round(nsigma,2), round(minrec,2))

    create_tables(cont_df, x, y, 'mRecSOR')

    print('mRecSOR completed')

def test2(y_true, indices, distances, idx):
    # data = normalize_coord(xyz_points)
    
    p_range = np.arange(0.1, 0.9, 0.05)
    knn_range = np.arange(1, 25+1, 1)
    for knn in knn_range:
        f1_list = []
        scores = loop.LocalOutlierProbability(neighbor_matrix=indices[:,:knn], distance_matrix=distances[:,:knn], n_neighbors=knn).fit().local_outlier_probabilities
        idx_knn = idx[:, :int(knn), :int(knn)]
        reciprocal_counts = np.count_nonzero(idx_knn == np.arange(idx_knn.shape[0])[:, None, None], axis=(1, 2)) 
        reciprocity = 1-(reciprocal_counts / knn)

        combined_prob = scores * reciprocity
        for p in p_range:
            y_pred = np.zeros(len(y_true), np.int8)
            y_pred[combined_prob > p] = 1
            f1 = f1_score(y_true, y_pred, average= 'binary')
            f1_list.append(f1)
        print(max(f1_list))

def test(y_true, indices, distances, knn):
    # data = normalize_coord(xyz_points)
    
    p_range = np.arange(0.1, 0.9, 0.05)
    knn_range = np.arange(1, 25+1, 1)
    for knn in knn_range:
        f1_list = []
        scores = loop.LocalOutlierProbability(neighbor_matrix=indices[:,:knn], distance_matrix=distances[:,:knn], n_neighbors=knn).fit().local_outlier_probabilities
        
        
        for p in p_range:
            y_pred = np.zeros(len(y_true), np.int8)
            y_pred[scores > p] = 1
            f1 = f1_score(y_true, y_pred, average= 'binary')
            f1_list.append(f1)
        print(max(f1_list))


def median_outlier(y_true, distances):
    knn_range = np.arange(1, 25+1, 1)
    m_range = np.arange(1.5, 3.5+0.1, 0.1)

    for knn in knn_range:
        f1_list = []
        knn_distances = np.median(distances[:,:knn], axis=1)
        max_distances = np.max(distances[:,:knn], axis=1)
        median_distance = np.median(knn_distances)
        for m in m_range:
            y_pred = np.zeros(len(y_true), np.int8)
            y_pred[max_distances > (m*median_distance)] = 1
            f1 = f1_score(y_true, y_pred, average= 'binary')
            f1_list.append(f1)
        print(max(f1_list))


def msor(y_true, distances, min_k = 5, max_k = 55, min_nsigma=0.1, max_nsigma=3.0, step_k = 1, step_nsigma = 0.1):
    knn_range = np.arange(min_k, max_k+1, step_k)
    nsigma_range = np.arange(min_nsigma, max_nsigma+0.1, step_nsigma)
    m_range = np.arange(0.1, 1.0, 0.05)

    y_pred_final = np.zeros(len(y_true), np.int8)

    for knn in knn_range:
        knn_distances = np.min(distances[:,:knn], axis=1)
        avg_distance = np.mean(knn_distances)
        std_distance = np.std(knn_distances)
        nsigma_list = []
        for nsigma in nsigma_range:
            y_pred_test = np.zeros(len(y_true), np.int8)
            max_distance = avg_distance + std_distance * nsigma
            y_pred_test[knn_distances >= max_distance] = 1

            nsigma_list.append(f1_score(y_true, y_pred_test))

        nsigma_index = nsigma_list.index(max(nsigma_list))
        nsigma_best = list(nsigma_range)[nsigma_index]
        print(nsigma_best)
        
        max_distance = avg_distance + std_distance * nsigma_best
        y_pred_final[knn_distances >= max_distance] += 1


    f1_list = []
    for knn in knn_range:
        y_pred = np.zeros(len(y_true), np.int8)
        y_pred[y_pred_final > knn] = 1
        f1_list.append(f1_score(y_true, y_pred))

    print(max(f1_list))



if __name__ == '__main__':
    main()