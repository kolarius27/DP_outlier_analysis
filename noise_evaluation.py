#!/bin/python
import pylas
import numpy as np
import pandas as pd
import point_cloud_handeling as pc_h
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    test_boulder = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder4_pdal_oknn2_noise_oknn_strip_normals_factors.las'
    #boulder_cov = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder4_opals_eigen_covf_100.las'
    #boulder_cov_optim = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder4_opals_eigen_covf_optim.las'
    # boulder_cov = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder4_new_normal45_factors.las'
    boulder_mrecor_factors = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder4_attributes/boulder4_new_mrecsor_0.3_0.7_normal_0.45_50_factors.las'


    columns_cov = ['Density', 'DemantkeVerticality', 'SurfaceVariation', 'EigenvalueSum', 'Eigenentropy', 'Anisotropy', 'Omnivariance', 'Verticality', 'Scattering', 'Planarity', 'Linearity', 'Reciprocity', 'LocalOutlierFactor', 'LocalReachabilityDistance', 'NNDistance', 'OptimalRadius', 'OptimalKNN', 'point_source_id']
    columns_fact = ['scan_distance', 'scan_angle', 'incidence_angle', 'angle_difference', 'traj_azimuth', 'scan_azimuth', 'local_azimuth', 'azimuth_difference', 'slope']
    
    stats_visualization(test_boulder, columns_fact, 25)
    stats_visualization(boulder_mrecor_factors, columns_fact, 25)


def stats_visualization(las_path, columns, knn):
    las = pylas.read(las_path)
    print(list(las.point_format.dimension_names))

    las.classification = las.outliers

    las_df = las2df(las, columns)


    clean_df, noise_df = pc_splitbynoise2(las_df)


    histogram_visu(clean_df, noise_df, columns, knn)

    boxplot_visu(clean_df, noise_df, knn)



def las2df(las, columns):
    las_df = pd.DataFrame()
    for col in columns:
        las_df[col] = las[col]
    las_df['classification'] = las.classification
    return las_df


def pc_splitbynoise2(las_df):
    # split las to clean and noise
    clean_df = las_df[las_df['classification'] == 0].drop(columns='classification')
    noise_df = las_df[las_df['classification'] == 1].drop(columns='classification')

    return clean_df, noise_df


def pc_splitbynoise(las_df):
    # split las to clean and noise
    clean_df = las_df[las_df['classification'] != 7].drop(columns='classification')
    noise_df = las_df[las_df['classification'] == 7].drop(columns='classification')

    return clean_df, noise_df


def histogram_visu(clean_df, noise_df, columns, knn):
    # Determine subplot layout
    n_cols = 3
    n_rows = int(np.ceil(len(columns) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))

    while len(columns) != 0:
        for i in range(0,n_rows):
            for j in range(0,n_cols):
                col = columns[0]
                # perform Kolgormov-Smirnoff test
                stat, pval = stats.kstest(clean_df[col], noise_df[col], N=len(noise_df))

                # Plot the data
                sns.histplot(data=clean_df, x=col, color='blue', alpha=0.5, label='Clean', ax=axes[i,j])
                sns.histplot(data=noise_df, x=col, color='red', alpha=0.5, label='Noise', ax=axes[i,j])
                
                # Print statistical test results
                axes[i,j].set_title(f'{col} (p-value: {pval:.4f}, T: {stat:.4f})')
                if pval < 0.05:
                    axes[i,j].set_xlabel('Statistically different')
                else:
                    axes[i,j].set_xlabel('Statistically not different')
                axes[i,j].legend()

                columns.pop(0)

    # Hide any unused subplots
    for i in range(noise_df.shape[1], n_rows*n_cols):
        fig.delaxes(axes[i])

    fig.suptitle('Scanning geometry features'.format(knn), fontsize = 24)
    #plt.tight_layout()
    plt.show()


def boxplot_visu(clean_df, noise_df, knn):
    columns = clean_df.columns
    noise_df['Group'] = 'Noise'
    clean_df['Group'] = 'Clean'
    combined_df = pd.concat([clean_df, noise_df], ignore_index=True)
    # print(combined_df)
    # melted_df = pd.melt(combined_df, var_name='Attribute', value_name='Value', ignore_index=True)
    # print(melted_df)

    n_cols = 3
    n_rows = int(np.ceil(len(columns) / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15,5*n_rows))
    axs = axs.flatten()

    for i, attribute in enumerate(columns):
        sns.boxplot(data=combined_df, y='Group', x=attribute, ax=axs[i])
        # sns.boxplot(data=noise_df, y='Group', x=attribute, ax=axs[i], color='red')
        axs[i].set_title(attribute)

    fig.suptitle('Scanning geometry features'.format(knn), fontsize = 24)
    #plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
