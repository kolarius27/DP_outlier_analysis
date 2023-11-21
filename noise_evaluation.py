#!/bin/python
import pylas
import numpy as np
import pandas as pd
import point_cloud_handeling as pc_h
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import math as m
from statistics import mode
from astropy.stats import circstats
from pycircstat2 import Circular, load_data, descriptive
from pycircular.stats import periodic_mean_std, periodic
import pycircular


def main():
    test_boulder = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder4_attributes/boulder4_new_normal_0.45_50_factors.las'
    #boulder_cov = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder4_opals_eigen_covf_100.las'
    #boulder_cov_optim = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder4_opals_eigen_covf_optim.las'
    # boulder_cov = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder4_new_normal45_factors.las'
    boulder_mrecor_factors = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder4_attributes/boulder4_new_mrecsor_0.3_0.7_normal_0.45_50_factors.las'


    columns_cov = ['Density', 'DemantkeVerticality', 'SurfaceVariation', 'EigenvalueSum', 'Eigenentropy', 'Anisotropy', 'Omnivariance', 'Verticality', 'Scattering', 'Planarity', 'Linearity', 'Reciprocity', 'LocalOutlierFactor', 'LocalReachabilityDistance', 'NNDistance', 'OptimalRadius', 'OptimalKNN', 'point_source_id']
    columns_fact = ['scan_distance', 'scan_angle', 'incidence_angle', 'angle_difference', 'traj_azimuth', 'scan_azimuth', 'local_azimuth', 'azimuth_difference', 'slope']
    
    columns_fact_small = ['scan_distance', 'scan_angle', 'incidence_angle', 'slope']
    columns_fact_circular = ['traj_azimuth', 'scan_azimuth', 'local_azimuth']
    columns_fact_fused = ['angle_difference', 'azimuth_difference', 'azimuth_deviation']
    # stats_visualization(test_boulder, columns_fact, 25)
    # stats_visualization_new(boulder_mrecor_factors, columns_fact_small)
    stats_visualization_circular(boulder_mrecor_factors, columns_fact_circular)
    # stats_visualization_new(boulder_mrecor_factors, columns_fact_small)
    # stats_visualization_new(boulder_mrecor_factors, columns_fact_fused)
    # boxplot_test()
    # print(m.sin(m.pi/2))
    # pycircular_test()



def pycircular_test(las_df):
    c21 = Circular(data=las_df, unit='degree')

    fig, ax = plt.subplot_mosaic(mosaic="ab", figsize=(12, 6), per_subplot_kw={'b': {'projection': 'polar'}})

    ax['a'].scatter(np.arange(len(las_df)), las_df, s=10, color='black')
    ax['a'].set_xlabel('Observation number')
    ax['a'].set_ylabel('Wind direction (in radians')

    c21.plot(
        ax=ax['b'],
        plot_rose=False,
        plot_density=False,
        plot_mean=False,
        plot_median=False,
        r_max_scatter=1,
        marker_color='black'
    )

    from matplotlib import ticker

    position_major = np.arange(0, 2 * np.pi, 2 * np.pi / 4)
    labels = ['N', 'E', 'S', 'W']
    ax['b'].xaxis.set_major_locator(ticker.FixedLocator(position_major))
    ax['b'].xaxis.set_major_formatter(ticker.FixedFormatter(labels))



def stats_visualization(las_path, columns, knn):
    las = pylas.read(las_path)
    print(list(las.point_format.dimension_names))

    las.classification = las.outliers

    las_df = las2df(las, columns)


    clean_df, noise_df = pc_splitbynoise2(las_df)


    histogram_visu(clean_df, noise_df, columns, knn)

    boxplot_visu(clean_df, noise_df, knn)


def stats_visualization_new(las_path, columns):
    las = pylas.read(las_path)
    print(list(las.point_format.dimension_names))

    las.classification = las.outliers

    las_df_ = las2df(las, columns)
    las_df = pd.melt(las_df_, id_vars='classification', value_vars=columns)


    # clean_df, noise_df = pc_splitbynoise2(las_df)

    # pairplot_visu(las_df)

    # long_df = histogram_visu_new(las_df, columns)
    # df = pd.melt(las_df, id_vars='classification', value_vars=columns)

    histogram_visu_new(las_df)
    boxplot_visu_new(las_df)
    # boxplot_visu(clean_df, noise_df, knn)

    summary_stats(las_df_, columns)


def stats_visualization_circular(las_path, columns):
    las = pylas.read(las_path)

    las.classification = las.outliers

    las_df_ = las2df(las, columns)
    las_df = pd.melt(las_df_, id_vars='classification', value_vars=columns)
    las_df['value'] = np.deg2rad(las_df['value'])
    #for col in columns:
    #    data = las_df_[col].values
    #    data_rad = np.deg2rad(data)
    #    print(data)
    #    print(data_rad)
    #    pycircular_test(data)

    # circular_histo(las_df)
    # circular_boxplot_new(las_df)
    circular_summary_stats(las_df_, columns)


def las2df(las, columns):
    las_df = pd.DataFrame()
    for col in columns:
        las_df[col] = las[col]
    las_df['classification'] = np.where(las.classification == 0, 'inliers', 'outliers')

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


def histogram_visu_new(df):
    sns.set_theme()
    
    print(df)
    sns.color_palette()

    g = sns.FacetGrid(df, col='variable', sharey=False, sharex=False, height=6)
    g.map_dataframe(sns.histplot, x='value', hue="classification", stat='density', common_norm=False, kde=True, element="step")
    # sns.histplot(data=las_df, x=col, hue='classification', stat='density', common_norm=False, kde=True, element="step", alpha=0.5)
    # g=sns.displot(df, x='value', col='variable', hue='classification', kde=True, stat='density', common_norm=False, common_bins=False, element='step', facet_kws={'sharex':False, 'sharey': False})
    g.axes[0,0].set(xscale="log")
    g.add_legend()
    return df


def circular_histo(df):
    sns.set_theme()

    sns.color_palette(['blue', 'orange'])

    g = sns.FacetGrid(df, col='variable', palette= {'inliers': 'blue', 'outliers': 'orange'}, subplot_kws=dict(projection='polar', ), sharex=False, sharey=False, despine=False, height=5)
    g.map_dataframe(sns.histplot, x='value', hue='classification', stat='density', common_norm=False, kde=True, element="step", color={'inliers': 'blue', 'outliers': 'orange'})

    g.add_legend()
    # plt.gca().set_theta_zero_location("N")
    # plt.gca().set_theta_direction(-1)
    for ax in g.axes[0]:
        rmax = ax.get_rmax()
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rorigin(-rmax*0.5)
        #ax.set_ylim(rmax, 0.0)


def pairplot_visu(df):
    sns.set_theme(style="ticks")

    sns.pairplot(df, hue="classification", kind='kde', stat='density')


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
                sns.histplot(data=clean_df, x=col, stat='density', common_norm=False, kde=True, element="step", color='blue', alpha=0.5, label='inliers', ax=axes[i,j])
                sns.histplot(data=noise_df, x=col, stat='density', common_norm=False, kde=True, element="step", color='orange', alpha=0.5, label='outliers', ax=axes[i,j])
                
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


def boxplot_visu_new(df):

    g = sns.FacetGrid(df, palette= {'inliers': 'blue', 'outliers': 'orange'}, col='variable', sharey=False, sharex=False, height=6)
    g.map_dataframe(sns.boxplot, x='classification', y='value', palette= {'inliers': sns.color_palette()[0], 'outliers': sns.color_palette()[1]})
    g.axes[0,0].set(yscale="log")


def circular_boxplot_new(df):

    #df['value'] = np.deg2rad(df['value'])
    g = sns.FacetGrid(df, col='variable', subplot_kws=dict(projection='polar'), sharex=False, sharey=False, despine=False, height=5)
    g.map_dataframe(sns.boxplot, x='value', hue='classification')

    # g.add_legend()
    # plt.gca().set_theta_zero_location("N")
    # plt.gca().set_theta_direction(-1)
    for ax in g.axes[0]:
        rmax = ax.get_rmax()
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rorigin(-rmax*0.3)

def boxplot_test():
    # Generate some sample data
    np.random.seed(42)
    data = np.random.randn(100, 4)

    # Create a pandas DataFrame for easy handling of the data
    df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])

    # Convert Cartesian coordinates to polar coordinates
    theta = np.linspace(0, 2*np.pi, 100, endpoint=False)

    # Create a polar plot
    fig, ax = plt.subplots(subplot_kw=dict(polar=True))

    print(df.values.T)
    print(theta)

    # Plot boxplots in polar projection
    bp = ax.boxplot(df.values.T, positions=theta, widths=0.2, patch_artist=True)

    # Customize the boxplot colors if needed
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Add labels
    ax.set_xticks(theta)
    ax.set_xticklabels(df.columns)

    # Display the plot
    plt.show()

def boxplot_visu(clean_df, noise_df, knn):
    columns = clean_df.columns
    noise_df['Group'] = 'outliers'
    clean_df['Group'] = 'inliers'
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


def summary_stats(las_df, attributes):
    cols = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'mode', 'mad', 'variance', 'range', 'rms', 'skewness', 'kurtosis']
    idx = ['inliers', 'outliers']
    stats_df = pd.DataFrame(columns=idx, index=cols)
    # print(stats_df)
    data_id1 = las_df[las_df['classification']==idx[0]].drop('classification', axis=1)
    data_id2 = las_df[las_df['classification']==idx[1]].drop('classification', axis=1)
    stats1 = data_id1.describe()
    stats2 = data_id2.describe()
    stats1_new = add_stats(data_id1, stats1)
    print(stats1_new)
    stats_merge = pd.concat([stats1, stats2],axis=1, keys=idx).swaplevel(axis=1)[attributes]

    # print(stats2)
    # print(stats_merge)

        # stats2 = stats.describe(data_id)
        # print(stats2)
        #stats1['median'] = np.median(band)
        #stats1['mode'] = scipy.stats.mode(band).mode[0]
        #stats1['mad'] = np.median(abs(band-np.median(band)))
        #stats1['variance'] = stats2.variance
        #stats1['range'] = np.ptp(band)
        #stats1['rms'] = np.sqrt(np.mean(band**2))
        #stats1['skewness'] = stats2.skewness
        #stats1['kurtosis'] = stats2.kurtosis
    # return stats1

def circular_summary_stats(las_df, attributes):
    cols = ['circmean', 'circstd', 'circvar']
    idx = ['inliers', 'outliers']
    stats_in = pd.DataFrame(columns=attributes, index=cols)
    stats_out = pd.DataFrame(columns=attributes, index=cols)
    data_in = las_df[las_df['classification']==idx[0]].drop('classification', axis=1)
    data_out = las_df[las_df['classification']==idx[1]].drop('classification', axis=1)
    add_circ_stats(data_in, stats_in)
    add_circ_stats(data_out, stats_out)
    stats_merge = pd.concat([stats_in, stats_out], axis=1, keys=idx).swaplevel(axis=1)[attributes]
    print(stats_merge)




def add_circ_stats(data, stats_table):
    for col in data.columns:
        data_rad = np.deg2rad(data[col])
    
        stats_table.at['circmean', col] = np.rad2deg(circstats.circmean(data_rad))
        print(np.rad2deg(periodic_mean_std(data_rad)))
        stats_table.at['circstd', col] = circstats.circstd(data_rad)
        stats_table.at['circvar', col] = circstats.circvar(data_rad)

def add_stats(data, stats_table):
    #stats_T = stats_table.transpose()            
    for col in data.columns:
        stats_table.at['mode', col] = mode(data[col])
        stats_table.at['mad', col] = stats.median_abs_deviation(data[col], scale=1)
    
    #print(stats_T)
    # print(data.apply(stats.mode()))
    # stats_t['mode'] = data.apply(stats.mode()).mode[0]
    #stats_table = stats_T.transpose()
    return stats_table

if __name__ == '__main__':
    main()
