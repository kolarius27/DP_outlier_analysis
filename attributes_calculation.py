#!/bin/python
import pylas
import pdal
import pandas as pd
import numpy as np
import geopandas as gpd
import math as m
import point_cloud_handeling as pc_h
from scipy.spatial import cKDTree
from scipy.stats import entropy
from scipy.linalg import svd
from scipy.stats.mstats import gmean
from sklearn.decomposition import PCA
from multiprocessing import cpu_count
import json
from shapely import Polygon, Point, LineString


def main():
    test_boulder = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder4_attributes/boulder4_new_normal_0.45_50.las'
    pdal_boulder = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder4_opals_eigen.las'
    trajectory = r'E:/NATUR_CUNI/_DP/data/Trajectory/HEK_trajectory.csv'

    # boulder4 = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder4_attributes/boulder4_new_mrecsor_0.3_0.7_normal_0.45_50.las'
    boulder4 = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder4_attributes/boulder4_normfix_0.08_0.4_100.las'
    #calculate_factors(las_path=test_boulder, trj_path=trajectory)
    # calculate_features_knn(pdal_boulder, 25)
    # pc_eigen(pdal_boulder, 5)
    # pc_lof(pdal_boulder, 25)
    # pc_cov_features(pdal_boulder, 25, False)
    
    # boulder2_noise = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder4_new/boulder4_noise.las'
    # boulder2 = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder2_merge_normal45_factors_eig_25.las'
    # boulder4 = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/adjusted/boulder4_new_normal45_factors_eig_25.las'
    #calculate_factors(boulder2, trajectory)
    # calculate_factors(boulder4, trajectory)

    # calculate_features_knn(boulder2, 25)
    # calculate_features_knn(boulder4, 25)
    # pc_cov_features(boulder2, knn=25, optimized=False)
    # pc_cov_features(boulder4, knn=25, optimized=False)

    calculate_factors(boulder4, trajectory)


def calculate_factors(las_path, trj_path):
    # load point cloud and trajectory
    las = pylas.read(las_path)
    trj = pd.read_csv(trj_path)


    # pick scanner position
    gdf_las = pick_scanner_position(las, trj)

    # calculate angles and azimuths
    pc_angles(las, gdf_las)
    pc_azimuths(las, gdf_las)

    # save las
    las.write(las_path.replace('.las', '_factors.las'))

def calculate_features_knn(las_path, knn):
    las = pylas.read(las_path)   

    points_xyz = pc_h.get_xyz(las)

    points_knn = pick_neighborhoods(points_xyz, knn)

    eig = pc_eigenvalues4(points_knn)
    # eig_t = eig.T

    eig = np.array([eig[2], eig[1], eig[0]])
    pc_h.add_dimension(las, 'eig1', 'f8', '_', eig[0])
    pc_h.add_dimension(las, 'eig2', 'f8', '_', eig[1])
    pc_h.add_dimension(las, 'eig3', 'f8', '_', eig[2])

    pc_geom_features(las, eig, knn)

    print(list(las.point_format.dimension_names))

    las.write(las_path.replace('.las', '_eig_{}.las'.format(knn)))



'''
¨functions used for calculation of factors (scan angle, incidence angle...)
'''

def pick_scanner_position(las, trj):
    print(list(las.point_format.dimension_names))
    # prepare values od gdf_las
    dict_las = {'gps_time': las.gps_time, 
                'intensity': las.intensity,
                'NormalX': las.Nx,
                'NormalY': las.Ny,
                'NormalZ': las.Nz}

    # create GeoDataframes of las and trj
    gdf_las = gpd.GeoDataFrame(dict_las, geometry=gpd.points_from_xy(las.x, las.y, las.z))
    gdf_trj = gpd.GeoDataFrame(trj, geometry=gpd.points_from_xy(trj['Easting[m]'], trj['Northing[m]'], trj['Height[m]']))
    
    print(gdf_las)
    # add index 
    gdf_las['las_index'] = gdf_las.index

    # connect point and scanner position, create table
    merge_gdf = pd.merge_asof(gdf_las.sort_values('gps_time'), gdf_trj, left_on='gps_time', right_on='Time[s]', direction='backward')
    # print(merge_gdf)


    # calculate scan_distance
    differences = np.column_stack([
    merge_gdf['geometry_x'].geometry.x.values - merge_gdf['geometry_y'].geometry.x.values,
    merge_gdf['geometry_x'].geometry.y.values - merge_gdf['geometry_y'].geometry.y.values,
    merge_gdf['geometry_x'].geometry.z.values - merge_gdf['geometry_y'].geometry.z.values
    ])

    # Calculate the Euclidean distances using np.linalg.norm
    merge_gdf['scan_distance'] = np.linalg.norm(differences, axis=1)

    # sort by original index
    merge_gdf.index = merge_gdf['las_index']
    sort_gdf = merge_gdf.sort_index()
    # print('geometry_x ', sort_gdf['geometry_x'])

    # save scan_distance into las
    pc_h.add_dimension(las, 'scan_distance', 'f8', 'Distance from scanner', sort_gdf['scan_distance'])
    return sort_gdf


def pc_angles(las, gdf_las):
    # prepare variables
    roll = gdf_las['Roll[deg]'].apply(m.radians)
    pitch = gdf_las['Pitch[deg]'].apply(m.radians)
    yaw = gdf_las['Yaw[deg]'].apply(m.radians)
    rot_angles = np.array([roll, pitch, yaw]).transpose()
    normals = np.array([gdf_las['NormalX'], gdf_las['NormalY'], gdf_las['NormalZ']]).transpose()


    # calculate scan angle
    scan_angle, v2 = pc_h.compute_scan_angle_new(gdf_las['geometry_x'], gdf_las['geometry_y'], rot_angles)

    # calculate incidence angle 
    incidence_angle = pc_h.compute_incidence_angle_new(-v2, normals)

    # calculate slope
    slope = np.rad2deg(np.arccos(gdf_las['NormalZ']))

    # calculate angle difference
    angle_sum = (scan_angle + incidence_angle)/2

    # save everything to gdf
    gdf_las['scan_angle'] = scan_angle
    gdf_las['incidence_angle'] = incidence_angle
    gdf_las['slope2'] = slope
    gdf_las['angle_sum'] = angle_sum

    # add dimensions
    pc_h.add_dimension(las, 'scan_angle', 'f8', 'Scanning angle', scan_angle)
    pc_h.add_dimension(las, 'incidence_angle', 'f8', 'Incidence angle of given point', incidence_angle)
    pc_h.add_dimension(las, 'slope', 'f8', 'Max dip', slope)
    pc_h.add_dimension(las, 'angle_mean', 'f8', 'Mean of scan and incidence angle', angle_sum)


def pc_azimuths(las, gdf_las):
    # calculate scan azimuth
    scan_azimuth = pc_h.pc_scan_azimuth(gdf_las['geometry_x'], gdf_las['geometry_y'])

    # calculate local azimuth
    local_azimuth = pc_h.pc_local_azimuth(gdf_las['NormalY'], gdf_las['NormalX'])

    # calculate azimuth difference
    azimuth_deviation = abs((scan_azimuth-local_azimuth) % 360. - 180.)

    # save everything to gdf_las
    gdf_las['traj_azimuth'] = gdf_las['Yaw[deg]'] 
    gdf_las['scan_azimuth'] = scan_azimuth
    gdf_las['local_azimuth'] = local_azimuth
    gdf_las['azimuth_deviation'] = azimuth_deviation

    # add dimensions
    pc_h.add_dimension(las, 'traj_azimuth', 'f8', 'Azimuth of trajectory', gdf_las['Yaw[deg]'])
    pc_h.add_dimension(las, 'scan_azimuth', 'f8', 'Azimuth between point/scanner', gdf_las['scan_azimuth'])
    pc_h.add_dimension(las, 'local_azimuth', 'f8', 'Local azimuth od surface', gdf_las['local_azimuth'])
    pc_h.add_dimension(las, 'azimuth_deviation', 'f8', 'deviation of azimuths', gdf_las['azimuth_deviation'])
'''
functions to calculate attributes (eigenentropy...)
'''
def pick_neighborhoods(points_xyz, knn):
    kdtree = cKDTree(points_xyz)
    knn_indices = kdtree.query(points_xyz, knn+1, workers=cpu_count()-1)[1]
    return points_xyz[knn_indices]


def pick_neighborhoods_by_radius(points_xyz, knn):
    kdtree = cKDTree(points_xyz)
    knn_indices = kdtree.query(points_xyz, knn, workers=cpu_count()-1)[1]
    return points_xyz[knn_indices]


def pc_eigenvalues(points_knn):
    scatter_matrices = np.matmul(points_knn.transpose(0, 2, 1), points_knn)
    return np.linalg.eigvalsh(scatter_matrices)


def pc_eigenvalues2(points_knn):
    # Compute the covariance matrices for all neighborhoods
    covariance_matrices = np.array([np.cov(points.T) for points in points_knn])

    # Compute the eigenvalues for all covariance matrices
    _, eigenvalues, _ = np.linalg.svd(covariance_matrices, full_matrices=False)

    return eigenvalues

def pc_eigenvalues3(points_knn):
    pca = PCA()
    eigenvalues = np.array([PCA().fit(neighbours).explained_variance_ for neighbours in points_knn])
    return eigenvalues.T

def compute_pca(neighbors):
    pca = PCA()
    pca.fit(neighbors)
    return pca.explained_variance_

def compute_eigenvalues(neighbors):
    return np.sort(np.linalg.eigvals(np.cov(neighbors.T)))


def pc_eigenvalues4(neighbors):
    return np.array(list(map(compute_eigenvalues, neighbors))).T


def pc_geom_features(las, eig, knn):
    print(len((eig[0]-eig[1])/eig[0]))
    print(len((eig[1]-eig[2])/eig[0]))
    print(len(np.cbrt(eig[0]*eig[1]*eig[2])))
    print(-eig[0]*np.log(eig[0])-eig[1]*np.log(eig[1])-eig[2]*np.log(eig[2]))
    df = pd.DataFrame({
        'Linearity': (eig[0]-eig[1])/eig[0],
        'Planarity': (eig[1]-eig[2])/eig[0],
        'Scattering': eig[2]/eig[0],
        'Omnivariance': np.cbrt(eig[0]*eig[1]*eig[2]),
        'Anisotropy': (eig[0]-eig[2])/eig[0],
        'Eigenentropy': entropy(eig),
        'Sum_of_eig': eig[0]+eig[1]+eig[2],
        'Change_of_curvature': eig[2]/(eig[0]+eig[1]+eig[2])
    })

    for col in df.columns:
        print(col)
        print()
        pc_h.add_dimension(las, '{}_{}'.format(col, knn), 'f8', '_', df[col])




def pc_eigen(las_path, knn):
    print('start')
    output_path = las_path.replace('.las', '_eigen.las')
    pipeline = pdal.Pipeline(json.dumps({
        "pipeline": [
            {
                "type": "readers.las",
                "filename": las_path
            }
            ,
            {
                "type": "filters.eigenvalues",
                "knn": knn
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

def pc_lof(las_path, knn):
    print('start')
    output_path = las_path.replace('.las', '_lof.las')
    pipeline = pdal.Pipeline(json.dumps({
        "pipeline": [
            {
                "type": "readers.las",
                "filename": las_path
            }
            ,
            {
                "type": "filters.lof",
                "minpts": knn
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

def pc_cov_features(las_path, knn, optimized):
    print('start')
    output_path = las_path.replace('.las', '_covf_{}.las'.format(knn))
    pipeline = pdal.Pipeline(json.dumps({
        "pipeline": [
            {
                "type": "readers.las",
                "filename": las_path
            }
            ,
            {
                "type":"filters.optimalneighborhood",
                "min_k": 5,
                "max_k": 100
            },
            {
                "type": "filters.covariancefeatures",
                "knn": knn,
                "threads": 7,
                "optimized": optimized,
                "feature_set": "all"
            },
            {
                "type": "filters.lof",
                "minpts": knn
            },
            {
                "type":"filters.reciprocity",
                "knn": knn
            },
            {
                "type": "writers.las",
                "compression": "laszip",
                "filename": output_path,
                "extra_dims": "all",
                "forward": "all"
            }
        ]
    }))

    pipeline.execute()

    print("done")

if __name__ == '__main__':
    main()