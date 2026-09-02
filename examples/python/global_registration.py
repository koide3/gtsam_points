#!/usr/bin/env python3
"""
This example demonstrates global registration (FPFH + RANSAC / GNC) followed by fine
registration with a GICP factor.
"""

import numpy
import gtsam
import gtsam_points


def preprocess(points):
  """Downsample points and estimate normals and covariances."""
  frame = gtsam_points.PointCloudCPU(points)
  frame = gtsam_points.voxelgrid_sampling(frame, 0.5, num_threads=4)
  frame.add_normals(gtsam_points.estimate_normals(frame, k_neighbors=10, num_threads=4))
  frame.add_covs(gtsam_points.estimate_covariances(frame, k_neighbors=10, num_threads=4))
  return frame


def main():
  target = preprocess(gtsam_points.read_points('data/kitti_00/000000.bin'))
  source = preprocess(gtsam_points.read_points('data/kitti_00/000001.bin'))

  # Extract FPFH features
  fpfh_params = gtsam_points.FPFHEstimationParams()
  fpfh_params.search_radius = 5.0
  fpfh_params.num_threads = 4
  target_features = gtsam_points.estimate_fpfh(target, params=fpfh_params)
  source_features = gtsam_points.estimate_fpfh(source, params=fpfh_params)

  # Global registration with RANSAC (alternatively, use estimate_pose_gnc with GNCParams)
  ransac_params = gtsam_points.RANSACParams()
  ransac_params.num_threads = 4
  result = gtsam_points.estimate_pose_ransac(target, source, target_features, source_features, params=ransac_params)
  print(f'RANSAC inlier_rate: {result.inlier_rate:.3f}')
  print('T_target_source (initial estimate) =')
  print(result.T_target_source)

  # Fine registration with a GICP factor starting from the RANSAC estimate
  graph = gtsam.NonlinearFactorGraph()
  gicp_factor = gtsam_points.IntegratedGICPFactor(gtsam.Pose3(), 1, target, source)
  gicp_factor.set_num_threads(4)
  graph.add(gicp_factor)

  values = gtsam.Values()
  values.insert(1, gtsam.Pose3(result.T_target_source))

  optimizer = gtsam_points.LevenbergMarquardtOptimizerExt(graph, values)
  values = optimizer.optimize()

  print('T_target_source (refined) =')
  print(values.atPose3(1).matrix())


if __name__ == '__main__':
  main()
