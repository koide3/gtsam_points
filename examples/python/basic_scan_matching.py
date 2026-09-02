#!/usr/bin/env python3
"""
This example demonstrates how to perform simple frame-to-frame scan matching with gtsam_points.
It is a python version of src/example/basic_scan_matching.cpp and shows that gtsam_points factors
can be seamlessly used with the GTSAM python bindings.
"""

import numpy
import gtsam
import gtsam_points


def main():
  # Read target and source point clouds
  target_points = gtsam_points.read_points('data/kitti_00/000000.bin')
  source_points = gtsam_points.read_points('data/kitti_00/000001.bin')

  # Create gtsam_points.PointCloudCPU instances that hold point data
  target_frame = gtsam_points.PointCloudCPU(target_points)
  source_frame = gtsam_points.PointCloudCPU(source_points)

  # Create GTSAM values and graph
  values = gtsam.Values()
  values.insert(0, gtsam.Pose3())  # Target pose initial guess
  values.insert(1, gtsam.Pose3())  # Source pose initial guess

  graph = gtsam.NonlinearFactorGraph()

  # Fix the target pose at the origin
  graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(), gtsam.noiseModel.Isotropic.Precision(6, 1e6)))

  # Create an ICP factor between target and source poses
  icp_factor = gtsam_points.IntegratedICPFactor(0, 1, target_frame, source_frame)
  icp_factor.set_max_correspondence_distance(5.0)
  icp_factor.set_num_threads(4)
  graph.add(icp_factor)

  # Create LM optimizer
  lm_params = gtsam_points.LevenbergMarquardtExtParams()
  lm_params.set_verbose()
  optimizer = gtsam_points.LevenbergMarquardtOptimizerExt(graph, values, lm_params)

  # Optimize
  values = optimizer.optimize()

  T_target_source = values.atPose3(1).matrix()
  print('T_target_source =')
  print(T_target_source)

  # Visualization (requires iridescence python bindings: https://github.com/koide3/iridescence)
  try:
    from pyridescence import guik, glk
  except ImportError:
    print('pyridescence is not available for visualization')
    return

  viewer = guik.viewer()
  viewer.update_drawable('target', glk.create_pointcloud_buffer(target_points), guik.FlatRed())
  viewer.update_drawable('source', glk.create_pointcloud_buffer(source_points), guik.FlatGreen())
  viewer.update_drawable('aligned', glk.create_pointcloud_buffer(source_points), guik.FlatBlue(T_target_source.astype(numpy.float32)))
  viewer.spin()


if __name__ == '__main__':
  main()
