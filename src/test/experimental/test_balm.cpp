#include <vector>
#include <iostream>
#include <boost/format.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/factors/bundle_adjustment_factor_evm.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>

#include <glk/colormap.hpp>
#include <glk/pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>

/*
template <typename Func>
Eigen::MatrixXd numerical_hessian(const Func& f, const Eigen::VectorXd& x, double eps = 1e-6) {
  const int N = x.size();
  Eigen::MatrixXd h(N, N);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      dx[i] = eps;

      auto first = [&](const Eigen::VectorXd& dy) {
        const double y0 = f(x - dx + dy);
        const double y1 = f(x + dx + dy);
        return (y1 - y0) / (2.0 * eps);
      };

      Eigen::VectorXd dy = Eigen::VectorXd::Zero(N);
      dy[j] = eps;

      const double dx0 = first(-dy);
      const double dx1 = first(dy);

      h(i, j) = (dx1 - dx0) / (2.0 * eps);
    }
  }

  return h;
}
*/

int main(int argc, char** argv) {
  auto viewer = guik::LightViewer::instance();

  gtsam::Values values;
  std::vector<gtsam_points::PointCloudCPU::Ptr> frames;

  std::ifstream ifs("data/newer_01/graph.txt");
  for (int i = 0; i < 5; i++) {
    auto points_path = (boost::format("data/newer_01/planes_%06d.bin") % (i * 10)).str();
    auto points = gtsam_points::read_points(points_path);
    frames.push_back(gtsam_points::PointCloudCPU::Ptr(new gtsam_points::PointCloudCPU(points)));

    std::string token;
    Eigen::Vector3d trans;
    Eigen::Quaterniond quat;
    ifs >> token >> trans.x() >> trans.y() >> trans.z() >> quat.x() >> quat.y() >> quat.z() >> quat.w();
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    pose.linear() = quat.toRotationMatrix();
    pose.translation() = trans;

    gtsam::Pose3 noise = gtsam::Pose3::Expmap(gtsam::Vector6::Random() * 0.03);

    values.insert(i, gtsam::Pose3(pose.matrix()) * noise);

    viewer->update_drawable("frame_" + std::to_string(i), std::make_shared<glk::PointCloudBuffer>(points), guik::Rainbow(values.at<gtsam::Pose3>(i).matrix().cast<float>()));
  }

  gtsam::NonlinearFactorGraph graph;
  for (int i = 0; i < frames.size(); i++) {
    graph.add(gtsam::PriorFactor<gtsam::Pose3>(i, values.at<gtsam::Pose3>(i), gtsam::noiseModel::Isotropic::Precision(6, 0.1)));
  }

  Eigen::Vector3f center(0.0f, 0.0f, 0.0f);
  viewer->register_ui_callback("callback", [&] {
    if(ImGui::IsMouseClicked(1)) {
      auto mouse_pos = ImGui::GetMousePos();
      float depth = viewer->pick_depth(Eigen::Vector2i(mouse_pos.x, mouse_pos.y));
      if(depth > 0.0f) {
        center = viewer->unproject(Eigen::Vector2i(mouse_pos.x, mouse_pos.y), depth);
        viewer->update_drawable("center", glk::Primitives::sphere(), guik::FlatColor(1.0, 0.0, 0.0, 0.4, Eigen::Translation3f(center)).make_transparent());
      }
    }

    if(ImGui::Button("add factor")) {
      gtsam_points::PlaneEVMFactor::shared_ptr plane_factor(new gtsam_points::PlaneEVMFactor);

      for(int i=0; i<frames.size(); i++) {
        for(int j=0; j<frames[i]->size(); j++) {
          const Eigen::Vector3d pt = frames[i]->points[j].head<3>();
          const Eigen::Vector3d transed_pt = values.at<gtsam::Pose3>(i) * pt;
          const double dist = (transed_pt - center.cast<double>()).norm();
          if (dist > 1.0) {
            continue;
          }

          std::cout << i << ":" << transed_pt.transpose() << std::endl;
          plane_factor->add(pt, i);
        }
      }

      if(plane_factor->num_points() > 6) {
        graph.add(plane_factor);
      }
    }

    if(ImGui::Button("optimize")) {
      gtsam_points::LevenbergMarquardtExtParams lm_params;
      lm_params.callback = [&](const gtsam_points::LevenbergMarquardtOptimizationStatus& status, const gtsam::Values& values) {
        viewer->append_text(status.to_string());
        for (int i = 0; i < frames.size(); i++) {
          auto drawable = viewer->find_drawable("frame_" + std::to_string(i));
          drawable.first->add("model_matrix", values.at<gtsam::Pose3>(i).matrix().cast<float>().eval());
        }
      };

      gtsam_points::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);
      values = optimizer.optimize();
    }

  });

  viewer->spin();

  return 0;
}
