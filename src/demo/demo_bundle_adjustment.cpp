#include <thread>
#include <vector>
#include <fstream>
#include <iostream>
#include <boost/format.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/factors/bundle_adjustment_factor_evm.hpp>
#include <gtsam_points/factors/bundle_adjustment_factor_lsq.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>

class BundleAdjustmentDemo {
public:
  BundleAdjustmentDemo() {
    auto viewer = guik::LightViewer::instance();
    viewer->enable_vsync();
    viewer->append_text("Usage: Right click a map point and then click [add factor] and [optimize]");

    // Read test data
    const std::string data_path = "data/newer_01";
    std::ifstream ifs(data_path + "/graph.txt");
    if (!ifs) {
      std::cerr << "error: failed to open " << data_path + "/graph.txt" << std::endl;
      abort();
    }

    for (int i = 0; i < 5; i++) {
      std::string token;
      gtsam::Vector3 trans;
      gtsam::Quaternion quat;

      ifs >> token >> trans.x() >> trans.y() >> trans.z() >> quat.x() >> quat.y() >> quat.z() >> quat.w();
      poses_gt.insert(i, gtsam::Pose3(gtsam::Rot3(quat), trans));

      const std::string edge_path = (boost::format("%s/edges_%06d.bin") % data_path % (i * 10)).str();
      const std::string plane_path = (boost::format("%s/planes_%06d.bin") % data_path % (i * 10)).str();

      std::cout << "loading " << edge_path << std::endl;
      const auto edge_points = gtsam_points::read_points(edge_path);

      std::cout << "loading " << plane_path << std::endl;
      const auto plane_points = gtsam_points::read_points(plane_path);

      edge_frames.push_back(std::make_shared<gtsam_points::PointCloudCPU>(edge_points));
      plane_frames.push_back(std::make_shared<gtsam_points::PointCloudCPU>(plane_points));

      viewer->update_drawable("edge_" + std::to_string(i), std::make_shared<glk::PointCloudBuffer>(edge_points), guik::Rainbow());
      viewer->update_drawable("plane_" + std::to_string(i), std::make_shared<glk::PointCloudBuffer>(plane_points), guik::Rainbow());
    }

    update_viewer(poses_gt);

    pose_noise_scale = 0.1f;
    center << 0.0, 0.0, 0.0;

    factor_type = 0;
    factor_types.push_back("EVM");
    factor_types.push_back("LSQ");

    edge_plane = 1;
    viewer->register_drawable_filter("filter", [this](const std::string& name) {
      if (edge_plane == 0 && name.find("plane") != std::string::npos) {
        return false;
      }
      if (edge_plane == 1 && name.find("edge") != std::string::npos) {
        return false;
      }
      return true;
    });

    viewer->register_ui_callback("callback", [this] {
      auto viewer = guik::LightViewer::instance();

      if (ImGui::IsMouseClicked(1)) {
        // Calculate the 3D coordinates of the right clicked position
        auto mouse_pos = ImGui::GetMousePos();
        float depth = viewer->pick_depth(Eigen::Vector2i(mouse_pos.x, mouse_pos.y));
        if (depth >= 0.0f && depth < 1.0f) {
          Eigen::Vector3f pt = viewer->unproject(Eigen::Vector2i(mouse_pos.x, mouse_pos.y), depth);
          center = pt.cast<double>();
          viewer->update_drawable("sphere", glk::Primitives::sphere(), guik::FlatColor(1.0f, 0.0f, 0.0f, 0.4f, Eigen::Translation3f(pt)).make_transparent());
        }
      }

      ImGui::DragFloat("pose noise scale", &pose_noise_scale, 0.01f, 0.0f);

      if (ImGui::Button("initialize graph")) {
        graph.resize(0);
        poses.clear();
        for (int i = 0; i < 5; i++) {
          gtsam::Pose3 noise = gtsam::Pose3::Expmap(gtsam::Vector6::Random() * pose_noise_scale);
          gtsam::Pose3 pose = poses_gt.at<gtsam::Pose3>(i) * noise;
          poses.insert(i, pose);

          // Add weak priors to avoid degenerated system
          graph.add(gtsam::PriorFactor<gtsam::Pose3>(i, pose, gtsam::noiseModel::Isotropic::Precision(6, 0.1)));
        }
        update_viewer(poses);
      }

      ImGui::Separator();
      ImGui::Combo("factor type", &factor_type, factor_types.data(), factor_types.size());

      if (factor_types[factor_type] == std::string("EVM")) {
        std::vector<const char*> edge_plane_labels = {"EDGE", "PLANE"};
        ImGui::Combo("feature", &edge_plane, edge_plane_labels.data(), edge_plane_labels.size());
      } else {
        // Force set the selection mode to the plane mode, because LSQ BA only supports plane features
        edge_plane = 1;
      }

      // Add a BA factor with points around the selected point
      if (ImGui::Button("add factor")) {
        add_factor();
      }

      // Run optimization
      if (ImGui::Button("optimize")) {
        if (optimization_thread.joinable()) {
          optimization_thread.join();
        }
        optimization_thread = std::thread([this] { run_optimization(); });
      }
    });
  }

  ~BundleAdjustmentDemo() {
    if (optimization_thread.joinable()) {
      optimization_thread.join();
    }
  }

  void update_viewer(const gtsam::Values& values) {
    guik::LightViewer::instance()->invoke([=] {
      auto viewer = guik::LightViewer::instance();
      for (int i = 0; i < 5; i++) {
        gtsam::Pose3 pose = values.at<gtsam::Pose3>(i);
        auto edge_drawable = viewer->find_drawable("edge_" + std::to_string(i));
        auto plane_drawable = viewer->find_drawable("plane_" + std::to_string(i));
        edge_drawable.first->add("model_matrix", pose.matrix().cast<float>().eval());
        plane_drawable.first->add("model_matrix", pose.matrix().cast<float>().eval());

        viewer->update_drawable("coord_" + std::to_string(i), glk::Primitives::coordinate_system(), guik::VertexColor(pose.matrix().cast<float>()));
      }
    });
  }

  void add_factor() {
    const auto& frames = edge_plane == 0 ? edge_frames : plane_frames;

    gtsam_points::BundleAdjustmentFactorBase::shared_ptr factor;
    if (edge_plane == 0) {
      factor.reset(new gtsam_points::EdgeEVMFactor());
    } else {
      if (factor_types[factor_type] == std::string("EVM")) {
        factor.reset(new gtsam_points::PlaneEVMFactor());
      } else if (factor_types[factor_type] == std::string("LSQ")) {
        factor.reset(new gtsam_points::LsqBundleAdjustmentFactor());
      }
    }

    // Find points around the selected point and add them to the factor
    for (int i = 0; i < frames.size(); i++) {
      const gtsam::Pose3 pose = poses.at<gtsam::Pose3>(i);
      for (int j = 0; j < frames[i]->size(); j++) {
        const Eigen::Vector3d pt = frames[i]->points[j].head<3>();
        const Eigen::Vector3d transed_pt = pose * pt;
        if ((transed_pt - center).norm() > 1.0) {
          continue;
        }

        factor->add(pt, i);
      }
    }

    if (factor->num_points() > 6) {
      const std::string feature_type = edge_plane == 0 ? "edge" : "plane";
      guik::LightViewer::instance()->append_text(feature_type + " factor is created with " + std::to_string(factor->num_points()) + " points");
      graph.add(factor);
    } else {
      guik::LightViewer::instance()->append_text("ignoring the factor with too few points: " + std::to_string(factor->num_points()));
    }
  }

  void run_optimization() {
    gtsam_points::LevenbergMarquardtExtParams lm_params;
    lm_params.callback = [this](const gtsam_points::LevenbergMarquardtOptimizationStatus& status, const gtsam::Values& values) {
      guik::LightViewer::instance()->append_text(status.to_string());
      update_viewer(values);
    };
    gtsam_points::LevenbergMarquardtOptimizerExt optimizer(graph, poses, lm_params);
    poses = optimizer.optimize();
  }

private:
  std::vector<gtsam_points::PointCloud::Ptr> edge_frames;   // Frames containing edge points
  std::vector<gtsam_points::PointCloud::Ptr> plane_frames;  // Frames containing plane points

  float pose_noise_scale;

  std::vector<const char*> factor_types;
  int factor_type;

  int edge_plane;        // 0 = selecting edge points, 1 = plane points
  gtsam::Point3 center;  // Center point for points extraction

  gtsam::Values poses;     // Current estimate
  gtsam::Values poses_gt;  // True poses
  gtsam::NonlinearFactorGraph graph;

  std::thread optimization_thread;
};

int main(int argc, char** argv) {
  BundleAdjustmentDemo demo;
  guik::LightViewer::instance()->spin();
  return 0;
}