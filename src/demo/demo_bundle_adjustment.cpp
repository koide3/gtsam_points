#include <thread>
#include <vector>
#include <fstream>
#include <iostream>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include <gtsam_ext/util/read_points.hpp>
#include <gtsam_ext/types/frame_cpu.hpp>
#include <gtsam_ext/factors/bundle_adjustment_factor_evm.hpp>
#include <gtsam_ext/factors/bundle_adjustment_factor_lsq.hpp>
#include <gtsam_ext/optimizers/levenberg_marquardt_ext.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>

class BundleAdjustmentDemo {
public:
  BundleAdjustmentDemo() {
    auto viewer = guik::LightViewer::instance();
    viewer->enable_vsync();
    viewer->append_text("Usage: Right click a map point and then click [add factor] and [optimize]");

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
      const auto edge_points = gtsam_ext::read_points(edge_path);

      std::cout << "loading " << plane_path << std::endl;
      const auto plane_points = gtsam_ext::read_points(plane_path);

      edge_frames.push_back(std::make_shared<gtsam_ext::FrameCPU>(edge_points));
      plane_frames.push_back(std::make_shared<gtsam_ext::FrameCPU>(plane_points));

      viewer->update_drawable("edge_" + std::to_string(i), std::make_shared<glk::PointCloudBuffer>(edge_points), guik::Rainbow());
      viewer->update_drawable("plane_" + std::to_string(i), std::make_shared<glk::PointCloudBuffer>(plane_points), guik::Rainbow());
    }

    update_viewer(poses_gt);

    noise_scale = 0.1f;
    center << 0.0, 0.0, 0.0;

    edge_plane = 1;
    viewer->add_drawable_filter("filter", [this](const std::string& name) {
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

      // click callback
      if (ImGui::IsMouseClicked(1)) {
        auto mouse_pos = ImGui::GetMousePos();
        float depth = viewer->pick_depth(Eigen::Vector2i(mouse_pos.x, mouse_pos.y));
        if (depth >= 0.0f && depth < 1.0f) {
          Eigen::Vector3f pt = viewer->unproject(Eigen::Vector2i(mouse_pos.x, mouse_pos.y), depth);
          center = pt.cast<double>();
          viewer->update_drawable("sphere", glk::Primitives::sphere(), guik::FlatColor(1.0f, 0.0f, 0.0f, 0.4f, Eigen::Translation3f(pt)).make_transparent());
        }
      }

      ImGui::DragFloat("noise scale", &noise_scale, 0.01f, 0.0f);

      if (ImGui::Button("add noise")) {
        graph.resize(0);
        poses.clear();
        for (int i = 0; i < 5; i++) {
          gtsam::Pose3 noise = gtsam::Pose3::Expmap(gtsam::Vector6::Random() * noise_scale);
          gtsam::Pose3 pose = poses_gt.at<gtsam::Pose3>(i) * noise;
          poses.insert(i, pose);
          graph.add(gtsam::PriorFactor<gtsam::Pose3>(i, pose, gtsam::noiseModel::Isotropic::Precision(6, 0.1)));
        }
        update_viewer(poses);
      }

      ImGui::Separator();
      std::vector<const char*> edge_plane_labels = {"EDGE", "PLANE"};
      ImGui::Combo("feature", &edge_plane, edge_plane_labels.data(), edge_plane_labels.size());

      if (ImGui::Button("add factor")) {
        add_factor();
      }

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
    gtsam_ext::BundleAdjustmentFactorBase::shared_ptr factor;
    if (edge_plane == 0) {
      factor.reset(new gtsam_ext::EdgeEVMFactor());
    } else {
      // factor.reset(new gtsam_ext::PlaneEVMFactor());
      factor.reset(new gtsam_ext::LsqBundleAdjustmentFactor());
    }

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
    }
  }

  void run_optimization() {
    gtsam_ext::LevenbergMarquardtExtParams lm_params;
    lm_params.callback = [this](const gtsam_ext::LevenbergMarquardtOptimizationStatus& status, const gtsam::Values& values) {
      guik::LightViewer::instance()->append_text(status.to_string());
      update_viewer(values);
    };
    gtsam_ext::LevenbergMarquardtOptimizerExt optimizer(graph, poses, lm_params);
    poses = optimizer.optimize();
  }

private:
  std::vector<gtsam_ext::Frame::Ptr> edge_frames;
  std::vector<gtsam_ext::Frame::Ptr> plane_frames;

  float noise_scale;

  int edge_plane;
  gtsam::Point3 center;

  gtsam::Values poses;
  gtsam::Values poses_gt;
  gtsam::NonlinearFactorGraph graph;

  std::thread optimization_thread;
};

int main(int argc, char** argv) {
  BundleAdjustmentDemo demo;
  guik::LightViewer::instance()->spin();
  return 0;
}