#include <thread>
#include <iostream>
#include <boost/format.hpp>

#include <gtsam_ext/util/read_points.hpp>
#include <gtsam_ext/util/normal_estimation.hpp>
#include <gtsam_ext/util/covariance_estimation.hpp>
#include <gtsam_ext/types/frame.hpp>
#include <gtsam_ext/types/frame_cpu.hpp>
#include <gtsam_ext/factors/integrated_ct_icp_factor.hpp>
#include <gtsam_ext/factors/integrated_ct_gicp_factor.hpp>
#include <gtsam_ext/factors/continuous_time_icp_factor.hpp>
#include <gtsam_ext/optimizers/levenberg_marquardt_ext.hpp>

#include <glk/pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>

class ContinuousTimeDemo {
public:
  ContinuousTimeDemo() {
    auto viewer = guik::LightViewer::instance();
    viewer->enable_vsync();

    const std::string data_path = "data/newer_06";
    for (int i = 0; i < 3; i++) {
      const std::string times_path = (boost::format("%s/times_%02d.bin") % data_path % i).str();
      const std::string raw_path = (boost::format("%s/raw_%02d.bin") % data_path % i).str();
      const std::string deskewed_path = (boost::format("%s/deskewed_%02d.bin") % data_path % i).str();

      auto times = gtsam_ext::read_times(times_path);
      auto raw_points = gtsam_ext::read_points(raw_path);
      auto deskewed_points = gtsam_ext::read_points(deskewed_path);

      if (times.empty()) {
        std::cerr << "error: failed to read " << times_path << std::endl;
        abort();
      } else if (raw_points.size() != times.size()) {
        std::cerr << "error: num of points mismatch " << raw_path << std::endl;
        abort();
      } else if (deskewed_points.size() != times.size()) {
        std::cerr << "error: num of points mismatch " << deskewed_path << std::endl;
        abort();
      }

      for (auto& pt : raw_points) {
        Eigen::Quaternionf q(0, 0, 0, 1);
        pt = q * pt;
      }

      auto raw_frame = std::make_shared<gtsam_ext::FrameCPU>(raw_points);
      raw_frame->add_times(times);
      raw_frame->add_covs(gtsam_ext::estimate_covariances(raw_frame->points, raw_frame->size()));
      raw_frames.push_back(raw_frame);

      auto deskewed_frame = std::make_shared<gtsam_ext::FrameCPU>(deskewed_points);
      deskewed_frame->add_covs(gtsam_ext::estimate_covariances(deskewed_frame->points, deskewed_frame->size()));
      deskewed_frame->add_normals(gtsam_ext::estimate_normals(deskewed_frame->points, deskewed_frame->covs, deskewed_frame->size()));
      deskewed_frames.push_back(deskewed_frame);
    }

    noise_scale = 0.0f;
    pose_noise = gtsam::Pose3::identity();

    data_id = 0;
    setup(data_id);

    viewer->register_ui_callback("callback", [this] {
      ImGui::DragFloat("noise scale", &noise_scale, 0.01f, 0.0f);
      if (ImGui::Button("add noise")) {
        pose_noise = gtsam::Pose3::Expmap(gtsam::Vector6::Random() * noise_scale);
        setup(data_id);
      }

      ImGui::Separator();
      std::vector<const char*> data_id_labels = {"DATA0", "DATA1", "DATA2"};
      if (ImGui::Combo("DataID", &data_id, data_id_labels.data(), data_id_labels.size())) {
        setup(data_id);
      }

      if (ImGui::Button("optimize")) {
        if (optimization_thread.joinable()) {
          optimization_thread.join();
        }

        optimization_thread = std::thread([this] { run_optimization(data_id); });
      }
    });
  }
  ~ContinuousTimeDemo() {
    if (optimization_thread.joinable()) {
      optimization_thread.join();
    }
  }

  void setup(int data_id) {
    auto viewer = guik::LightViewer::instance();
    viewer->update_drawable(
      "target",
      std::make_shared<glk::PointCloudBuffer>(deskewed_frames[data_id]->points, deskewed_frames[data_id]->size()),
      guik::FlatColor(0.0, 1.0, 0.0, 1.0));

    auto cloud_buffer = std::make_shared<glk::PointCloudBuffer>(raw_frames[data_id]->points, raw_frames[data_id]->size());
    std::vector<double> times(raw_frames[data_id]->times, raw_frames[data_id]->times + raw_frames[data_id]->size());
    cloud_buffer->add_intensity(glk::COLORMAP::BLUE_RED, times, 1.0 / times.back());
    viewer->update_drawable("source", cloud_buffer, guik::VertexColor(pose_noise.matrix().cast<float>()));
  }

  void run_optimization(int data_id) {
    gtsam::Values values;
    values.insert(0, pose_noise);
    values.insert(1, pose_noise);

    gtsam::NonlinearFactorGraph graph;

    auto factor = gtsam::make_shared<gtsam_ext::IntegratedCT_GICPFactor>(0, 1, deskewed_frames[data_id], raw_frames[data_id]);
    // auto factor = gtsam_ext::create_integrated_cticp_factor(0, 1, deskewed_frames[data_id], raw_frames[data_id], noise_model);
    graph.add(factor);

    gtsam_ext::LevenbergMarquardtExtParams lm_params;
    lm_params.callback = [&](const gtsam_ext::LevenbergMarquardtOptimizationStatus& status, const gtsam::Values& values) {
      auto viewer = guik::LightViewer::instance();
      viewer->append_text(status.to_string());

      std::vector<double> times(raw_frames[data_id]->times, raw_frames[data_id]->times + raw_frames[data_id]->size());
      auto points = factor->deskewed_source_points(values);

      std::vector<Eigen::Isometry3f, Eigen::aligned_allocator<Eigen::Isometry3f>> poses;
      for (int i = 0; i < 4; i++) {
        gtsam::Pose3 pose = gtsam_ext::interpolate_pose(values.at<gtsam::Pose3>(0), values.at<gtsam::Pose3>(1), static_cast<double>(i) / 3);
        poses.push_back(Eigen::Isometry3f(pose.matrix().cast<float>()));
      }

      viewer->invoke([=] {
        auto cloud_buffer = std::make_shared<glk::PointCloudBuffer>(points);
        cloud_buffer->add_intensity(glk::COLORMAP::BLUE_RED, times, 1.0 / times.back());
        viewer->update_drawable("source", cloud_buffer, guik::VertexColor());

        for (int i = 0; i < poses.size(); i++) {
          viewer->update_drawable("coord_" + std::to_string(i), glk::Primitives::coordinate_system(), guik::VertexColor(poses[i] * Eigen::UniformScaling<float>(1.0f)));
        }
      });
    };

    gtsam_ext::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);
    optimizer.optimize();
  }

private:
  std::vector<gtsam_ext::Frame::Ptr> raw_frames;
  std::vector<gtsam_ext::Frame::Ptr> deskewed_frames;

  float noise_scale;
  gtsam::Pose3 pose_noise;

  int data_id;
  std::thread optimization_thread;
};

int main(int argc, char** argv) {
  ContinuousTimeDemo demo;
  guik::LightViewer::instance()->spin();
  return 0;
}