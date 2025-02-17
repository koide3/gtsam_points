#include <thread>
#include <iostream>
#include <boost/format.hpp>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam_points/util/read_points.hpp>
#include <gtsam_points/features/normal_estimation.hpp>
#include <gtsam_points/features/covariance_estimation.hpp>
#include <gtsam_points/types/point_cloud.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/factors/integrated_ct_icp_factor.hpp>
#include <gtsam_points/factors/integrated_ct_gicp_factor.hpp>
#include <gtsam_points/factors/experimental/continuous_time_icp_factor.hpp>
#include <gtsam_points/optimizers/levenberg_marquardt_ext.hpp>

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

      auto times = gtsam_points::read_times(times_path);
      auto raw_points = gtsam_points::read_points(raw_path);
      auto deskewed_points = gtsam_points::read_points(deskewed_path);

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

      // Source frames
      auto raw_frame = std::make_shared<gtsam_points::PointCloudCPU>(raw_points);
      raw_frame->add_times(times);
      raw_frame->add_covs(gtsam_points::estimate_covariances(raw_frame->points, raw_frame->size()));
      raw_frames.push_back(raw_frame);

      // Target frames
      auto deskewed_frame = std::make_shared<gtsam_points::PointCloudCPU>(deskewed_points);
      deskewed_frame->add_covs(gtsam_points::estimate_covariances(deskewed_frame->points, deskewed_frame->size()));
      deskewed_frame->add_normals(gtsam_points::estimate_normals(deskewed_frame->points, deskewed_frame->covs, deskewed_frame->size()));
      deskewed_frames.push_back(deskewed_frame);
    }

    pose_noise_scale = 0.0f;
    pose_noise = gtsam::Pose3::Identity();

    factor_types = std::vector<const char*>{"CT-ICP", "CT-GICP", "CT-ICP-EXPR"};
    factor_type = 0;

    enable_pose_constraint = true;
    rot_noise_scale = 10.0f;
    trans_noise_scale = 1000.0f;
    max_correspondence_distance = 1.0f;

    data_id = 0;
    setup(data_id);

    viewer->register_ui_callback("callback", [this] {
      ImGui::DragFloat("noise scale", &pose_noise_scale, 0.01f, 0.0f);
      if (ImGui::Button("add noise")) {
        pose_noise = gtsam::Pose3::Expmap(gtsam::Vector6::Random() * pose_noise_scale);
        setup(data_id);
      }

      ImGui::Separator();
      std::vector<const char*> data_id_labels = {"DATA0", "DATA1", "DATA2"};
      if (ImGui::Combo("DataID", &data_id, data_id_labels.data(), data_id_labels.size())) {
        setup(data_id);
      }

      // Configurations
      ImGui::Combo("factor type", &factor_type, factor_types.data(), factor_types.size());
      ImGui::Checkbox("enable pose constraint", &enable_pose_constraint);
      ImGui::DragFloat("pose rot noise scale", &rot_noise_scale, 0.1f, 0.0f);
      ImGui::DragFloat("pose trans noise scale", &trans_noise_scale, 10.0f, 0.0f);
      ImGui::DragFloat("max corresponding distance", &max_correspondence_distance, 0.1f, 0.0f);

      // Run optimization
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

    // Create relative pose constraint between t0 and t1
    if (enable_pose_constraint) {
      gtsam::Vector6 noise_scales;
      noise_scales << gtsam::Vector3::Ones() * rot_noise_scale, gtsam::Vector3::Ones() * trans_noise_scale;
      graph.add(gtsam::BetweenFactor(0, 1, gtsam::Pose3::Identity(), gtsam::noiseModel::Isotropic::Precisions(noise_scales)));
    }

    // Create continuous time ICP factor
    gtsam::NonlinearFactor::shared_ptr factor;
    if (factor_types[factor_type] == std::string("CT-ICP")) {
      auto f = gtsam::make_shared<gtsam_points::IntegratedCT_ICPFactor>(0, 1, deskewed_frames[data_id], raw_frames[data_id]);
      f->set_max_correspondence_distance(max_correspondence_distance);
      factor = f;
    } else if (factor_types[factor_type] == std::string("CT-GICP")) {
      auto f = gtsam::make_shared<gtsam_points::IntegratedCT_GICPFactor>(0, 1, deskewed_frames[data_id], raw_frames[data_id]);
      f->set_max_correspondence_distance(max_correspondence_distance);
      factor = f;
    } else if (factor_types[factor_type] == std::string("CT-ICP-EXPR")) {
      auto noise_model = gtsam::noiseModel::Isotropic::Precision(1, 1.0);
      auto robust_model = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Huber::Create(max_correspondence_distance), noise_model);
      factor = gtsam_points::create_integrated_cticp_factor(0, 1, deskewed_frames[data_id], raw_frames[data_id], robust_model);
    } else {
      std::cerr << "error: unknown factor type " << factor_types[factor_type] << std::endl;
    }

    graph.add(factor);

    gtsam_points::LevenbergMarquardtExtParams lm_params;
    lm_params.callback = [&](const gtsam_points::LevenbergMarquardtOptimizationStatus& status, const gtsam::Values& values) {
      auto viewer = guik::LightViewer::instance();
      viewer->append_text(status.to_string());

      // Calculate deskewed source points
      std::vector<Eigen::Vector4d> points;
      auto cticp_factor = boost::dynamic_pointer_cast<gtsam_points::IntegratedCT_ICPFactor>(factor);
      if (cticp_factor) {
        points = cticp_factor->deskewed_source_points(values);
      } else {
        auto cticp_factor_expr = boost::dynamic_pointer_cast<gtsam_points::IntegratedCTICPFactorExpr>(factor);
        auto deskewed = cticp_factor_expr->deskewed_source_points(values);
        std::transform(deskewed.begin(), deskewed.end(), std::back_inserter(points), [](const Eigen::Vector3d& p) {
          return (Eigen::Vector4d() << p, 1.0).finished();
        });
      }

      // Calculate interpolated poses for visualization
      std::vector<Eigen::Isometry3f> poses;
      for (int i = 0; i < 4; i++) {
        gtsam::Pose3 pose = gtsam_points::interpolate_pose(values.at<gtsam::Pose3>(0), values.at<gtsam::Pose3>(1), static_cast<double>(i) / 3);
        poses.push_back(Eigen::Isometry3f(pose.matrix().cast<float>()));
      }

      // Point times
      std::vector<double> times(raw_frames[data_id]->times, raw_frames[data_id]->times + raw_frames[data_id]->size());

      viewer->invoke([=] {
        auto cloud_buffer = std::make_shared<glk::PointCloudBuffer>(points);
        cloud_buffer->add_intensity(glk::COLORMAP::BLUE_RED, times, 1.0 / times.back());
        viewer->update_drawable("source", cloud_buffer, guik::VertexColor());

        for (int i = 0; i < poses.size(); i++) {
          viewer->update_drawable(
            "coord_" + std::to_string(i),
            glk::Primitives::coordinate_system(),
            guik::VertexColor(poses[i] * Eigen::UniformScaling<float>(1.0f)));
        }
      });
    };

    gtsam_points::LevenbergMarquardtOptimizerExt optimizer(graph, values, lm_params);
    optimizer.optimize();
  }

private:
  std::vector<gtsam_points::PointCloud::Ptr> raw_frames;
  std::vector<gtsam_points::PointCloud::Ptr> deskewed_frames;

  float pose_noise_scale;
  gtsam::Pose3 pose_noise;

  int factor_type;
  std::vector<const char*> factor_types;

  bool enable_pose_constraint;
  float rot_noise_scale;
  float trans_noise_scale;
  float max_correspondence_distance;

  int data_id;
  std::thread optimization_thread;
};

int main(int argc, char** argv) {
  ContinuousTimeDemo demo;
  guik::LightViewer::instance()->spin();
  return 0;
}