#include <iostream>
#include <gtsam/inference/Symbol.h>
#include <gtsam_points/util/continuous_trajectory.hpp>

#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>

int main(int argc, char** argv) {
  auto viewer = guik::LightViewer::instance();
  viewer->set_draw_xy_grid(false);

  // Generate random target poses
  std::vector<double> stamps;
  std::vector<gtsam::Pose3> poses;
  for (double t = 0.0; t < 10.0; t += 1.0) {
    stamps.push_back(t);

    const gtsam::Vector3 trans(2.0 * t, std::sin(2.0 * t), 0.0);
    const gtsam::Rot3 rot = gtsam::Rot3::Expmap(gtsam::Vector3::Random() * 0.5);
    poses.push_back(gtsam::Pose3(rot, trans));

    Eigen::Affine3f model_matrix(poses.back().matrix().cast<float>());
    viewer->update_drawable("target_" + std::to_string(t), glk::Primitives::coordinate_system(), guik::VertexColor(model_matrix.scale(0.5f)));
  }

  // Create continuous trajectory and optimize spline knots to fit the trajectory with the target poses
  gtsam_points::ContinuousTrajectory ct('x', stamps.front(), stamps.back(), 0.5);
  gtsam::Values values = ct.fit_knots(stamps, poses);

  for (int i = 0; i < ct.knot_max_id(); i++) {
    Eigen::Affine3f model_matrix(values.at<gtsam::Pose3>(gtsam::Symbol('x', i)).matrix().cast<float>());
    viewer->update_drawable("knot_" + std::to_string(i), glk::Primitives::coordinate_system(), guik::FlatColor(0.5f, 0.5f, 0.5f, 1.0f, model_matrix.scale(0.5f)));
  }

  // Drawable filter
  bool show_knots = true;
  bool show_targets = false;
  viewer->register_drawable_filter("filter", [&](const std::string& name) {
    if (!show_knots && name.find("knot_") != std::string::npos) {
      return false;
    }

    if (!show_targets && name.find("target_") != std::string::npos) {
      return false;
    }

    return true;
  });

  // Visualization
  float time = stamps.front();
  viewer->register_ui_callback("ui", [&] {
    ImGui::Checkbox("knots", &show_knots);
    ImGui::SameLine();
    ImGui::Checkbox("targets", &show_targets);
    ImGui::Separator();

    ImGui::DragFloat("time", &time, 0.01f, stamps.front(), stamps.back());

    // Calculate interpolated pose and linear acc and angular vel
    const gtsam::Pose3 pose = ct.pose(values, time);
    const gtsam::Vector6 imu = ct.imu(values, time);

    // There are also expression interfaces to easily create constraints
    const gtsam::Pose3_ pose_ = ct.pose(time, gtsam::Double_(time));
    const gtsam::Vector6_ imu_ = ct.imu(time, gtsam::Double_(time));

    Eigen::Affine3f model_matrix(pose.matrix().cast<float>());
    viewer->update_drawable("pose", glk::Primitives::coordinate_system(), guik::VertexColor(model_matrix.scale(0.5f)));

    ImGui::Text("Linear acc : %.3f %.3f %.3f", imu[0], imu[1], imu[2]);
    ImGui::Text("Angular vel: %.3f %.3f %.3f", imu[3], imu[4], imu[5]);
  });

  viewer->spin();
}