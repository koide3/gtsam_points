#include <fstream>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <gtsam_points/util/continuous_trajectory.hpp>
#include <guik/viewer/light_viewer.hpp>

class PoseInterpolator {
public:
  /**
   * @brief Load input poses from a file.
   * @param filename Input file name (TUM format).
   */
  void load_input_poses(const std::string& filename) {
    std::cout << "loading " << filename << std::endl;

    std::ifstream ifs(filename);
    if (!ifs) {
      std::cerr << "Failed to open file: " << filename << std::endl;
      abort();
    }

    std::string line;
    while (!ifs.eof() && std::getline(ifs, line) && !line.empty()) {
      std::istringstream iss(line);
      double timestamp;
      Eigen::Vector3d trans;
      Eigen::Quaterniond quat;
      iss >> timestamp >> trans.x() >> trans.y() >> trans.z() >> quat.x() >> quat.y() >> quat.z() >> quat.w();

      input_timestamps.emplace_back(timestamp);
      input_poses.emplace_back(gtsam::Pose3(gtsam::Rot3(quat.normalized().toRotationMatrix()), trans));
    }

    if (!std::is_sorted(input_timestamps.begin(), input_timestamps.end())) {
      std::cerr << "timestamps are not sorted" << std::endl;
    }
  }

  /**
   * @brief Load interpolation timestamps from a file.
   * @param filename Timestamp file name.
   */
  void load_timestamps(const std::string& filename) {
    std::cout << "loading timestamps from " << filename << std::endl;

    std::ifstream ifs(filename);
    if (!ifs) {
      std::cerr << "Failed to open file: " << filename << std::endl;
      abort();
    }

    timestamps.clear();
    double t;
    while (ifs >> t) {
      timestamps.push_back(t);
    }
  }

  /**
   * @brief Generate timestamps from start to end with step.
   * @param start Start time. If negative, use the first timestamp in the input.
   * @param end End time. If negative, use the last timestamp in the input.
   * @param step Time step.
   */
  void generate_timestamps_arange(double start, double end, double step) {
    if (start < 0.0) {
      start = input_timestamps.front();
    }
    if (end < 0.0) {
      end = input_timestamps.back();
    }

    std::cout << "generating timestamps from " << start << " to " << end << " with step " << step << std::endl;

    for (double t = start; t <= end; t += step) {
      timestamps.push_back(t);
    }
    if (std::abs(timestamps.back() - end) > 1e-6) {
      timestamps.push_back(end);
    }
  }

  /**
   * @brief Interpolate the trajectory.
   * @param knot_interval Knot interval. If negative, estimate the knot interval from the input timestamps.
   * @param smoothness Smoothness parameter for the spline.
   */
  void interpolate(double knot_interval, double smoothness) {
    if (knot_interval < 0.0) {
      std::vector<double> dts;
      for (size_t i = 1; i < input_timestamps.size(); i++) {
        dts.push_back(input_timestamps[i] - input_timestamps[i - 1]);
        if (dts.back() <= 0.0) {
          std::cerr << "negative dt detected: i=" << i << " t1=" << input_timestamps[i - 1] << " t2=" << input_timestamps[i] << std::endl;
        }
      }

      std::vector<double> dts_sorted = dts;
      std::sort(dts_sorted.begin(), dts_sorted.end());
      knot_interval = dts_sorted[dts_sorted.size() / 2];

      for (int i = 0; i < dts.size(); i++) {
        if (dts[i] < 0.5 * knot_interval || dts[i] > 2.0 * knot_interval) {
          std::cerr << "unusual dt detected: i=" << i << " dt=" << dts[i] << " t1=" << input_timestamps[i] << " t2=" << input_timestamps[i + 1]
                    << std::endl;
        }
      }
    }

    std::cout << "interpolating trajectory (knot_interval=" << knot_interval << ")" << std::endl;
    std::cout << "|input|=" << input_timestamps.size() << std::endl;

    gtsam_points::ContinuousTrajectory traj('x', input_timestamps.front(), input_timestamps.back(), knot_interval);
    knots = traj.fit_knots(input_timestamps, input_poses, smoothness, true);
    std::cout << "|knots|=" << knots.size() << std::endl;

    std::cout << "calculating interpolated poses" << std::endl;
    std::cout << "|timestamps|=" << timestamps.size() << std::endl;
    poses.resize(timestamps.size());
    std::transform(timestamps.begin(), timestamps.end(), poses.begin(), [&](double t) { return traj.pose(knots, t); });

    std::cout << "calculating IMU measurements on the interpolated trajectory" << std::endl;
    imus.resize(timestamps.size());
    std::transform(timestamps.begin(), timestamps.end(), imus.begin(), [&](double t) { return traj.imu(knots, t); });
  }

  /**
   * @brief Save the interpolated poses to a file.
   * @param filename Output file name.
   */
  void save_poses(const std::string& filename) {
    std::cout << "saving poses to " << filename << std::endl;
    std::ofstream ofs(filename);
    if (!ofs) {
      std::cerr << "Failed to open file: " << filename << std::endl;
      abort();
    }

    for (int i = 0; i < timestamps.size(); i++) {
      const gtsam::Pose3& pose = poses[i];
      const Eigen::Vector3d trans = pose.translation();
      const Eigen::Quaterniond quat(pose.rotation().toQuaternion());

      ofs << boost::format("%.9f %.6f %.6f %.6f %.6f %.6f %.6f %.6f") % timestamps[i] % trans.x() % trans.y() % trans.z() % quat.x() % quat.y() %
               quat.z() % quat.w()
          << std::endl;
    }
  }

  /**
   * @brief Save the IMU measurements to a file.
   * @param filename Output file name.
   */
  void save_imu(const std::string& filename) {
    std::cout << "saving IMU measurements to " << filename << std::endl;
    std::ofstream ofs(filename);
    if (!ofs) {
      std::cerr << "Failed to open file: " << filename << std::endl;
      abort();
    }

    for (int i = 0; i < timestamps.size(); i++) {
      const gtsam::Vector6& imu = imus[i];
      ofs << boost::format("%.9f %.6f %.6f %.6f %.6f %.6f %.6f") % timestamps[i] % imu[0] % imu[1] % imu[2] % imu[3] % imu[4] % imu[5] << std::endl;
    }
  }

  /**
   * @brief Visualize the interpolated trajectory.
   */
  void visualize() {
    auto viewer = guik::viewer();

    bool show_inputs = false;
    bool show_knots = false;
    bool show_interpolated_traj = true;
    bool show_interpolated_poses = true;

    int show_input_step = 10;
    int show_interpolated_pose_step = 100;

    viewer->register_ui_callback("ui_callback", [&] {
      ImGui::Checkbox("show_inputs", &show_inputs);
      ImGui::Checkbox("show_knots", &show_knots);
      ImGui::Checkbox("show_interpolated_traj", &show_interpolated_traj);
      ImGui::Checkbox("show_interpolated_poses", &show_interpolated_poses);

      ImGui::DragInt("show_input_step", &show_input_step, 0.1, 1, 1000);
      ImGui::DragInt("show_interpolated_pose_step", &show_interpolated_pose_step, 0.1, 1, 1000);
    });

    viewer->register_drawable_filter("filter", [&](const std::string& name) {
      if (name.find("input_") != std::string::npos) {
        if (!show_inputs) {
          return false;
        }

        const int i = std::stoi(name.substr(6));
        return i % show_input_step == 0;
      }

      if (name.find("knot_") != std::string::npos) {
        if (!show_knots) {
          return false;
        }

        return true;
      }

      if (name.find("interpolated_pose_") != std::string::npos) {
        if (!show_interpolated_poses) {
          return false;
        }

        const int i = std::stoi(name.substr(18));
        return i % show_interpolated_pose_step == 0;
      }

      if (name == "interpolated_traj") {
        return show_interpolated_traj;
      }

      return true;
    });

    // Register drawables
    for (int i = 0; i < input_poses.size(); i++) {
      viewer->update_coord("input_" + std::to_string(i), guik::VertexColor(input_poses[i].matrix()));
    }

    for (int i = 0; i < knots.size(); i++) {
      const gtsam::Pose3 pose = knots.at<gtsam::Pose3>(gtsam::Symbol('x', i));
      viewer->update_coord("knot_" + std::to_string(i), guik::FlatGray(pose.matrix()));
    }

    for (int i = 0; i < poses.size(); i++) {
      viewer->update_coord("interpolated_pose_" + std::to_string(i), guik::VertexColor(poses[i].matrix()));
    }

    std::vector<Eigen::Vector3d> interpolated_traj(poses.size());
    std::transform(poses.begin(), poses.end(), interpolated_traj.begin(), [](const gtsam::Pose3& pose) { return pose.translation(); });
    viewer->update_thin_lines("interpolated_traj", interpolated_traj, true, guik::FlatGreen());

    std::vector<int> indices(poses.size());
    std::iota(indices.begin(), indices.end(), 0);

    viewer->update_plot_line("acc", "x", indices, [&](int i) -> Eigen::Vector2d { return {timestamps[i], imus[i][0]}; });
    viewer->update_plot_line("acc", "y", indices, [&](int i) -> Eigen::Vector2d { return {timestamps[i], imus[i][1]}; });
    viewer->update_plot_line("acc", "z", indices, [&](int i) -> Eigen::Vector2d { return {timestamps[i], imus[i][2]}; });

    viewer->update_plot_line("gyro", "x", indices, [&](int i) -> Eigen::Vector2d { return {timestamps[i], imus[i][3]}; });
    viewer->update_plot_line("gyro", "y", indices, [&](int i) -> Eigen::Vector2d { return {timestamps[i], imus[i][4]}; });
    viewer->update_plot_line("gyro", "z", indices, [&](int i) -> Eigen::Vector2d { return {timestamps[i], imus[i][5]}; });

    viewer->spin();
  }

private:
  std::vector<double> input_timestamps;   // Input timestamps
  std::vector<gtsam::Pose3> input_poses;  // Input poses

  gtsam::Values knots;  // Spline knots fitted to input poses

  std::vector<double> timestamps;    // Interpolation timestamps
  std::vector<gtsam::Pose3> poses;   // Interpolated poses
  std::vector<gtsam::Vector6> imus;  // IMU measurements on the interpolated trajectory
};

int main(int argc, char** argv) {
  using namespace boost::program_options;

  options_description desc("Options");
  desc.add_options()                                                                                                                           //
    ("help,h", "Help")                                                                                                                         //
    ("input", value<std::string>(), "Input trajectory filename (TUM format [t, x, y, z, qx, qy, qz, qw]).")                                    //
    ("timestamps", value<std::string>(), "Interpolation timestamps filename (timestamp in each row).")                                         //
    ("output", value<std::string>(), "Interpolated trajectory output filename.")                                                               //
    ("imu_output", value<std::string>(), "Interpolated IMU output filename.")                                                                  //
    ("t_begin", value<double>()->default_value(-1.0), "Interpolation begin time. -1 for using the beginning time of the input traj.")          //
    ("t_end", value<double>()->default_value(-1.0), "Interpolation end time. -1 for using the end time of the input traj.")                    //
    ("t_step", value<double>()->default_value(0.01), "Interpolation time step.")                                                               //
    ("knot_interval", value<double>()->default_value(-1.0), "Interpolation knot interval. -1 for using the time interval of the input traj.")  //
    ("smoothness", value<double>()->default_value(1e-3), "Interpolation smoothness.")                                                          //
    ("visualize,v", "Enable visualization.")                                                                                                   //

    ;

  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);
  notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    return 0;
  }

  PoseInterpolator interpolator;

  // Input
  interpolator.load_input_poses(vm["input"].as<std::string>());
  if (vm.count("timestamps")) {
    interpolator.load_timestamps(vm["timestamps"].as<std::string>());
  } else {
    interpolator.generate_timestamps_arange(vm["t_begin"].as<double>(), vm["t_end"].as<double>(), vm["t_step"].as<double>());
  }

  // Interpolate!!
  interpolator.interpolate(vm["knot_interval"].as<double>(), vm["smoothness"].as<double>());

  // Output
  if (vm.count("visualize")) {
    interpolator.visualize();
  }
  if (vm.count("output")) {
    interpolator.save_poses(vm["output"].as<std::string>());
  }
  if (vm.count("imu_output")) {
    interpolator.save_imu(vm["imu_output"].as<std::string>());
  }

  return 0;
}