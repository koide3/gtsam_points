#include <benchmark/dataset.hpp>

#define FMT_HEADER_ONLY
#include <regex>
#include <fstream>
#include <fmt/format.h>

Dataset::Dataset(const std::filesystem::path& data_path, bool force_recreate_voxelmaps) : data_path(data_path) {
  std::cout << "reading " + data_path.string() + "...\n" << std::flush;

  std::ifstream ifs(data_path);
  if (!ifs) {
    std::cerr << "error: failed to open " << data_path << std::endl;
    return;
  }

  bool reading_keyframes = false;
  std::string line;
  while (!ifs.eof() && std::getline(ifs, line) && !line.empty()) {
    if (line == "frames") {
      continue;
    }
    if (line == "keyframes") {
      reading_keyframes = true;
      continue;
    }

    const std::string f = "[+-]?[0-9]+\\.[0-9]+";
    std::regex pattern(fmt::format("([0-9]+) ({}) se3\\(({}),({}),({}),({}),({}),({}),({})+\\)", f, f, f, f, f, f, f, f));
    std::smatch match;
    if (std::regex_search(line, match, pattern)) {
      auto frame = std::make_shared<Frame>();
      frame->id = std::stoi(match[1]);
      frame->stamp = std::stod(match[2]);

      const auto found = std::find_if(frames.begin(), frames.end(), [frame](const auto& f) { return f->id == frame->id; });
      if (reading_keyframes && found != frames.end()) {
        keyframes.emplace_back(*found);
        continue;
      }

      const Eigen::Vector3d trans(std::stod(match[3]), std::stod(match[4]), std::stod(match[5]));
      const Eigen::Quaterniond quat(std::stod(match[9]), std::stod(match[6]), std::stod(match[7]), std::stod(match[8]));
      frame->pose.setIdentity();
      frame->pose.linear() = quat.normalized().toRotationMatrix();
      frame->pose.translation() = trans;

      const std::string framename = reading_keyframes ? "keyframe" : "frame";
      const std::string path = data_path.parent_path() / fmt::format("{}_{:06d}", framename, frame->id);

      auto points = gtsam_points::PointCloudCPU::load(path);
      frame->points = gtsam_points::PointCloudGPU::clone(*points);

      const std::array<double, 2> resolutions = {0.5, 1.0};
      for (const auto r : resolutions) {
        const std::string voxelmap_path = fmt::format("{}/voxelmap_{:.3f}.bin", path, r);
        auto voxelmap = gtsam_points::GaussianVoxelMapGPU::load(voxelmap_path);

        if (voxelmap && !force_recreate_voxelmaps) {
          frame->voxelmaps.emplace_back(voxelmap);
        } else {
          std::cerr << "error: failed to load " << voxelmap_path << std::endl;
          std::cerr << "creating a new voxelmap" << std::endl;

          auto voxelmap = std::make_shared<gtsam_points::GaussianVoxelMapGPU>(r, 8192 * 8, 10, 0.0);
          voxelmap->insert(*frame->points);
          frame->voxelmaps.emplace_back(voxelmap);

          voxelmap->save_compact(voxelmap_path);
        }
      }

      if (!reading_keyframes) {
        frames.emplace_back(frame);
      } else {
        keyframes.emplace_back(frame);
      }
    } else {
      std::cerr << "warning : invalid line " << line << std::endl;
    }
  }

  std::cout << fmt::format("|frames|={} |keyframes|={}", frames.size(), keyframes.size()) << std::endl;
}
