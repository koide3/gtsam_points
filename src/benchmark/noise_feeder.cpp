#include <benchmark/noise_feeder.hpp>

#include <random>
#include <fstream>
#include <iostream>
#include <algorithm>

NoiseFeeder::NoiseFeeder() : count(0) {
  std::mt19937 mt;
  std::normal_distribution<> ndist(0.0, 1.0);
  noise.resize(8192);
  std::generate(noise.begin(), noise.end(), [&] { return ndist(mt); });
}

void NoiseFeeder::read(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) {
    std::cerr << "error : failed to open " << path << std::endl;
    return;
  }

  noise.clear();
  std::string line;
  while (!ifs.eof() && std::getline(ifs, line) && !line.empty()) {
    noise.emplace_back(std::stod(line));
  }
}