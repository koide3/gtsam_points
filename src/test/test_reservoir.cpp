#include <vector>
#include <numeric>
#include <iostream>

#include <gtest/gtest.h>
#include <gtsam_points/util/reservoir.hpp>

template <int k>
void test_reservoir() {
  std::mt19937 mt(0);

  constexpr int num_classes = 10;
  constexpr size_t num_samples = 100000;
  constexpr size_t num_samples_lb = (num_samples / num_classes) * k * 0.9;
  constexpr size_t num_samples_ub = (num_samples / num_classes) * k * 1.1;

  std::vector<int> counts(num_classes, 0);
  for (int i = 0; i < num_samples; i++) {
    gtsam_points::Reservoir<int, k> reservoir;
    for (int i = 0; i < counts.size(); i++) {
      reservoir.push(i, mt);
    }

    for (const auto& sample : reservoir.samples) {
      counts[sample]++;
    }
  }

  for (int i = 0; i < counts.size(); i++) {
    EXPECT_GE(counts[i], num_samples_lb);
    EXPECT_LE(counts[i], num_samples_ub);
  }
}

TEST(TestReservoir, TestReservoir) {
  test_reservoir<1>();
  test_reservoir<2>();
  test_reservoir<4>();
  test_reservoir<8>();
}

template <int k>
void test_weighted_reservoir() {
  std::mt19937 mt(0);

  constexpr int num_classes = 10;
  constexpr size_t num_samples = 1000000;

  std::vector<double> weights(num_classes);
  for (int i = 0; i < num_classes; i++) {
    weights[i] = static_cast<double>(i + 1);
  }

  std::vector<int> counts(num_classes, 0);
  for (int i = 0; i < num_samples; i++) {
    gtsam_points::WeightedReservoir<int, k> reservoir;
    for (int i = 0; i < counts.size(); i++) {
      reservoir.push(weights[i], i, mt);
    }

    for (const auto& sample : reservoir.samples) {
      counts[sample]++;
    }
  }

  // Each class can be selected at most once per iteration, so its count must not exceed num_samples.
  for (int i = 0; i < counts.size(); i++) {
    EXPECT_GE(counts[i], 0);
    EXPECT_LE(counts[i], static_cast<int>(num_samples)) << "k=" << k << ", class=" << i;
  }

  if (k == 1) {
    // For k == 1, the marginal inclusion probability is exactly proportional to the weight:
    //   P(class i) = w_i / sum_j w_j
    const double weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (int i = 0; i < counts.size(); i++) {
      const double expected_ratio = weights[i] / weight_sum;
      const double actual_ratio = static_cast<double>(counts[i]) / num_samples;
      EXPECT_NEAR(actual_ratio, expected_ratio, 0.01) << "class=" << i;
    }
  } else {
    // For k > 1 (weighted sampling without replacement), marginal inclusion probabilities are no
    // longer proportional to the weights, but they remain monotonically non-decreasing in weight.
    // Since weights are strictly increasing in the class index, counts should be (approximately)
    // monotonically non-decreasing as well.
    for (int i = 1; i < counts.size(); i++) {
      EXPECT_GE(counts[i], counts[i - 1]) << "k=" << k << ", class=" << i;
    }
  }
}

TEST(TestReservoir, TestWeightedReservoir) {
  test_weighted_reservoir<1>();
  test_weighted_reservoir<2>();
  test_weighted_reservoir<4>();
  test_weighted_reservoir<8>();
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}