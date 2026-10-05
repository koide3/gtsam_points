// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cstdlib>
#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>
#include <gtsam_points/ann/knn_result.hpp>

namespace {
struct TransformState {
  int live_instances = 0;
  size_t offset = 100;
};

struct TrackedTransform {
  explicit TrackedTransform(const std::shared_ptr<TransformState>& state) : state(state) { ++state->live_instances; }
  TrackedTransform(const TrackedTransform& other) : state(other.state) { ++state->live_instances; }
  ~TrackedTransform() { --state->live_instances; }
  size_t operator()(size_t index) const { return state->offset + index; }
  std::shared_ptr<TransformState> state;
};

template <int N>
void check_temporary_knn() {
  constexpr int capacity = N > 0 ? N : 3;
  std::array<size_t, capacity> indices;
  std::array<double, capacity> distances;
  auto state = std::make_shared<TransformState>();
  gtsam_points::KnnResult<N, TrackedTransform> result(indices.data(), distances.data(), N > 0 ? -1 : capacity, TrackedTransform(state));
  // Check ownership before invoking the callback: the original implementation
  // destroys the sole transform here, and calling it would be undefined behavior.
  ASSERT_EQ(state->live_instances, 1);
  result.push(4, 4.0);
  result.push(2, 2.0);
  result.push(8, 8.0);
  result.push(1, 1.0);
  ASSERT_EQ(result.num_found(), capacity);
  EXPECT_EQ(indices.front(), 101);
  EXPECT_DOUBLE_EQ(distances.front(), 1.0);
  if constexpr (capacity == 3) {
    EXPECT_EQ(indices[1], 102);
    EXPECT_EQ(indices[2], 104);
    EXPECT_DOUBLE_EQ(distances[1], 2.0);
    EXPECT_DOUBLE_EQ(distances[2], 4.0);
  }
}
}  // namespace

TEST(KnnResultLifetime, StaticOneTemporary) {
  check_temporary_knn<1>();
}
TEST(KnnResultLifetime, StaticThreeTemporary) {
  check_temporary_knn<3>();
}
TEST(KnnResultLifetime, DynamicTemporary) {
  check_temporary_knn<-1>();
}

TEST(KnnResultLifetime, RadiusTemporary) {
  auto state = std::make_shared<TransformState>();
  gtsam_points::RadiusSearchResult<TrackedTransform> result{TrackedTransform(state)};
  ASSERT_EQ(state->live_instances, 1);
  result.push(9, 9.0);
  result.push(1, 1.0);
  result.sort();
  ASSERT_EQ(result.num_found(), 2);
  EXPECT_EQ(result.neighbors[0].first, 101);
  EXPECT_EQ(result.neighbors[1].first, 109);
  EXPECT_DOUBLE_EQ(result.neighbors[0].second, 1.0);
  EXPECT_DOUBLE_EQ(result.neighbors[1].second, 9.0);
}

TEST(KnnResultLifetime, KnnOutlivesInputTransform) {
  size_t index;
  double distance;
  auto state = std::make_shared<TransformState>();
  std::optional<gtsam_points::KnnResult<1, TrackedTransform>> result;
  {
    TrackedTransform transform(state);
    result.emplace(&index, &distance, -1, transform);
  }
  ASSERT_EQ(state->live_instances, 1);
  result->push(7, 0.5);
  EXPECT_EQ(index, 107);
  EXPECT_DOUBLE_EQ(distance, 0.5);
  result.reset();
  EXPECT_EQ(state->live_instances, 0);
}

TEST(KnnResultLifetime, RadiusOutlivesInputTransform) {
  auto state = std::make_shared<TransformState>();
  std::optional<gtsam_points::RadiusSearchResult<TrackedTransform>> result;
  {
    TrackedTransform transform(state);
    result.emplace(transform);
  }
  ASSERT_EQ(state->live_instances, 1);
  result->push(7, 0.5);
  ASSERT_EQ(result->num_found(), 1);
  EXPECT_EQ(result->neighbors[0].first, 107);
  result.reset();
  EXPECT_EQ(state->live_instances, 0);
}

TEST(KnnResultLifetime, KnnCopyOutlivesOriginal) {
  size_t index;
  double distance;
  auto state = std::make_shared<TransformState>();
  std::optional<gtsam_points::KnnResult<1, TrackedTransform>> copy;
  {
    TrackedTransform transform(state);
    gtsam_points::KnnResult<1, TrackedTransform> original(&index, &distance, -1, transform);
    copy.emplace(original);
  }
  ASSERT_EQ(state->live_instances, 1);
  copy->push(3, 0.25);
  EXPECT_EQ(index, 103);
  copy.reset();
  EXPECT_EQ(state->live_instances, 0);
}

TEST(KnnResultLifetime, RadiusCopyOutlivesOriginal) {
  auto state = std::make_shared<TransformState>();
  std::optional<gtsam_points::RadiusSearchResult<TrackedTransform>> copy;
  {
    TrackedTransform transform(state);
    gtsam_points::RadiusSearchResult<TrackedTransform> original(transform);
    original.push(2, 2.0);
    copy.emplace(original);
  }
  ASSERT_EQ(state->live_instances, 1);
  copy->push(1, 1.0);
  copy->sort();
  ASSERT_EQ(copy->num_found(), 2);
  EXPECT_EQ(copy->neighbors[0].first, 101);
  EXPECT_EQ(copy->neighbors[1].first, 102);
  copy.reset();
  EXPECT_EQ(state->live_instances, 0);
}

TEST(KnnResultBehavior, IdentityAndDistanceLimit) {
  std::array<size_t, 3> indices;
  std::array<double, 3> distances;
  const gtsam_points::identity_transform transform;
  gtsam_points::KnnResult<-1> result(indices.data(), distances.data(), 3, transform, 3.0);
  result.push(5, 5.0);
  result.push(2, 2.0);
  result.push(1, 1.0);
  EXPECT_EQ(result.num_found(), 2);
  EXPECT_EQ(indices[0], 1);
  EXPECT_EQ(indices[1], 2);
  EXPECT_EQ(indices[2], gtsam_points::KnnResult<-1>::INVALID);
  EXPECT_DOUBLE_EQ(result.worst_distance(), 3.0);
}

TEST(KnnResultBehavior, RadiusIdentity) {
  const gtsam_points::identity_transform transform;
  gtsam_points::RadiusSearchResult<> result(transform);
  result.push(4, 4.0);
  result.push(2, 2.0);
  result.sort();
  ASSERT_EQ(result.num_found(), 2);
  EXPECT_EQ(result.neighbors[0].first, 2);
  EXPECT_EQ(result.neighbors[1].first, 4);
}

TEST(KnnResultBehavior, KnnReferenceCapture) {
  size_t offset = 10;
  const auto transform = [&offset](size_t index) { return offset + index; };
  size_t index;
  double distance;
  gtsam_points::KnnResult<1, decltype(transform)> result(&index, &distance, -1, transform);
  offset = 20;
  result.push(2, 0.5);
  EXPECT_EQ(index, 22);
}

TEST(KnnResultBehavior, RadiusReferenceCapture) {
  size_t offset = 10;
  const auto transform = [&offset](size_t index) { return offset + index; };
  gtsam_points::RadiusSearchResult<decltype(transform)> result(transform);
  result.push(1, 1.0);
  offset = 20;
  result.push(2, 2.0);
  ASSERT_EQ(result.num_found(), 2);
  EXPECT_EQ(result.neighbors[0].first, 11);
  EXPECT_EQ(result.neighbors[1].first, 22);
}
