// SPDX-License-Identifier: MIT
// Copyright (c) 2024  Kenji Koide (k.koide@aist.go.jp)
#pragma once

#include <deque>
#include <cstdint>
#include <iostream>

namespace gtsam_points {

/**
 * @brief A sliding window container that allows absolute indexed access.
 *  It automatically shrinks by removing leading null elements.
 */
template <typename T>
class IndexedSlidingWindow {
public:
  /**
   * @brief Constructor
   * @param auto_shrink If true, the container will automatically shrink
   *        by removing leading null elements when adding or accessing
   *        new elements.
   */
  IndexedSlidingWindow(bool auto_shrink = true) : auto_shrink(auto_shrink), total_size(0) {}

  bool empty() const { return total_size == 0; }
  size_t size() const { return total_size; }
  size_t inner_size() const { return data.size(); }

  void clear() {
    data.clear();
    total_size = 0;
  }

  void push_back(const T& value) {
    if (auto_shrink) {
      shrink();
    }

    data.push_back(value);
    total_size++;
  }

  void emplace_back(T&& value) {
    if (auto_shrink) {
      shrink();
    }

    data.emplace_back(std::move(value));
    total_size++;
  }

  /// @brief Remove leading null elements from the container.
  void shrink() {
    while (!data.empty() && !data.front()) {
      data.pop_front();
    }
  }

  T& operator[](size_t index) {
    if (auto_shrink) {
      shrink();
    }

    const std::int64_t local_index = static_cast<std::int64_t>(index) - static_cast<std::int64_t>(total_size - data.size());
    if (local_index < 0 || local_index >= static_cast<std::int64_t>(data.size())) {
      std::cerr << "local_index: " << local_index << ", data.size(): " << data.size() << ", total_size: " << total_size << std::endl;
      throw std::out_of_range("IndexedSlidingWindow: index out of range");
    }
    return data[local_index];
  }

  const T& operator[](size_t index) const {
    const std::int64_t local_index = static_cast<std::int64_t>(index) - static_cast<std::int64_t>(total_size - data.size());
    if (local_index < 0 || local_index >= static_cast<std::int64_t>(data.size())) {
      std::cerr << "local_index: " << local_index << ", data.size(): " << data.size() << ", total_size: " << total_size << std::endl;
      throw std::out_of_range("IndexedSlidingWindow: index out of range");
    }
    return data[local_index];
  }

  typename std::deque<T>::iterator inner_begin() { return data.begin(); }
  typename std::deque<T>::iterator inner_end() { return data.end(); }
  typename std::deque<T>::const_iterator inner_begin() const { return data.begin(); }
  typename std::deque<T>::const_iterator inner_end() const { return data.end(); }

  T& inner_front() { return data.front(); }
  const T& inner_front() const { return data.front(); }

  T& inner_back() { return data.back(); }
  const T& inner_back() const { return data.back(); }

  T& back() { return data.back(); }
  const T& back() const { return data.back(); }

private:
  bool auto_shrink;    ///< If true, automatically shrink leading null elements.
  size_t total_size;   ///< Total number of inserted elements.
  std::deque<T> data;  ///< Internal storage for non-null elements.
};

}  // namespace gtsam_points
