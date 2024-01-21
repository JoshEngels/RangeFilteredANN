#pragma once

#include "algorithms/utils/types.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "algorithms/utils/point_range.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "pybind11/numpy.h"

#include "postfilter_vamana.h"
#include "prefiltering.h"

#include "tree_utils.h"

template <typename T, typename Point,
          template <typename, typename, typename> class RangeSpatialIndex =
              PrefilterIndex,
          typename FilterType = float_t>
struct SuperOptimizedPostfilterTree {
  using pid = std::pair<index_type, float>;

  using PR = PointRange<T, Point>;
  using SubsetRange = SubsetPointRange<T, Point>;
  using SubsetRangePtr = std::unique_ptr<SubsetRange>;

  using SpatialIndex = RangeSpatialIndex<T, Point, SubsetRange>;
  using SpatialIndexPtr = std::unique_ptr<SpatialIndex>;

  using FilterRange = std::pair<FilterType, FilterType>;
  using FilterList = parlay::sequence<FilterType>;

  // This constructor just sorts the input points and turns them into a
  // structure that's easier to work with. The actual work of building the index
  // happens in the private constructor below.
  SuperOptimizedPostfilterTree(py::array_t<T> points,
                               py::array_t<FilterType> filter_values,
                               int32_t cutoff, float split_factor,
                               float shift_factor, BuildParams build_params) {

    auto [sorted_point_range, sorted_filter_values, decoding] =
        sort_python_and_convert<FilterType, T, Point>(points, filter_values);

    *this =
        SuperOptimizedPostfilterTree<T, Point, RangeSpatialIndex, FilterType>(
            std::make_unique<PR>(sorted_point_range), sorted_filter_values,
            decoding, cutoff, split_factor, shift_factor, build_params);
  }

  /* the bounds here are inclusive */
  NeighborsAndDistances batch_search(
      py::array_t<T, py::array::c_style | py::array::forcecast> &queries,
      const std::vector<FilterRange> &filters, uint64_t num_queries,
      QueryParams query_params) {
    size_t knn = query_params.k;
    py::array_t<unsigned int> ids({num_queries, knn});
    py::array_t<float> dists({num_queries, knn});

    parlay::parallel_for(0, num_queries, [&](auto i) {
      Point q = Point(queries.data(i), _points->dimension(),
                      _points->aligned_dimension(), i);
      FilterRange filter = filters[i];

      parlay::sequence<pid> results;
      results = super_optimized_postfiltering_search(q, filter, query_params);
      for (auto j = 0; j < knn; j++) {
        if (j < results.size()) {
          ids.mutable_at(i, j) =
              _sorted_index_to_original_point_id.at(results[j].first);
          dists.mutable_at(i, j) = results[j].second;
        } else {
          ids.mutable_at(i, j) = 0;
          dists.mutable_at(i, j) = std::numeric_limits<float>::max();
        }
      }
    });
    return std::make_pair(ids, dists);
  }

private:
  std::vector<size_t> _bucket_sizes;
  std::vector<size_t> _bucket_shifts;
  std::vector<std::vector<SpatialIndexPtr>> _spatial_indices;

  parlay::sequence<size_t> _sorted_index_to_original_point_id;

  FilterList _filter_values;

  int32_t _cutoff;

  std::unique_ptr<PR> _points;

  float _split_factor, _shift_factor;

  static SpatialIndexPtr create_index(FilterList &filter_values, size_t start,
                                      size_t end, PR *points,
                                      BuildParams build_params) {
    auto filter_length = end - start;
    parlay::sequence<int32_t> subset_of_indices = parlay::tabulate<int32_t>(
        filter_length, [&](auto i) { return i + start; });
    SubsetRangePtr subset_points = points->make_subset(subset_of_indices);
    FilterList subset_of_filter_values =
        FilterList(filter_values.begin() + start, filter_values.begin() + end);

    return std::make_unique<SpatialIndex>(
        std::move(subset_points), subset_of_filter_values, build_params);
  }

  SuperOptimizedPostfilterTree(std::unique_ptr<PR> points,
                               const FilterList &filter_values,
                               const parlay::sequence<size_t> &decoding,
                               int32_t cutoff, float split_factor,
                               float shift_factor, BuildParams build_params)
      : _sorted_index_to_original_point_id(decoding), _cutoff(cutoff),
        _filter_values(filter_values), _points(std::move(points)),
        _split_factor(split_factor), _shift_factor(shift_factor) {

    if (split_factor <= 1) {
      throw std::runtime_error("split_factor must be greater than 1");
    }
    if (shift_factor >= 1 || shift_factor <= 0) {
      throw std::runtime_error("shift_factor must be between 0 and 1");
    }

    size_t n = _filter_values.size();

    _spatial_indices.push_back(std::vector<SpatialIndexPtr>(1));
    _spatial_indices.at(0).at(0) = create_index(
        _filter_values, 0, _filter_values.size(), _points.get(), build_params);
    _bucket_sizes.push_back(_filter_values.size());
    _bucket_shifts.push_back(0);

    // TODO: Add analysis of expected space, possibly add option to tune
    // shift_factor and split_factor to maintain a worst case blowup

    while (_bucket_sizes.back() > cutoff) {

      size_t last_row_bucket_size = _bucket_sizes.back();
      size_t bucket_size =
          (last_row_bucket_size + split_factor - 1) / split_factor;
      size_t bucket_shift = ceil(bucket_size * shift_factor);
      _bucket_sizes.push_back(bucket_size);
      _bucket_shifts.push_back(bucket_shift);

      // The last bucket start must be at least n - bucket_size
      // For example, say n is 20, bucket_size is 3, and bucket_shift is 2.
      // Then the last bucket start must be at least 20 - 3 = 17, and the
      // total number of buckets is ceil[17/2] + 1 = 10. An equivalent way of
      // writing this is floor[(17+2-1)/ 2] + 1 = 10.
      size_t num_buckets =
          ((n - bucket_size) + bucket_shift - 1) / bucket_shift + 1;

      _spatial_indices.push_back(std::vector<SpatialIndexPtr>(num_buckets));
      parlay::parallel_for(0, num_buckets, [&](auto bucket_id) {
        size_t bucket_start = bucket_id * bucket_shift;
        size_t bucket_end = std::min(bucket_start + bucket_size, n);
        _spatial_indices.back().at(bucket_id) =
            create_index(_filter_values, bucket_start, bucket_end,
                         _points.get(), build_params);
      });
    }
  }

  bool check_empty(const FilterRange &range) {
    bool empty = range.second < _filter_values.front() ||
                 range.first > _filter_values.back();
    if (empty) {
      std::cout << "Query range is entirely outside the index range ("
                << _filter_values.front() << ", " << _filter_values.back()
                << ") index range vs. (" << range.first << ", " << range.second
                << ") This shouldn't happen but does not directly impact "
                   "correctness"
                << std::endl;
    }
    return empty;
  }

  parlay::sequence<pid> super_optimized_postfiltering_search(
      const Point &query, const FilterRange &range, QueryParams query_params) {

    auto start_time = std::chrono::high_resolution_clock::now();

    // if the query range is entirely outside the index range, return
    if (check_empty(range)) {
      return parlay::sequence<pid>();
    }

    auto inclusive_start =
        first_greater_than_or_equal_to<FilterType>(range.first, _filter_values);
    auto exclusive_end = first_greater_than_or_equal_to<FilterType>(
        range.second, _filter_values);

    int64_t current_index, current_row = 0;

    for (current_row = _bucket_sizes.size() - 1; current_row >= 0;
         current_row--) {
      if (current_row == 0) {
        current_index = 0;
        break;
      }

      size_t bucket_size = _bucket_sizes.at(current_row);
      if (bucket_size < exclusive_end - inclusive_start) {
        continue;
      }

      size_t bucket_shift = _bucket_shifts.at(current_row);
      size_t first_possible_bucket = inclusive_start / bucket_shift;
      size_t last_possible_bucket = (exclusive_end - 1) / bucket_shift;
      first_possible_bucket = std::min(
          first_possible_bucket, _spatial_indices.at(current_row).size() - 1);
      last_possible_bucket = std::min(
          last_possible_bucket, _spatial_indices.at(current_row).size() - 1);

      for (size_t test_bucket = first_possible_bucket;
           test_bucket <= last_possible_bucket; test_bucket++) {
        if (query_params.verbose) {
          std::cout << "Testing bucket " << test_bucket << std::endl;
        }
        auto bucket_start = test_bucket * bucket_shift;
        auto bucket_end =
            std::min(bucket_start + bucket_size, _filter_values.size());
        if (inclusive_start >= bucket_start && exclusive_end <= bucket_end) {
          if (query_params.verbose) {
            std::cout << "Query range = (" << inclusive_start << ","
                      << exclusive_end << "), smallest containing range (size "
                      << bucket_size << ") = (" << bucket_start << ","
                      << bucket_end << ")" << std::endl;
          }
          current_index = test_bucket;
          goto done;
        }
      }
    }

  done:

    auto bucket_end_time = std::chrono::high_resolution_clock::now();
    if (query_params.verbose) {
      std::cout << "Time to find bucket: "
                << std::chrono::duration_cast<std::chrono::nanoseconds>(
                       bucket_end_time - start_time)
                       .count()
                << "ns" << std::endl;
    }

    auto result = _spatial_indices.at(current_row)
                      .at(current_index)
                      ->query(query, range, query_params);

    if (query_params.verbose) {
      std::cout << "Time to do searcht: "
                << std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::high_resolution_clock::now() -
                       bucket_end_time)
                       .count()
                << "ns" << std::endl;
    }

    return result;
  }
};
