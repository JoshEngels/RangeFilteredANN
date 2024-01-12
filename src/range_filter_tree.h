#pragma once

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "algorithms/utils/point_range.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include "pybind11/numpy.h"

#include "postfilter_vamana.h"
#include "prefiltering.h"

using index_type = int32_t;

namespace py = pybind11;
using NeighborsAndDistances =
    std::pair<py::array_t<unsigned int>, py::array_t<float>>;

template <typename T, typename Point>
PointRange<T, Point> numpy_point_range(py::array_t<T> points) {
  py::buffer_info buf = points.request();
  if (buf.ndim != 2) {
    throw std::runtime_error("NumPy array must be 2-dimensional");
  }


  auto n = buf.shape[0];         // number of points
  auto dimension = buf.shape[1]; // dimension of each point

  T *numpy_data = static_cast<T *>(buf.ptr);

  return std::move(PointRange<T, Point>(numpy_data, n, dimension));
}

template <typename T, typename Point,
          template <typename, typename, typename> class RangeSpatialIndex =
              PrefilterIndex, typename FilterType = float_t>
struct RangeFilterTreeIndex {
  using pid = std::pair<index_type, float>;

  using PR = PointRange<T, Point>;
  using SubsetRange = SubsetPointRange<T, Point>;
  using SubsetRangePtr = std::unique_ptr<SubsetRange>;

  using SpatialIndex = RangeSpatialIndex<T, Point, SubsetRange>;
  using SpatialIndexPtr = std::unique_ptr<SpatialIndex>;


  // This constructor just sorts the input points and turns them into a
  // structure that's easier to work with. The actual work of building the index
  // happens in the private constructor below.
  RangeFilterTreeIndex(py::array_t<T> points,
                       py::array_t<FilterType> filter_values,
                       int32_t cutoff = 1000) {
    py::buffer_info points_buf = points.request();
    if (points_buf.ndim != 2) {
      throw std::runtime_error("points numpy array must be 2-dimensional");
    }
    auto n = points_buf.shape[0];         // number of points
    auto dimension = points_buf.shape[1]; // dimension of each point

    py::buffer_info filter_values_buf = filter_values.request();
    if (filter_values_buf.ndim != 1) {
      throw std::runtime_error("filter data numpy array must be 1-dimensional");
    }

    if (filter_values_buf.shape[0] != n) {
      throw std::runtime_error("filter data numpy array must have the same "
                               "number of elements as the points array");
    }

    FilterType *filter_values_data =
        static_cast<FilterType *>(filter_values_buf.ptr);

    parlay::sequence<FilterType> filter_values_seq =
        parlay::sequence<FilterType>(filter_values_data,
                                     filter_values_data + n);

    auto filter_indices_sorted =
        parlay::tabulate(n, [](index_type i) { return i; });

    parlay::sort_inplace(filter_indices_sorted, [&](auto i, auto j) {
      return filter_values_seq[i] < filter_values_seq[j];
    });

    T *numpy_data = static_cast<T *>(points_buf.ptr);

    auto data_sorted = parlay::sequence<T>(n * dimension);
    auto decoding = parlay::sequence<size_t>(n, 0);

    parlay::parallel_for(0, n, [&](size_t sorted_id) {
      for (size_t d = 0; d < dimension; d++) {
        data_sorted.at(sorted_id * dimension + d) =
            numpy_data[filter_indices_sorted.at(sorted_id) * dimension + d];
      }
      decoding.at(sorted_id) = filter_indices_sorted.at(sorted_id);
    });

    auto filter_values_sorted = parlay::sequence<FilterType>(n);
    parlay::parallel_for(0, n, [&](auto i) {
      filter_values_sorted.at(i) =
          filter_values_seq.at(filter_indices_sorted.at(i));
    });

    PointRange<T, Point> point_range =
        PointRange<T, Point>(data_sorted.data(), n, dimension);

    *this = RangeFilterTreeIndex<T, Point, RangeSpatialIndex, FilterType>(
        std::make_unique<PR>(point_range), filter_values_sorted, decoding,
        cutoff, true);
  }

  /* the bounds here are inclusive */
  NeighborsAndDistances batch_filter_search(
      py::array_t<T, py::array::c_style | py::array::forcecast> &queries,
      const std::vector<std::pair<FilterType, FilterType>> &filters,
      uint64_t num_queries, QueryParams qp) {
    size_t knn = qp.k;
    py::array_t<unsigned int> ids({num_queries, knn});
    py::array_t<float> dists({num_queries, knn});

    parlay::parallel_for(0, num_queries, [&](auto i) {
      Point q = Point(queries.data(i), _points->dimension(),
                      _points->aligned_dimension(), i);
      std::pair<FilterType, FilterType> filter = filters[i];

      auto results = fenwick_tree_search(q, filter, qp);

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

  /* not really needed but just to highlight it */
  inline parlay::sequence<pid>
  self_query(const Point &query, const std::pair<FilterType, FilterType> &range,
             uint64_t knn, QueryParams qp) {
    // Ensure that qp knn is correct, in case we are using a default qp
    // TODO: Just remove knn
    qp.k = knn;
    return _spatial_indices->query(query, range, qp);
  }

  size_t first_greater_than(const FilterType &filter_value) {
    size_t start = 0;
    size_t end = _filter_values.size();
    while (start + 1 < end) {
      size_t mid = (start + end) / 2;
      if (_filter_values[mid] > filter_value) {
        end = mid;
      } else {
        start = mid;
      }
    }
    return end;
  }

  size_t first_greater_than_or_equal_to(const FilterType &filter_value) {
    size_t start = 0;
    size_t end = _filter_values.size();
    while (start + 1 < end) {
      size_t mid = (start + end) / 2;
      if (_filter_values[mid] >= filter_value) {
        end = mid;
      } else {
        start = mid;
      }
    }
    return end;
  }

private:
  std::vector<size_t> _bucket_sizes;
  std::vector<std::vector<size_t>> _starts;
  std::vector<std::vector<SpatialIndexPtr>> _spatial_indices;

  parlay::sequence<size_t> _sorted_index_to_original_point_id;

  parlay::sequence<FilterType> _filter_values;

  int32_t _cutoff = 1000;

  std::unique_ptr<PR> _points;

  RangeFilterTreeIndex(std::unique_ptr<PR> points,
                       const parlay::sequence<FilterType> &filter_values,
                       const parlay::sequence<size_t> &decoding,
                       int32_t cutoff = 1000, bool recurse = true)
      : _sorted_index_to_original_point_id(decoding), _cutoff(cutoff),
        _filter_values(filter_values),
        _points(std::move(points)) {

    auto n = _points->size();

    // TODO: Make this parallel once it works
    // TODO: Be able to pass in index location
    size_t current_filter_size = cutoff;
    while (current_filter_size < 2 * n) {
      _bucket_sizes.push_back(current_filter_size);
      size_t num_buckets = (n + current_filter_size - 1) / current_filter_size;
      _spatial_indices.push_back(std::vector<SpatialIndexPtr>(num_buckets));
      _starts.push_back(std::vector<size_t>(num_buckets));
      parlay::parallel_for(0, num_buckets, [&](auto bucket_id) {
      // for (size_t bucket_id = 0; bucket_id < num_buckets; bucket_id++) {
        size_t start = bucket_id * current_filter_size;
        auto this_filter_size = std::min(current_filter_size, n - start);
        _starts.at(_starts.size() - 1).at(bucket_id) = start;
        parlay::sequence<int32_t> subset_of_indices = parlay::tabulate<int32_t>(
            this_filter_size, [&](auto i) { return i + start; });
        SubsetRangePtr subset_points = _points->make_subset(subset_of_indices);
        parlay::sequence<FilterType> subset_of_filter_values = parlay::sequence<FilterType>(
            _filter_values.begin() + start,
            _filter_values.begin() + start + this_filter_size);
        _spatial_indices.at(_spatial_indices.size() - 1).at(bucket_id) = std::make_unique<SpatialIndex>(std::move(subset_points), subset_of_filter_values);
      });
      current_filter_size *= 2;
    }
  }


  bool check_empty(const std::pair<FilterType, FilterType> &range) {
    bool empty = range.second < _filter_values.front() ||
           range.first > _filter_values.back();
    if (empty) {
      std::cout
          << "Query range is entirely outside the index range ("
          << _filter_values.front() << ", " << _filter_values.back()
          << ") index range vs. (" << range.first << ", " << range.second
          << ") This shouldn't happen but does not directly impact correctness"
          << std::endl;
    }
    return empty;
  }

  // Returns uncoverted ids
  parlay::sequence<pid>
  fenwick_tree_search(const Point &query,
                      const std::pair<FilterType, FilterType> &range,
                      QueryParams qp) {
    // if the query range is entirely outside the index range, return
    if (check_empty(range)) {
      return parlay::sequence<pid>();
    }

    size_t knn = qp.k;

    auto inclusive_start = first_greater_than(range.first);
    auto exclusive_end = first_greater_than_or_equal_to(range.second);

    // std::cout << "QUERY " << range.first << " " << range.second << std::endl;

    std::pair<size_t, size_t> last_range = {0, 0};
    std::vector<std::pair<size_t, size_t>> indices_to_search;
    for (int64_t bucket_size_index = _bucket_sizes.size() - 1; bucket_size_index >= 0; bucket_size_index--) {

      // std::cout << "Examing " << bucket_size_index << std::endl;
        
      size_t current_bucket_size = _bucket_sizes[bucket_size_index];

      if (last_range.first == 0 && last_range.second == 0) {
        size_t min_possible_bucket_index_inclusive =
            inclusive_start / current_bucket_size;
        size_t max_possible_bucket_index_exclusive =
            exclusive_end / current_bucket_size;
        for (size_t possible_bucket_index = min_possible_bucket_index_inclusive;
             possible_bucket_index < max_possible_bucket_index_exclusive;
             possible_bucket_index++) {
          size_t bucket_start = possible_bucket_index * current_bucket_size;
          size_t bucket_end = bucket_start + current_bucket_size;
          if (bucket_start >= inclusive_start && bucket_end <= exclusive_end) {
            indices_to_search.push_back(
                std::make_pair(bucket_size_index, bucket_start / current_bucket_size));
            // std::cout << "Adding " << bucket_size_index << " " << bucket_start / current_bucket_size << std::endl;
            last_range = std::make_pair(bucket_start,
                                        bucket_start + current_bucket_size);
          }
        }
      }

      if (last_range.first == 0 && last_range.second == 0) {
        continue;
      }

      size_t left_space = last_range.first - inclusive_start;
      size_t right_space = exclusive_end - last_range.second;
      if (left_space > current_bucket_size) {
        last_range.first -= current_bucket_size;
        // std::cout << "Adding " << bucket_size_index << " " << last_range.first / current_bucket_size << std::endl;
        indices_to_search.push_back(
            std::make_pair(bucket_size_index, last_range.first / current_bucket_size));
      }
      if (right_space > current_bucket_size) {
        // std::cout << "Adding " << bucket_size_index << " " << last_range.second / current_bucket_size << std::endl;
        indices_to_search.push_back(
            std::make_pair(bucket_size_index, last_range.second / current_bucket_size));
        last_range.second += current_bucket_size;
      }
    }

    parlay::sequence<pid> frontier;
    for (auto index_pair : indices_to_search) {
      auto bucket_size_index = index_pair.first;
      auto bucket_index = index_pair.second;
      // std::cout << _bucket_sizes.at(bucket_size_index) << " " << bucket_index * _bucket_sizes.at(bucket_size_index) << std::endl;
      auto search_results = _spatial_indices.at(bucket_size_index).at(bucket_index)->query(query, range, qp);
      for (auto pid : search_results) {
        frontier.push_back(pid);
      }
    }

    if (last_range.first == 0 && last_range.second == 0) {
      for (size_t i = inclusive_start; i < exclusive_end; i++) {
        frontier.push_back({i, (*_points)[i].distance(query)});
      }
    } else {
      for (size_t i = inclusive_start; i < last_range.first; i++) {
        frontier.push_back({i, (*_points)[i].distance(query)});
      }
      for (size_t i = last_range.second; i < exclusive_end; i++) {
        frontier.push_back({i, (*_points)[i].distance(query)});
      }
    }

    parlay::sort_inplace(frontier, [&](auto a, auto b) {
      return a.second < b.second;
    });

    if (frontier.size() > knn) {
      frontier.pop_tail(frontier.size() - knn);
    }

    return frontier;
  }

  parlay::sequence<pid> optimized_postfiltering_search(
        const Point &query,
        const std::pair<FilterType, FilterType> &range,
        QueryParams qp
    ) {

        // if the query range is entirely outside the index range, return
        if (check_empty(range)) {
          return parlay::sequence<pid>();
        }

        size_t knn = qp.k;

        auto inclusive_start = first_greater_than(range.first);
        auto exclusive_end = first_greater_than_or_equal_to(range.second);

        std::pair<size_t, size_t> index_key;
        for (int64_t bucket_id = _bucket_sizes.size() - 1; bucket_id >= 0; bucket_id--) {
            size_t bucket_size = _bucket_sizes[bucket_id];
            size_t start_bucket = inclusive_start / bucket_size;
            size_t end_bucket = (exclusive_end - 1) / bucket_size;
            if (start_bucket == end_bucket) {
                index_key = {bucket_id, start_bucket};
            }
        }
 
        return _spatial_indices.at(index_key.first).at(index_key.second)->query(query, range, qp);
    }
};
