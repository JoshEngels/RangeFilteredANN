#pragma once

#include "algorithms/utils/types.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "algorithms/utils/point_range.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
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
              PrefilterIndex,
          typename FilterType = float_t>
struct RangeFilterTreeIndex {
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
  RangeFilterTreeIndex(py::array_t<T> points,
                       py::array_t<FilterType> filter_values,
                       int32_t cutoff = 1000, size_t split_factor = 2) {
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

    FilterList filter_values_seq =
        FilterList(filter_values_data, filter_values_data + n);

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

    auto filter_values_sorted = FilterList(n);
    parlay::parallel_for(0, n, [&](auto i) {
      filter_values_sorted.at(i) =
          filter_values_seq.at(filter_indices_sorted.at(i));
    });

    PointRange<T, Point> point_range =
        PointRange<T, Point>(data_sorted.data(), n, dimension);

    *this = RangeFilterTreeIndex<T, Point, RangeSpatialIndex, FilterType>(
        std::make_unique<PR>(point_range), filter_values_sorted, decoding,
        cutoff, split_factor);
  }

  /* the bounds here are inclusive */
  NeighborsAndDistances batch_search(
      py::array_t<T, py::array::c_style | py::array::forcecast> &queries,
      const std::vector<FilterRange> &filters, uint64_t num_queries,
      const std::string &query_method, QueryParams qp) {
    size_t knn = qp.k;
    py::array_t<unsigned int> ids({num_queries, knn});
    py::array_t<float> dists({num_queries, knn});

    parlay::parallel_for(0, num_queries, [&](auto i) {
      Point q = Point(queries.data(i), _points->dimension(),
                      _points->aligned_dimension(), i);
      FilterRange filter = filters[i];

      parlay::sequence<pid> results;
      if (query_method == "optimized_postfilter") {
        results = optimized_postfiltering_search(q, filter, qp);
      } else if (query_method == "three_split") {
        results = three_split_search(q, filter, qp);
      } else {
        results = fenwick_tree_search(q, filter, qp);
      }

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
  inline parlay::sequence<pid> self_query(const Point &query,
                                          const FilterRange &range,
                                          uint64_t knn, QueryParams qp) {
    // Ensure that qp knn is correct, in case we are using a default qp
    // TODO: Just remove knn
    qp.k = knn;
    return _spatial_indices->query(query, range, qp);
  }

  // Returns the index of the filter value that is the first filter value greater
  // than or equal to the passed in filter value. This will be equal to (_filters.size())
  // if the passed in value is greater than all filter values.
  size_t first_greater_than_or_equal_to(const FilterType &filter_value) {
    if (_filter_values[0] >= filter_value) {
      return 0;
    }
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
  // Inclusive starts, exclusive ends
  // Goes largest to smallest, row i contains buckets of size _bucket_offsets[1]
  // + or - 1 (but not both)
  std::vector<std::vector<size_t>> _bucket_offsets;
  std::vector<std::vector<SpatialIndexPtr>> _spatial_indices;

  parlay::sequence<size_t> _sorted_index_to_original_point_id;

  FilterList _filter_values;

  int32_t _cutoff = 1000;

  std::unique_ptr<PR> _points;

  size_t _split_factor;

  static SpatialIndexPtr create_index(FilterList &filter_values, size_t start,
                                      size_t end, PR *points) {
    auto filter_length = end - start;
    parlay::sequence<int32_t> subset_of_indices = parlay::tabulate<int32_t>(
        filter_length, [&](auto i) { return i + start; });
    SubsetRangePtr subset_points = points->make_subset(subset_of_indices);
    FilterList subset_of_filter_values =
        FilterList(filter_values.begin() + start, filter_values.begin() + end);

    return std::make_unique<SpatialIndex>(std::move(subset_points),
                                          subset_of_filter_values);
  }

  RangeFilterTreeIndex(std::unique_ptr<PR> points,
                       const FilterList &filter_values,
                       const parlay::sequence<size_t> &decoding,
                       int32_t cutoff = 1000, size_t split_factor = 2)
      : _sorted_index_to_original_point_id(decoding), _cutoff(cutoff),
        _filter_values(filter_values), _points(std::move(points)),
        _split_factor(split_factor) {

    auto n = _points->size();

    _spatial_indices.push_back(std::vector<SpatialIndexPtr>(1));
    _spatial_indices.at(0).at(0) =
        create_index(_filter_values, 0, _filter_values.size(), _points.get());
    _bucket_offsets.push_back({0, _filter_values.size()});

    // TODO: Parallelize the outer loop?
    // TODO: Be able to pass in index location

    while (_bucket_offsets.back().at(1) > cutoff) {

      auto last_num_buckets = _spatial_indices.back().size();
      _bucket_offsets.push_back(
          std::vector<size_t>(last_num_buckets * _split_factor + 1));
      _bucket_offsets.back().back() = _filter_values.size();
      _spatial_indices.push_back(
          std::vector<SpatialIndexPtr>(last_num_buckets * _split_factor));

      parlay::parallel_for(0, last_num_buckets, [&](auto last_bucket_id) {
        auto last_start =
            _bucket_offsets.at(_bucket_offsets.size() - 2).at(last_bucket_id);
        auto last_end = _bucket_offsets.at(_bucket_offsets.size() - 2)
                            .at(last_bucket_id + 1);
        auto last_size = last_end - last_start;

        auto large_bucket_size =
            (last_size + _split_factor - 1) / _split_factor;
        auto small_bucket_size = large_bucket_size - 1;
        auto num_larger_buckets = last_size - small_bucket_size * _split_factor;

        parlay::parallel_for(0, num_larger_buckets, [&](auto i) {
          auto start = last_start + i * large_bucket_size;
          auto end = start + large_bucket_size;
          _bucket_offsets.back().at(last_bucket_id * _split_factor + i) = start;
          _spatial_indices.back().at(last_bucket_id * _split_factor + i) =
              create_index(_filter_values, start, end, _points.get());
          // std::cout << "start = " << start << ", end = " << end << std::endl;
        });

        parlay::parallel_for(num_larger_buckets, _split_factor, [&](auto i) {
          auto start = last_start + num_larger_buckets * large_bucket_size +
                       (i - num_larger_buckets) * small_bucket_size;
          auto end = start + small_bucket_size;
          _bucket_offsets.back().at(last_bucket_id * _split_factor + i) = start;
          _spatial_indices.back().at(last_bucket_id * _split_factor + i) =
              create_index(_filter_values, start, end, _points.get());
          // std::cout << "start = " << start << ", end = " << end << std::endl;
        });
      });
    }
  }

  bool check_empty(const FilterRange &range) {
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

  parlay::sequence<pid> three_split_search(const Point &query,
                                           const FilterRange &filter_range,
                                           QueryParams qp) {
    return {};
  }

  struct SequentialBuckets {
    size_t bucket_row;
    size_t bucket_start_index;
    size_t bucket_end_index;
    size_t start_filter_cover;
    size_t end_filter_cover;
  };

  size_t find_range_containing_index(size_t bucket_row, size_t index) {
    size_t left = 0;
    size_t right = _bucket_offsets[bucket_row].size() - 1;

    while (left < right) {
      size_t mid = (left + right) / 2;

      if (index >= _bucket_offsets[bucket_row][mid] &&
          index < _bucket_offsets[bucket_row][mid + 1]) {
        return mid;
      } else if (index < _bucket_offsets[bucket_row][mid]) {
        right = mid;
      } else {
        left = mid;
      }
    }

    throw std::runtime_error(
        "This should not be possible if index is within the filter range");
  }

  std::optional<SequentialBuckets>
  find_largest_ranges_within_query_range(size_t inclusive_start,
                                         size_t exclusive_end) {
    size_t range_size = exclusive_end - inclusive_start;

    // First find row where first bucket is smaller than range
    std::optional<size_t> first_possible_bucket_row = std::nullopt;
    for (size_t bucket_row = 0; bucket_row < _bucket_offsets.size();
         bucket_row++) {
      auto bucket_end = _bucket_offsets[bucket_row][1];
      // Subtract one because bucket size may possibly be one smaller than first
      // bucket
      auto bucket_size =
          _bucket_offsets[bucket_row][1] - _bucket_offsets[bucket_row][0] - 1;
      if (bucket_size <= range_size) {
        first_possible_bucket_row = bucket_row;
        break;
      }
    }

    if (!first_possible_bucket_row.has_value()) {
      return std::nullopt;
    }

    auto current_row = first_possible_bucket_row.value();

    // Find first range that does not contain inclusive_start - 1 (or 0 if
    // inclusive_start == 0)
    auto first_range_index =
        inclusive_start == 0
            ? 0
            : find_range_containing_index(current_row, inclusive_start - 1) + 1;
    auto start = _bucket_offsets.at(current_row).at(first_range_index);
    auto end = _bucket_offsets.at(current_row).at(first_range_index + 1);
    if (end > exclusive_end) {
      current_row += 1;
      first_range_index =
          inclusive_start == 0
              ? 0
              : find_range_containing_index(current_row, inclusive_start - 1) +
                    1;
      start = _bucket_offsets.at(current_row).at(first_range_index);
      end = _bucket_offsets.at(current_row).at(first_range_index + 1);
    }

    auto last_range_index = first_range_index + 1;
    while (last_range_index < _bucket_offsets.at(current_row).size() - 1) {
      auto next_end = _bucket_offsets.at(current_row).at(last_range_index + 1);
      if (next_end > exclusive_end) {
        break;
      }
      last_range_index++;
      end = next_end;
    }

    return SequentialBuckets{current_row, first_range_index, last_range_index,
                             start, end};
  }

  parlay::sequence<pid> fenwick_tree_search(const Point &query,
                                            const FilterRange &range,
                                            QueryParams qp) {
    if (check_empty(range)) {
      return parlay::sequence<pid>();
    }

    size_t knn = qp.k;

    auto inclusive_start = first_greater_than_or_equal_to(range.first);
    auto exclusive_end = first_greater_than_or_equal_to(range.second);

    auto center_ranges_opt =
        find_largest_ranges_within_query_range(inclusive_start, exclusive_end);

    auto ranges_to_search = std::vector<std::pair<size_t, size_t>>(0);
    std::optional<size_t> cover_inclusive_start,
        cover_exclusive_end = std::nullopt;

    if (center_ranges_opt.has_value()) {
      SequentialBuckets center_range = center_ranges_opt.value();
      for (size_t bucket_index = center_range.bucket_start_index;
           bucket_index < center_range.bucket_end_index; bucket_index++) {
        ranges_to_search.push_back(
            std::make_pair(center_range.bucket_row, bucket_index));
      }

      cover_inclusive_start = center_range.start_filter_cover;
      cover_exclusive_end = center_range.end_filter_cover;
      size_t last_included_left_index = center_range.bucket_start_index;
      size_t last_included_right_index = center_range.bucket_end_index - 1;
      for (size_t bucket_row = center_range.bucket_row + 1;
           bucket_row < _bucket_offsets.size(); bucket_row++) {
        last_included_left_index *= _split_factor;
        last_included_right_index *= _split_factor;
        last_included_right_index += _split_factor - 1;

        while (last_included_left_index > 0) {
          auto next_left_bucket_start =
              _bucket_offsets.at(bucket_row).at(last_included_left_index - 1);
          if (next_left_bucket_start < inclusive_start) {
            break;
          }
          cover_inclusive_start = next_left_bucket_start;
          last_included_left_index -= 1;
          ranges_to_search.push_back(
              std::make_pair(bucket_row, last_included_left_index));
        }

        while (last_included_right_index <
               _bucket_offsets[bucket_row].size() - 2) {
          auto next_right_bucket_end =
              _bucket_offsets.at(bucket_row).at(last_included_right_index + 2);
          if (next_right_bucket_end > exclusive_end) {
            break;
          }
          cover_exclusive_end = next_right_bucket_end;
          last_included_right_index += 1;
          ranges_to_search.push_back(
              std::make_pair(bucket_row, last_included_right_index));
        }
      }
    }

    if (qp.verbose) {
      std::cout << "Query range: " << inclusive_start << " " << exclusive_end
                << std::endl;
    }
    parlay::sequence<pid> frontier;
    for (auto index_pair : ranges_to_search) {
      auto bucket_row_index = index_pair.first;
      auto bucket_index = index_pair.second;
      if (qp.verbose) {
        std::cout << "Searching bucket: "
                  << _bucket_offsets.at(bucket_row_index).at(bucket_index)
                  << " "
                  << _bucket_offsets.at(bucket_row_index).at(bucket_index + 1)
                  << std::endl;
      }
      auto search_results = _spatial_indices.at(bucket_row_index)
                                .at(bucket_index)
                                ->query(query, range, qp);
      for (auto pid : search_results) {
        frontier.push_back(pid);
      }
    }

    if (cover_inclusive_start.has_value() && cover_exclusive_end.has_value()) {
      for (size_t i = inclusive_start; i < *cover_inclusive_start; i++) {
        frontier.push_back({i, (*_points)[i].distance(query)});
      }
      for (size_t i = *cover_exclusive_end; i < exclusive_end; i++) {
        frontier.push_back({i, (*_points)[i].distance(query)});
      }
    } else {
      for (size_t i = inclusive_start; i < exclusive_end; i++) {
        frontier.push_back({i, (*_points)[i].distance(query)});
      }
    }

    parlay::sort_inplace(frontier,
                         [&](auto a, auto b) { return a.second < b.second; });

    if (frontier.size() > knn) {
      frontier.pop_tail(frontier.size() - knn);
    }

    return frontier;
  }

  parlay::sequence<pid> optimized_postfiltering_search(const Point &query,
                                                       const FilterRange &range,
                                                       QueryParams qp) {

    // if the query range is entirely outside the index range, return
    if (check_empty(range)) {
      return parlay::sequence<pid>();
    }

    size_t knn = qp.k;

    auto inclusive_start = first_greater_than_or_equal_to(range.first);
    auto exclusive_end = first_greater_than_or_equal_to(range.second);

    if (4 * (exclusive_end - inclusive_start) < _cutoff) {
      return fenwick_tree_search(query, range, qp);
    }

    size_t current_row = 0;
    size_t current_index = 0;

    while (current_row < _bucket_offsets.size()) {
      size_t next_row = current_row + 1;
      std::optional<size_t> next_working_index = std::nullopt;
      for (size_t possible_next_index = current_index * _split_factor;
           possible_next_index < current_index * _split_factor + _split_factor;
           possible_next_index++) {
        if (possible_next_index == _spatial_indices.at(next_row).size()) {
          break;
        }
        auto next_start = _bucket_offsets.at(next_row).at(possible_next_index);
        auto next_end =
            _bucket_offsets.at(next_row).at(possible_next_index + 1);
        if (inclusive_start >= next_start && exclusive_end <= next_end) {
          next_working_index = possible_next_index;
        }
      }
      if (!next_working_index.has_value()) {
        break;
      }
      current_index = next_working_index.value();
      current_row = next_row;
    }


    auto bucket_start = _bucket_offsets.at(current_row).at(current_index);
    auto bucket_end = _bucket_offsets.at(current_row).at(current_index + 1);
    auto bucket_size = bucket_end - bucket_start;

    if (qp.verbose) {
      std::cout << "Query range = (" << inclusive_start << ","
                << exclusive_end << "), smallest containing range (size "
                << bucket_size << ") = (" << bucket_start
                << "," << bucket_end << ")"
                << std::endl;
    }

    float bucket_size_to_query_size_ratio =
        (float)bucket_size / (exclusive_end - inclusive_start);
    if (qp.min_query_to_bucket_ratio.has_value() &&
        bucket_size_to_query_size_ratio >
            qp.min_query_to_bucket_ratio.value()) {
      return fenwick_tree_search(query, range, qp);
    }

    return _spatial_indices.at(current_row)
        .at(current_index)
        ->query(query, range, qp);
  }

  // parlay::sequence<pid> three_split_search(const Point &query,
  //                                          const FilterRange &filter_range,
  //                                          QueryParams qp) {

  //   auto inclusive_start = first_greater_than(filter_range.first);
  //   auto exclusive_end = first_greater_than_or_equal_to(filter_range.second);

  //   std::optional<std::pair<size_t, size_t>> index_to_search = std::nullopt;

  //   for (int64_t bucket_size_index = _bucket_sizes.size() - 1;
  //        bucket_size_index >= 0; bucket_size_index--) {

  //     size_t current_bucket_size = _bucket_sizes[bucket_size_index];

  //     size_t min_possible_bucket_index_inclusive =
  //         inclusive_start / current_bucket_size;
  //     size_t max_possible_bucket_index_exclusive =
  //         exclusive_end / current_bucket_size;

  //     for (size_t possible_bucket_index =
  //     min_possible_bucket_index_inclusive;
  //          possible_bucket_index < max_possible_bucket_index_exclusive;
  //          possible_bucket_index++) {
  //       size_t bucket_start = possible_bucket_index * current_bucket_size;
  //       size_t bucket_end =
  //           std::min(bucket_start + current_bucket_size, _points->size());
  //       if (bucket_start >= inclusive_start && bucket_end <= exclusive_end) {
  //         index_to_search =
  //             std::make_pair(bucket_size_index, possible_bucket_index);
  //         goto loop_done;
  //       }
  //     }
  //   }

  // loop_done:

  //   if (!index_to_search.has_value()) {
  //     return fenwick_tree_search(query, filter_range, qp);
  //   }

  //   std::vector<parlay::sequence<pid>> frontiers;
  //   auto bucket_size_index = index_to_search->first;
  //   auto bucket_index = index_to_search->second;
  //   // Force final_beam_multiply to be 1
  //   QueryParams qp_copy = {qp.k,
  //                          qp.beamSize,
  //                          qp.cut,
  //                          qp.limit,
  //                          qp.degree_limit,
  //                          1,
  //                          qp.postfiltering_max_beam,
  //                          qp.min_query_to_bucket_ratio,
  //                          qp.verbose};

  //   // std::cout << "Center" << std::endl;
  //   frontiers.push_back(_spatial_indices.at(bucket_size_index)
  //                           .at(bucket_index)
  //                           ->query(query, filter_range, qp_copy));

  //   auto middle_start = bucket_index * _bucket_sizes[bucket_size_index];
  //   auto middle_end = std::min(middle_start +
  //   _bucket_sizes[bucket_size_index],
  //                              _points->size());
  //   size_t left_space = middle_start - inclusive_start;
  //   size_t right_space = exclusive_end - middle_end;

  //   // std::cout << "Left" << std::endl;
  //   // std::cout << inclusive_start << " " << middle_start << std::endl;
  //   if (left_space > 0) {
  //     FilterRange left_filter_range =
  //         std::make_pair(filter_range.first, _filter_values[middle_start]);
  //     frontiers.push_back(
  //         optimized_postfiltering_search(query, left_filter_range, qp));
  //   }

  //   // std::cout << "Right" << std::endl;
  //   // std::cout << middle_end << " " << exclusive_end << std::endl;
  //   if (right_space > 0) {
  //     FilterRange right_filter_range =
  //         std::make_pair(_filter_values[middle_end], filter_range.second);
  //     frontiers.push_back(
  //         optimized_postfiltering_search(query, right_filter_range, qp));
  //   }

  //   parlay::sequence<pid> frontier;

  //   for (auto part_frontier : frontiers) {
  //     for (auto pid : part_frontier) {
  //       frontier.push_back(pid);
  //     }
  //   }

  //   parlay::sort_inplace(frontier,
  //                        [&](auto a, auto b) { return a.second < b.second;
  //                        });

  //   if (frontier.size() > qp.k) {
  //     frontier.pop_tail(frontier.size() - qp.k);
  //   }

  //   return frontier;
  // }
};
