#pragma once

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "algorithms/utils/point_range.h"

#include <algorithm>
#include <limits>
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

  auto n = buf.shape[0];    // number of points
  auto dimension = buf.shape[1]; // dimension of each point

  T *numpy_data = static_cast<T *>(buf.ptr);

  return std::move(PointRange<T, Point>(numpy_data, n, dimension));
}

template <typename T, typename Point,
          template <typename, typename, typename> class RangeSpatialIndex =
              PrefilterIndex,
          class PR = PointRange<T, Point>, typename FilterType = float_t>
struct RangeFilterTreeIndex {
  using pid = std::pair<index_type, float>;

  // because the spatial index is responsible for the points, we do not store
  // them here
  std::unique_ptr<RangeSpatialIndex<T, Point, PR>> spatial_index;

  bool has_children = false;
  std::pair<
      std::unique_ptr<RangeFilterTreeIndex<
          T, Point, RangeSpatialIndex, SubsetPointRange<T, Point>, FilterType>>,
      std::unique_ptr<RangeFilterTreeIndex<
          T, Point, RangeSpatialIndex, SubsetPointRange<T, Point>, FilterType>>>
      children;

  std::pair<FilterType, FilterType> range; // min and max filter values
  std::pair<FilterType, FilterType>
      median; // median filter values, delineating the range between the two
              // children

  parlay::sequence<size_t> sorted_index_to_original_point_id;

  int32_t cutoff = 1000;

  size_t n;

  RangeFilterTreeIndex(std::unique_ptr<PR> points,
                       const parlay::sequence<FilterType> &filter_values,
                       const parlay::sequence<size_t> &decoding,
                       int32_t cutoff = 1000, bool recurse = true)
      : spatial_index(std::make_unique<RangeSpatialIndex<T, Point, PR>>(
            std::move(points), filter_values)),
        sorted_index_to_original_point_id(decoding),
        cutoff(cutoff) {

    n = this->spatial_index->points->size();

    // get the min and max filter values
    range = this->spatial_index->range;

    // get the median filter value
    // median = this->spatial_index->median;

    if (recurse) {
      build_children_recursive();
    }
  }

  RangeFilterTreeIndex(py::array_t<T> points,
                       py::array_t<FilterType> filter_values,
                       int32_t cutoff = 1000) {
    py::buffer_info points_buf = points.request();
    if (points_buf.ndim != 2) {
      throw std::runtime_error("points numpy array must be 2-dimensional");
    }
    n = points_buf.shape[0];         // number of points
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

    auto filter_indices_sorted = parlay::tabulate(n, [](index_type i) { return i; });

    parlay::sort_inplace(filter_indices_sorted, [&](auto i, auto j) {
      return filter_values_seq[i] < filter_values_seq[j];
    });

    T *numpy_data = static_cast<T *>(points_buf.ptr);

    auto data_sorted = parlay::sequence<T>(n * dimension);
    auto decoding = parlay::sequence<size_t>(n, 0);

    parlay::parallel_for(0, n, [&](size_t sorted_id) { 
      for (size_t d = 0; d < dimension; d++) {
        data_sorted.at(sorted_id * dimension + d) = numpy_data[filter_indices_sorted.at(sorted_id) * dimension + d];
      }
      decoding.at(sorted_id) = filter_indices_sorted.at(sorted_id);
    });

    auto filter_values_sorted = parlay::sequence<FilterType>(n);
    parlay::parallel_for(0, n, [&](auto i) {
      filter_values_sorted.at(i) = filter_values_seq.at(filter_indices_sorted.at(i));
    });

    PointRange<T, Point> point_range =
        PointRange<T, Point>(data_sorted.data(), n, dimension);

    *this = RangeFilterTreeIndex<T, Point, RangeSpatialIndex, PR, FilterType>(
        std::make_unique<PR>(point_range), filter_values_sorted, decoding, cutoff, true);
  }

  void build_children() {
    if (has_children) {
      throw std::runtime_error(
          "Cannot build children of a node that already has children");
      return;
    }

    // if we were being deliberate, we would split on the first occurence of the
    // median value
    auto n1 = n / 2;
    auto n2 = n - n1;

    auto &points = this->spatial_index->points;
    auto &indices = this->spatial_index->indices;

    std::unique_ptr<SubsetPointRange<T, Point>> points1 = points->make_subset(
        parlay::map(parlay::make_slice(indices.begin(), indices.begin() + n1),
                    [&](auto i) { return i; }));
    std::unique_ptr<SubsetPointRange<T, Point>> points2 = points->make_subset(
        parlay::map(parlay::make_slice(indices.begin() + n1, indices.end()),
                    [&](auto i) { return i; }));

    auto filter_values1 = parlay::sequence<FilterType>(
        this->spatial_index->filter_values.begin(),
        this->spatial_index->filter_values.begin() + n1);
    auto filter_values2 = parlay::sequence<FilterType>(
        this->spatial_index->filter_values.begin() + n1,
        this->spatial_index->filter_values.end());

    auto sorted_index_to_original_point_id1 = parlay::sequence<size_t>(
        sorted_index_to_original_point_id.begin(),
        sorted_index_to_original_point_id.begin() + n1);
    auto sorted_index_to_original_point_id2 = parlay::sequence<size_t>(
        sorted_index_to_original_point_id.begin() + n1,
        sorted_index_to_original_point_id.end());

    std::unique_ptr<RangeFilterTreeIndex<
        T, Point, RangeSpatialIndex, SubsetPointRange<T, Point>, FilterType>>
        index1 = std::make_unique<
            RangeFilterTreeIndex<T, Point, RangeSpatialIndex,
                                 SubsetPointRange<T, Point>, FilterType>>(
            std::move(points1), filter_values1, sorted_index_to_original_point_id1, this->cutoff, false);

    std::unique_ptr<RangeFilterTreeIndex<
        T, Point, RangeSpatialIndex, SubsetPointRange<T, Point>, FilterType>>
        index2 = std::make_unique<
            RangeFilterTreeIndex<T, Point, RangeSpatialIndex,
                                 SubsetPointRange<T, Point>, FilterType>>(
            std::move(points2), filter_values2, sorted_index_to_original_point_id2, this->cutoff, false);

    this->children = std::make_pair(std::move(index1), std::move(index2));

    this->has_children = true;

    this->median = std::make_pair(this->children.first->range.second,
                                  this->children.second->range.first);
  }

  void build_children_recursive() {
    if (has_children) {
      throw std::runtime_error(
          "Cannot build children of a node that already has children");
      return;
    }

    if (n <= cutoff * 2) {
      return;
    }

    build_children();

    parlay::par_do([&] { this->children.first->build_children_recursive(); },
                   [&] { this->children.second->build_children_recursive(); });
  }

  /* the bounds here are inclusive */
  NeighborsAndDistances batch_filter_search(
      py::array_t<T, py::array::c_style | py::array::forcecast> &queries,
      const std::vector<std::pair<FilterType, FilterType>> &filters,
      uint64_t num_queries, uint64_t knn) {
    py::array_t<unsigned int> ids({num_queries, knn});
    py::array_t<float> dists({num_queries, knn});

    parlay::parallel_for(0, num_queries, [&](auto i) {
      Point q = Point(queries.data(i), this->spatial_index->points->dimension(),
                      this->spatial_index->points->aligned_dimension(), i);
      std::pair<FilterType, FilterType> filter = filters[i];

      auto results = orig_serial_query(q, filter, knn);

      for (auto j = 0; j < knn; j++) {
        ids.mutable_at(i, j) = sorted_index_to_original_point_id.at(results[j].first);
        dists.mutable_at(i, j) = results[j].second;
      }
    });
    return std::make_pair(ids, dists);
  }

  /* not really needed but just to highlight it */
  inline parlay::sequence<pid>
  self_query(const Point &query, const std::pair<FilterType, FilterType> &range,
             uint64_t knn) {
    return spatial_index->query(query, range);
  }

  parlay::sequence<pid>
  orig_serial_query(const Point &query,
                    const std::pair<FilterType, FilterType> &range,
                    uint64_t knn) {
    // if the query range is entirely outside the index range, return
    if (range.second < this->range.first || range.first > this->range.second) {
      std::cout
          << "Query range is entirely outside the index range ("
          << this->range.first << ", " << this->range.second
          << ") index range vs. (" << range.first << ", " << range.second
          << ") This shouldn't happen but does not directly impact correctness"
          << std::endl;
      return parlay::sequence<pid>();
    }

    parlay::sequence<pid> frontier;

    // if there are no children, search the elements within the target range
    if (!has_children || (range.first <= this->range.first &&
                          range.second >= this->range.second)) {
      frontier = self_query(query, range, knn);
    } else {
      // recurse on the children
      auto &[index1, index2] = this->children;
      parlay::sequence<pid> results1, results2;

      if (range.first <= median.first) {
        results1 = index1->orig_serial_query(query, range, knn);
      }

      if (range.second >= median.second) {
        results2 = index2->orig_serial_query(query, range, knn);
      }

      if (results1.size() == 0) {
        frontier = results2;
      } else if (results2.size() == 0) {
        frontier = results1;
      } else {
        // this is pretty lazy and inefficient
        // frontier = parlay::merge(results1, results2, [&](auto a, auto b) {
        //     return a.second < b.second;
        // });
        frontier = results1;
        for (pid p : results2) {
          frontier.push_back(p);
        }
        parlay::sort_inplace(
            frontier, [&](auto a, auto b) { return a.second < b.second; });

        if (frontier.size() > knn) {
          // resize is probably the right thing here but not terribly well
          // documented
          frontier.pop_tail(frontier.size() - knn);
        }
      }
    }

    return frontier;
  }
};
