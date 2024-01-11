#pragma once

#include "parlay/sequence.h"

#include "algorithms/utils/point_range.h"

#include <vector>

#include "pybind11/numpy.h"

using index_type = int32_t;
using FilterType = float;

namespace py = pybind11;
using NeighborsAndDistances =
    std::pair<py::array_t<unsigned int>, py::array_t<float>>;

using pid = std::pair<index_type, float>;

/* a minimal index that does prefiltering at query time. A good faith
 * prefiltering should probably be a fenwick tree */
template <typename T, class Point, class PR = SubsetPointRange<T, Point>>
struct PrefilterIndex {
  std::unique_ptr<PR> points;
  parlay::sequence<index_type>
      indices; // the indices of the points in the original dataset
  parlay::sequence<FilterType> filter_values;
  parlay::sequence<FilterType> filter_values_sorted;
  parlay::sequence<index_type>
      filter_indices_sorted; // the indices of the points sorted by filter value

  std::pair<FilterType, FilterType> range;

  PrefilterIndex(std::unique_ptr<PR> &&points,
                 parlay::sequence<FilterType> filter_values);

  PrefilterIndex(py::array_t<T> points, py::array_t<FilterType> filter_values);

  NeighborsAndDistances batch_query(
      py::array_t<T, py::array::c_style | py::array::forcecast> &queries,
      const std::vector<std::pair<FilterType, FilterType>> &filters,
      uint64_t num_queries, uint64_t knn);

  /* processes a single query */
  parlay::sequence<pid> query(Point q, std::pair<FilterType, FilterType> filter,
                              uint64_t knn = 10);
};
