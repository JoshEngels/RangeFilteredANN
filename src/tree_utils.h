#pragma once

#include "algorithms/utils/point_range.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "pybind11/numpy.h"
#include <vector>

namespace py = pybind11;
using NeighborsAndDistances =
    std::pair<py::array_t<unsigned int>, py::array_t<float>>;

using index_type = int32_t;

// Returns the index of the filter value that is the first filter value
// greater than or equal to the passed in filter value. This will be equal to
// (_filters.size()) if the passed in value is greater than all filter values.
template <typename FilterType>
inline size_t first_greater_than_or_equal_to(
    const FilterType &filter_value,
    const parlay::sequence<FilterType> &filter_values) {
  if (filter_values[0] >= filter_value) {
    return 0;
  }
  size_t start = 0;
  size_t end = filter_values.size();
  while (start + 1 < end) {
    size_t mid = (start + end) / 2;
    if (filter_values[mid] >= filter_value) {
      end = mid;
    } else {
      start = mid;
    }
  }
  return end;
}

template <typename FilterType, typename T, typename Point>
auto sort_python_and_convert(py::array_t<T> points,
                             py::array_t<FilterType> filter_values) {

  using FilterList = parlay::sequence<FilterType>;

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

  return std::make_tuple(point_range, filter_values_sorted, decoding);
}