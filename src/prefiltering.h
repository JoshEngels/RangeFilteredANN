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
                 parlay::sequence<FilterType> filter_values)
      : points(std::move(points)), filter_values(std::move(filter_values)) {
    auto n = this->points->size();

    if constexpr (std::is_same<PR, PointRange<T, Point>>()) {
      indices = parlay::tabulate(n, [](int32_t i) { return i; });
    } else {
      indices = parlay::tabulate(
          n, [&](int32_t i) { return this->points->subset[i]; });
    }

    filter_values_sorted = parlay::sequence<FilterType>(n);
    filter_indices_sorted = parlay::tabulate(n, [](index_type i) { return i; });

    // argsort the filter values to get sorted indices
    parlay::sort_inplace(filter_indices_sorted, [&](auto i, auto j) {
      return this->filter_values[i] < this->filter_values[j];
    });

    // sort the filter values
    parlay::parallel_for(0, n, [&](auto i) {
      filter_values_sorted[i] = this->filter_values[filter_indices_sorted[i]];
    });

    range =
        std::make_pair(filter_values_sorted[0], filter_values_sorted[n - 1]);
  }

  PrefilterIndex(py::array_t<T> points, py::array_t<FilterType> filter_values) {
    py::buffer_info points_buf = points.request();
    if (points_buf.ndim != 2) {
      throw std::runtime_error("points numpy array must be 2-dimensional");
    }
    auto n = points_buf.shape[0];    // number of points
    auto dims = points_buf.shape[1]; // dimension of each point

    // avoiding this copy may have dire consequences from gc
    T *numpy_data = static_cast<T *>(points_buf.ptr);

    this->points = std::make_unique<PR>(numpy_data, n, dims);

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

    this->filter_values = parlay::sequence<FilterType>(filter_values_data,
                                                       filter_values_data + n);

    indices = parlay::tabulate(n, [](int32_t i) { return i; });
    filter_values_sorted = parlay::sequence<FilterType>(n);
    filter_indices_sorted = parlay::tabulate(n, [](index_type i) { return i; });

    // argsort the filter values to get sorted indices
    parlay::sort_inplace(filter_indices_sorted, [&](auto i, auto j) {
      return this->filter_values[i] < this->filter_values[j];
    });

    // sort the filter values
    parlay::parallel_for(0, n, [&](auto i) {
      filter_values_sorted[i] = this->filter_values[filter_indices_sorted[i]];
    });

    range =
        std::make_pair(filter_values_sorted[0], filter_values_sorted[n - 1]);
  }

  NeighborsAndDistances batch_search(
      py::array_t<T, py::array::c_style | py::array::forcecast> &queries,
      const std::vector<std::pair<FilterType, FilterType>> &filters,
      uint64_t num_queries, uint64_t knn) {
    py::array_t<unsigned int> ids({num_queries, knn});
    py::array_t<float> dists({num_queries, knn});

    parlay::parallel_for(0, num_queries, [&](auto i) {
      Point q = Point(queries.data(i), this->points->dimension(),
                      this->points->aligned_dimension(), i);
      std::pair<FilterType, FilterType> filter = filters[i];

      // hopefully I can trust these results
      size_t start;

      size_t l, r, mid;
      l = 0;
      r = filter_values_sorted.size() - 1;
      while (l < r) {
        mid = (l + r) / 2;
        if (filter_values_sorted[mid] < filter.first) {
          l = mid + 1;
        } else {
          r = mid;
        }
      }
      start = l;

      size_t end;

      l = 0;
      r = filter_values_sorted.size() - 1;
      while (l < r) {
        mid = (l + r) / 2;
        if (filter_values_sorted[mid] < filter.second) {
          l = mid + 1;
        } else {
          r = mid;
        }
      }
      end = l;

      auto frontier = parlay::sequence<std::pair<index_type, float>>(
          knn, std::make_pair(-1, std::numeric_limits<float>::max()));

      for (auto j = start; j < end; j++) {
        index_type index = filter_indices_sorted[j];
        Point p = (*points)[index];
        float dist = q.distance(p);
        if (dist < frontier[knn - 1].second) {
          frontier[knn - 1] = std::make_pair(indices[index], dist);
          parlay::sort_inplace(
              frontier, [&](auto a, auto b) { return a.second < b.second; });
        }
      }

      for (auto j = 0; j < knn; j++) {
        ids.mutable_at(i, j) = frontier[j].first;
        dists.mutable_at(i, j) = frontier[j].second;
      }
    });

    return std::make_pair(ids, dists);
  }

  parlay::sequence<pid> query(Point q, std::pair<FilterType, FilterType> filter,
                              QueryParams qp) {
    return query_knn(q, filter, qp.k);
  }

  /* processes a single query */
  parlay::sequence<pid> query_knn(Point q,
                                  std::pair<FilterType, FilterType> filter,
                                  uint64_t knn = 10) {
    size_t start;

    size_t l, r, mid;
    l = 0;
    r = filter_values_sorted.size() - 1;
    while (l < r) {
      mid = (l + r) / 2;
      if (filter_values_sorted[mid] < filter.first) {
        l = mid + 1;
      } else {
        r = mid;
      }
    }
    start = l;

    size_t end;

    l = 0;
    r = filter_values_sorted.size() - 1;
    while (l < r) {
      mid = (l + r) / 2;
      if (filter_values_sorted[mid] < filter.second) {
        l = mid + 1;
      } else {
        r = mid;
      }
    }
    end = l;

    auto frontier = parlay::sequence<std::pair<index_type, float>>(
        knn, std::make_pair(-1, std::numeric_limits<float>::max()));

    for (auto j = start; j < end; j++) {
      index_type index = filter_indices_sorted[j];
      Point p = (*points)[index];
      float dist = q.distance(p);
      if (dist < frontier[knn - 1].second) {
        frontier[knn - 1] = std::make_pair(indices[index], dist);
        parlay::sort_inplace(
            frontier, [&](auto a, auto b) { return a.second < b.second; });
      }
    }

    return frontier;
  }
};
