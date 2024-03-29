#pragma once

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "algorithms/utils/graph.h"
#include "algorithms/utils/point_range.h"
#include "algorithms/utils/types.h"

#include "algorithms/vamana/index.h"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <type_traits>
#include <vector>

#include "pybind11/numpy.h"

#include "prefiltering.h"

using index_type = int32_t;

namespace py = pybind11;
using NeighborsAndDistances =
    std::pair<py::array_t<unsigned int>, py::array_t<float>>;

template <typename T, typename Point, class PR = PointRange<T, Point>,
          typename FilterType = float_t>
struct PostfilterVamanaIndex {
  using pid = std::pair<index_type, float>;

  std::shared_ptr<PR> points;
  Graph<index_type> G;
  BuildParams build_params;

  parlay::sequence<FilterType> filter_values;

  std::pair<FilterType, FilterType> range;

  parlay::sequence<index_type> indices;

  PostfilterVamanaIndex(std::shared_ptr<PR> &&points,
                        parlay::sequence<FilterType> filter_values,
                        BuildParams build_params)
      : points(std::move(points)), filter_values(filter_values),
        build_params(build_params) {

    this->range = std::make_pair(
        *(std::min_element(filter_values.begin(), filter_values.end())),
        *(std::max_element(filter_values.begin(), filter_values.end())));

    const auto &cache_path = build_params.cache_path;
    if (cache_path != "" &&
        std::filesystem::exists(this->graph_filename(cache_path))) {
      std::cout << "Loading graph from " << this->graph_filename(cache_path)
                << std::endl;

      std::string filename = this->graph_filename(cache_path);
      this->G = Graph<index_type>(filename.data());
    } else {
      // std::cout << "Building graph" << std::endl;
      // this->start_point = indices[0];
      knn_index<Point, PR, index_type> I(build_params);
      stats<index_type> BuildStats(this->points->size());

      // std::cout << "This filter has " << indices.size() << " points" <<
      // std::endl;

      this->G = Graph<index_type>(build_params.R, this->points->size());
      I.build_index(this->G, *(this->points), BuildStats);

      if (cache_path != "") {
        this->save_graph(cache_path);
        std::cout << "Graph built, saved to " << graph_filename(cache_path)
                  << std::endl;
      }
    }

    if constexpr (std::is_same<PR, PointRange<T, Point>>::value) {
      this->indices = parlay::tabulate(this->points->size(),
                                       [&](index_type i) { return i; });
    } else {
      this->indices = parlay::tabulate(this->points->size(), [&](index_type i) {
        return this->points->subset[i];
      });
    }
  }

  PostfilterVamanaIndex(py::array_t<T> points,
                        py::array_t<FilterType> filter_values,
                        BuildParams build_params) {
    py::buffer_info points_buf = points.request();
    if (points_buf.ndim != 2) {
      throw std::runtime_error("points numpy array must be 2-dimensional");
    }
    auto n = points_buf.shape[0];    // number of points
    auto dims = points_buf.shape[1]; // dimension of each point

    // avoiding this copy may have dire consequences from gc
    T *numpy_data = static_cast<T *>(points_buf.ptr);

    auto tmp_points = std::make_shared<PR>(numpy_data, n, dims);

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

    auto tmp_filter_values = parlay::sequence<FilterType>(
        filter_values_data, filter_values_data + n);

    *this = PostfilterVamanaIndex(std::move(tmp_points),
                                  std::move(tmp_filter_values), build_params);
  }

  std::string graph_filename(std::string cache_path) {
    return cache_path + "vamana_" + std::to_string(build_params.L) + "_" +
           std::to_string(build_params.R) + "_" +
           std::to_string(build_params.alpha) + "_" +
           std::to_string(range.first) + "_" + std::to_string(range.second) +
           "_" + std::to_string(points->size()) + ".bin";
  }

  void save_graph(std::string filename_prefix) {
    std::string filename = this->graph_filename(filename_prefix);

    this->G.save(filename.data());
  }

  // Does a postfiltering query on the underlying index
  parlay::sequence<pid> query(const Point &q,
                              const std::pair<FilterType, FilterType> filter,
                              QueryParams query_params) {
    size_t knn = query_params.k;
    QueryParams actual_params = {query_params.beamSize,
                                 query_params.beamSize,
                                 query_params.cut,
                                 query_params.limit,
                                 query_params.degree_limit,
                                 query_params.final_beam_multiply,
                                 query_params.postfiltering_max_beam,
                                 query_params.min_query_to_bucket_ratio,
                                 query_params.verbose};
    parlay::sequence<pid> frontier = {};
    if (query_params.verbose) {
      std::cout << "Starting optimized postfiltering, beam size = "
                << actual_params.beamSize << ", k = " << knn
                << ", final multiply = " << query_params.final_beam_multiply
                << ", n = " << filter_values.size() << std::endl;
    }
    while (frontier.size() < knn &&
           actual_params.beamSize < query_params.postfiltering_max_beam) {
      frontier = this->raw_query(q, filter, actual_params);
      if (query_params.verbose) {
        std::cout << "Finished a double, frontier size = " << frontier.size()
                  << ", beam size = " << actual_params.beamSize << std::endl;
      }
      if (frontier.size() < knn) {
        actual_params.beamSize *= 2;
        actual_params.k = actual_params.beamSize;
      }
    }
    size_t final_beam_size = std::min<size_t>(
        actual_params.beamSize * query_params.final_beam_multiply,
        query_params.postfiltering_max_beam);

    if (final_beam_size > actual_params.beamSize) {
      actual_params.beamSize = final_beam_size;
      actual_params.k = final_beam_size;
      frontier = this->raw_query(q, filter, actual_params);
    }
    if (query_params.verbose) {
      std::cout << "Final frontier size = " << frontier.size()
                << ", final beam size " << actual_params.beamSize << std::endl;
    }

    return frontier;
  }

  // Does a batch of doubling postfiltering queries on the underlying index
  NeighborsAndDistances batch_search(
      py::array_t<T, py::array::c_style | py::array::forcecast> &queries,
      const std::vector<std::pair<FilterType, FilterType>> &filters,
      uint64_t num_queries, QueryParams query_params) {

    size_t knn = query_params.k;

    py::array_t<unsigned int> ids({num_queries, knn});
    py::array_t<float> dists({num_queries, knn});

    parlay::parallel_for(0, num_queries, [&](size_t i) {
      Point q = Point(queries.data(i), points->dimension(),
                      points->aligned_dimension(), i);

      auto frontier = query(q, filters.at(i), query_params);

      for (auto j = 0; j < knn; j++) {
        if (j < frontier.size()) {
          ids.mutable_at(i, j) = frontier[j].first;
          dists.mutable_at(i, j) = frontier[j].second;
        } else {
          ids.mutable_at(i, j) = -1;
          dists.mutable_at(i, j) = std::numeric_limits<float>::max();
        }
      }
    });

    return std::make_pair(ids, dists);
  }

private:
  // Does a raw ANN query on the underlying index
  parlay::sequence<pid>
  raw_query(const Point &q, const std::pair<FilterType, FilterType> filter,
            QueryParams query_params) {
    auto [pairElts, dist_cmps] = beam_search<Point, PR, index_type>(
        q, this->G, *(this->points), 0, query_params);
    // auto [frontier, visited] = pairElts;
    auto frontier = pairElts.first;
    if (query_params.verbose) {
      std::cout << "Unfiltered return = " << frontier.size() << std::endl;
    }

    if constexpr (std::is_same<PR, PointRange<T, Point>>::value) {
      frontier = parlay::filter(frontier, [&](pid &p) {
        FilterType filter_value = filter_values[p.first];
        return filter_value >= filter.first && filter_value <= filter.second;
      });
    } else {
      // we actually want to filter and map to original coordinates at the same
      // time
      frontier = parlay::map_maybe(frontier, [&](pid &p) {
        FilterType filter_value = filter_values[p.first];
        if (filter_value >= filter.first && filter_value <= filter.second) {
          return std::optional<pid>(
              std::make_pair(points->subset[p.first], p.second));
        } else {
          return std::optional<pid>();
        }
      });
    }

    return frontier;
  }
};