// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// #define PYBIND11_DETAILED_ERROR_MESSAGES

#include <string>

#include "pybind11/numpy.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "algorithms/IVF/posting_list.h"
#include "algorithms/utils/filters.h"
#include "algorithms/utils/types.h"
#include "filtered_dataset.h"
#include "postfilter_vamana.h"
#include "prefiltering.h"
#include "python/builder.cpp"
#include "python/vamana_index.cpp"
#include "range_filter_tree.h"
#include "super_optimized_postfilter_tree.h"

PYBIND11_MAKE_OPAQUE(std::vector<uint32_t>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<int8_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>);

namespace py = pybind11;
using namespace pybind11::literals;

template <typename T, typename Point>
using posting_list_t = NaivePostingList<T, Point>;

template <typename T, typename Point>
using filtered_posting_list_t = FilteredPostingList<T, Point>;

// using NeighborsAndDistances = std::pair<py::array_t<unsigned int,
// py::array::c_style | py::array::forcecast>, py::array_t<float,
// py::array::c_style | py::array::forcecast>>;

struct Variant {
  std::string builder_name;
  std::string index_name;
  std::string ivf_name;
  std::string agnostic_name;
};

const Variant FloatEuclidianVariant{"build_vamana_float_euclidian_index",
                                    "VamanaFloatEuclidianIndex",
                                    "IVFFloatEuclidianIndex", "FloatEuclidian"};
const Variant FloatMipsVariant{"build_vamana_float_mips_index",
                               "VamanaFloatMipsIndex", "IVFFloatMipsIndex",
                               "FloatMips"};

const Variant UInt8EuclidianVariant{"build_vamana_uint8_euclidian_index",
                                    "VamanaUInt8EuclidianIndex",
                                    "IVFUInt8EuclidianIndex", "UInt8Euclidian"};
const Variant UInt8MipsVariant{"build_vamana_uint8_mips_index",
                               "VamanaUInt8MipsIndex", "IVFUInt8MipsIndex",
                               "UInt8Mips"};

const Variant Int8EuclidianVariant{"build_vamana_int8_euclidian_index",
                                   "VamanaInt8EuclidianIndex",
                                   "IVFInt8EuclidianIndex", "Int8Euclidian"};
const Variant Int8MipsVariant{"build_vamana_int8_mips_index",
                              "VamanaInt8MipsIndex", "IVFInt8MipsIndex",
                              "Int8Mips"};

BuildParams DEFAULT_BUILD_PARAMS = BuildParams(64, 500, 1.175, "index_cache");

template <typename T, typename Point>
inline void add_variant(py::module_ &m, const Variant &variant) {

  m.def(variant.builder_name.c_str(), build_vamana_index<T, Point>,
        "distance_metric"_a, "data_file_path"_a, "index_output_path"_a,
        "graph_degree"_a, "beam_width"_a, "alpha"_a);

  py::class_<VamanaIndex<T, Point>>(m, variant.index_name.c_str())
      .def(py::init<std::string &, std::string &, size_t, size_t>(),
           "index_path"_a, "data_path"_a, "num_points"_a,
           "dimensions"_a) // maybe these last two are unnecessary?
      // do we want to add options like visited limit, or leave those as
      // defaults?
      .def("batch_search", &VamanaIndex<T, Point>::batch_search, "queries"_a,
           "num_queries"_a, "knn"_a, "beam_width"_a)
      .def("batch_search_from_string",
           &VamanaIndex<T, Point>::batch_search_from_string, "queries"_a,
           "num_queries"_a, "knn"_a, "beam_width"_a)
      .def("check_recall", &VamanaIndex<T, Point>::check_recall, "gFile"_a,
           "neighbors"_a, "k"_a);

  py::class_<PrefilterIndex<T, Point>>(
      m, ("PrefilterIndex" + variant.agnostic_name).c_str())
      .def(py::init<py::array_t<T>, py::array_t<float_t>, BuildParams>(),
           "points"_a, "filter_values"_a,
           "build_params"_a = DEFAULT_BUILD_PARAMS)
      .def("batch_search", &PrefilterIndex<T, Point>::batch_search, "queries"_a,
           "filters"_a, "num_queries"_a, "query_params"_a);

  py::class_<RangeFilterTreeIndex<T, Point>>(
      m, ("RangeFilterTreeIndex" + variant.agnostic_name).c_str())
      .def(py::init<py::array_t<T>, py::array_t<float_t>, int32_t, size_t,
                    BuildParams>(),
           "points"_a, "filter_values"_a, "cutoff"_a = 1000,
           "split_factor"_a = 2, "build_params"_a = DEFAULT_BUILD_PARAMS)
      .def("batch_search", &RangeFilterTreeIndex<T, Point>::batch_search,
           "queries"_a, "filters"_a, "num_queries"_a, "query_method"_a,
           "query_params"_a);

  py::class_<PostfilterVamanaIndex<T, Point>>(
      m, ("PostfilterVamanaIndex" + variant.agnostic_name).c_str())
      .def(py::init<py::array_t<T>, py::array_t<float_t>, BuildParams>(),
           "points"_a, "filters"_a, "build_params"_a = DEFAULT_BUILD_PARAMS)
      .def("batch_search", &PostfilterVamanaIndex<T, Point>::batch_search,
           "queries"_a, "filters"_a, "num_queries"_a, "query_params"_a);

  py::class_<RangeFilterTreeIndex<T, Point, PostfilterVamanaIndex>>(
      m, ("VamanaRangeFilterTreeIndex" + variant.agnostic_name).c_str())
      .def(py::init<py::array_t<T>, py::array_t<float_t>, int32_t, size_t,
                    BuildParams>(),
           "points"_a, "filter_values"_a, "cutoff"_a = 1000,
           "split_factor"_a = 2, "build_params"_a = DEFAULT_BUILD_PARAMS)
      .def("batch_search",
           &RangeFilterTreeIndex<T, Point, PostfilterVamanaIndex>::batch_search,
           "queries"_a, "filters"_a, "num_queries"_a, "query_method"_a,
           "query_params"_a);

  py::class_<SuperOptimizedPostfilterTree<T, Point, PostfilterVamanaIndex>>(
      m, ("SuperOptimizedPostfilterTreeIndex" + variant.agnostic_name).c_str())
      .def(py::init<py::array_t<T>, py::array_t<float_t>, int32_t, float, float,
                    BuildParams>(),
           "points"_a, "filter_values"_a, "cutoff"_a = 1000,
           "split_factor"_a = 2, "shift_factor"_a = 0.5,
           "build_params"_a = DEFAULT_BUILD_PARAMS)
      .def("batch_search",
           &SuperOptimizedPostfilterTree<T, Point,
                                         PostfilterVamanaIndex>::batch_search,
           "queries"_a, "filters"_a, "num_queries"_a, "query_params"_a);
}

PYBIND11_MODULE(window_ann, m) {
  m.doc() = "WindowANN Python Bindings";
#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif

  // let's re-export our defaults
  py::module_ default_values = m.def_submodule("defaults");

  default_values.attr("METRIC") = "Euclidian";
  default_values.attr("ALPHA") = 1.2;
  default_values.attr("GRAPH_DEGREE") = 64;
  default_values.attr("BEAMWIDTH") = 128;

  py::class_<csr_filters>(m, "csr_filters")
      .def(py::init<std::string &>())
      .def("match", &csr_filters::match, "p"_a, "f"_a)
      .def("first_label", &csr_filters::first_label, "p"_a)
      .def("print_stats", &csr_filters::print_stats)
      .def("filter_count", &csr_filters::filter_count, "f"_a)
      .def("point_count", &csr_filters::point_count, "p"_a)
      .def("transpose", &csr_filters::transpose)
      .def("transpose_inplace", &csr_filters::transpose_inplace);

  // should have initializers taking either one or two int32_t arguments
  py::class_<QueryFilter>(m, "QueryFilter")
      .def(py::init<int32_t>(), "a"_a)
      .def(py::init<int32_t, int32_t>(), "a"_a, "b"_a)
      .def("is_and", &QueryFilter::is_and)
      .def("__repr__",
           [](const QueryFilter &f) {
             return "<QueryFilter: " + std::to_string(f.a) + ", " +
                    std::to_string(f.b) + ">";
           })
      .def("__str__",
           [](const QueryFilter &f) {
             return "(" + std::to_string(f.a) + ", " + std::to_string(f.b) +
                    ")";
           })
      .def_readonly("a", &QueryFilter::a)
      .def_readonly("b", &QueryFilter::b);

  py::class_<QueryParams>(m, "QueryParams")
      .def(py::init<long, long, double, long, long, long, long,
                    std::optional<float>, bool>(),
           "k"_a, "beam_width"_a, "cut"_a, "limit"_a, "degree_limit"_a,
           "final_beam_multiply"_a, "postfiltering_max_beam"_a,
           "min_query_to_bucket_ratio"_a, "verbose"_a);

  py::class_<BuildParams>(m, "BuildParams")
      .def(py::init<long, long, double, std::string>(), "max_degree"_a,
           "limit"_a, "alpha"_a, "cache_path"_a);

  py::class_<FilteredDataset>(m, "FilteredDataset")
      .def(py::init<std::string &, std::string &>(), "points_filename"_a,
           "filters_filename"_a)
      .def("distance", &FilteredDataset::distance, "a"_a, "b"_a)
      .def("size", &FilteredDataset::size)
      .def("get_n_filters", &FilteredDataset::get_n_filters)
      .def("get_filter_size", &FilteredDataset::get_filter_size, "filter_id"_a)
      .def("get_point_size", &FilteredDataset::get_point_size, "point_id"_a)
      .def("get_filter_points", &FilteredDataset::get_filter_points,
           "filter_id"_a)
      .def("get_point_filters", &FilteredDataset::get_point_filters,
           "point_id"_a)
      .def("get_filter_intersection", &FilteredDataset::get_filter_intersection,
           "filter_id_1"_a, "filter_id_2"_a)
      .def("get_point_intersection", &FilteredDataset::get_point_intersection,
           "point_id_1"_a, "point_id_2"_a);

  add_variant<float, Euclidian_Point<float>>(m, FloatEuclidianVariant);
  add_variant<float, Mips_Point<float>>(m, FloatMipsVariant);
  add_variant<uint8_t, Euclidian_Point<uint8_t>>(m, UInt8EuclidianVariant);
  add_variant<uint8_t, Mips_Point<uint8_t>>(m, UInt8MipsVariant);
  add_variant<int8_t, Euclidian_Point<int8_t>>(m, Int8EuclidianVariant);
  add_variant<int8_t, Mips_Point<int8_t>>(m, Int8MipsVariant);
};