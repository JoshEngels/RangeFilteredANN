cmake_minimum_required(VERSION 3.5)
project(DPC-ANN LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(Python 3 COMPONENTS Interpreter Development.Module REQUIRED)

find_package(pybind11 CONFIG REQUIRED)

include_directories("ParlayANN/parlaylib/include")
include_directories("ParlayANN")

pybind11_add_module(window_ann python_bindings/python_bindings.cpp)