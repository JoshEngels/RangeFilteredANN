cmake_minimum_required(VERSION 3.5)
project(DPC-ANN LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(Python 3 COMPONENTS Interpreter Development.Module REQUIRED)

find_package(pybind11 CONFIG REQUIRED)

# -------------------------- Includes ------------------------------------------

include_directories("src")
include_directories("ParlayANN/parlaylib/include")
include_directories("ParlayANN")

# -------------------------- Options ------------------------------------------

set(COMPILE_OPTIONS

    $<$<CONFIG:Debug>:-std=c++17 -DPARLAY_SEQUENTIAL -mcx16 -pthread -march=native -g -O0 -DDEBUG -fPIC>
    
    $<$<CONFIG:RelWithDebInfo>:-std=c++17 -O3 -DHOMEGROWN -mcx16 -pthread -march=native -DNDEBUG -fPIC -g>

    $<$<CONFIG:Release>:-std=c++17 -O3 -DHOMEGROWN -mcx16 -pthread -march=native -DNDEBUG -fPIC>
)


# --------------------- Create Python Library --------------------------------------

set(CC_SOURCES
    src/filtered_dataset.cpp
    src/postfilter_vamana.cpp
    src/prefiltering.cpp
    src/range_filter_tree.cpp
    python_bindings/python_bindings.cpp
    )

pybind11_add_module(window_ann  ${CC_SOURCES})
target_compile_options(window_ann PRIVATE ${COMPILE_OPTIONS})
