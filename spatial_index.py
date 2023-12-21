import numpy as np
import diskannpy
import os
from math import log2
import argparse
from pathlib import Path
from utils import parse_ann_benchmarks_hdf5
import time

BUILD_THREADS = 0
# INDEX_DIR = "/ssd2/ben/range_indices"

class SpatialIndex:
    """An abstract class for spatial indices"""
    
    def __init__(self, data, distance_metric):
        self.data = data
        self.distance_metric = distance_metric

    def build(self, **kwargs):
        raise NotImplementedError

    def search(self, query, k):
        raise NotImplementedError
    
    def load(self, path):
        raise NotImplementedError
    

class DiskANNIndex(SpatialIndex):
    def __init__(self, name, index_directory, alpha, build_complexity, degree, distance_metric):
        self.name = name
        self.index_directory = index_directory
        self.alpha = alpha
        self.build_complexity = build_complexity
        self.degree = degree
        self.distance_metric = distance_metric

        self.index = None
        self.index_path = os.path.join(self.index_directory, self.name)

    def build(self, data):
        diskannpy.build_memory_index(
            data,
            alpha=self.alpha,
            complexity=self.build_complexity,
            graph_degree=self.degree,
            distance_metric=self.distance_metric,
            index_directory=self.index_directory,
            index_prefix=self.name,
            num_threads=BUILD_THREADS,
        )
        return self

    def load(self):
        self.index = diskannpy.StaticMemoryIndex(
            index_directory=self.index_directory,
            num_threads=0,
            initial_search_complexity=self.build_complexity,
            index_prefix=self.name,
        )
        return self

    def search(self, query, k, complexity):
        search_result = self.index.search(
            query=query, complexity=complexity, k_neighbors=k
        )
        return search_result.identifiers, search_result.distances
    
    def build_or_load(self, data):
        print(self.index_path)
        print(os.path.exists(self.index_path))
        if os.path.exists(self.index_path):
            return self.load()
        else:
            return self.build(data)


class DiskANNIndexFactory:
    def __init__(self, index_directory, alpha, build_complexity, degree, distance_metric):
        self.index_directory = index_directory
        self.alpha = alpha
        self.build_complexity = build_complexity
        self.degree = degree
        self.distance_metric = distance_metric

    def create_index(self, name):
        return DiskANNIndex(
            name=name,
            index_directory=self.index_directory,
            alpha=self.alpha,
            build_complexity=self.build_complexity,
            degree=self.degree,
            distance_metric=self.distance_metric,
        )