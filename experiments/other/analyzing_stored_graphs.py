# %%


def Graph(gFile):
    with open(gFile, "rb") as reader:
        # read num points and max degree
        num_points = int.from_bytes(reader.read(4), byteorder="little")
        max_deg = int.from_bytes(reader.read(4), byteorder="little")
        n = num_points
        maxDeg = max_deg
        print(n, maxDeg)

        # read degrees and perform scan to find offsets
        degrees = list(
            int.from_bytes(reader.read(4), byteorder="little") for _ in range(n)
        )
        offsets = [0] * (n + 1)
        total = 0
        for i in range(n):
            offsets[i] = total
            total += degrees[i]
        offsets[n] = total
        print(offsets[:5])

        # write to graph object
        graph = []
        BLOCK_SIZE = 1000000
        index = 0
        total_size_read = 0
        while index < n:
            g_floor = index
            g_ceiling = g_floor + BLOCK_SIZE if g_floor + BLOCK_SIZE <= n else n
            total_size_to_read = offsets[g_ceiling] - offsets[g_floor]
            edges = list(
                int.from_bytes(reader.read(4), byteorder="little")
                for _ in range(total_size_to_read)
            )
            for i in range(g_floor, g_ceiling):
                graph.append([])
                for j in range(degrees[i]):
                    graph[-1].append(edges[offsets[i] - total_size_read + j])
            total_size_read += total_size_to_read
            index = g_ceiling

    return degrees, graph


degrees_1, g_1 = Graph(
    "experiments/index_cache/postfilter_vamana/redcaps-512-angular_vamana_500_64_1.175000_1202510848.000000_1609459200.000000_11588824.bin"
)
degrees_2, g_2 = Graph(
    "experiments/index_cache/postfilter_vamana/vamana_500_64_1.175000_1202510848.000000_1609459200.000000_11588824.bin"
)
# %%
import numpy as np

print(np.mean(degrees_1), np.mean(degrees_2))
# %%

degrees_3, g_3 = Graph(
    "experiments/index_cache/postfilter_vamana/sift-128-euclidean_vamana_500_64_1.175000_0.000001_0.999998_1000000.bin"
)
# %%

degrees_3, g_3 = Graph(
    "experiments/index_cache/postfilter_vamana/redcaps-512-angular_vamana_500_64_1.000000_1202510848.000000_1609459200.000000_11588824.bin"
)

# %%

print(np.mean(degrees_3))
# %%
