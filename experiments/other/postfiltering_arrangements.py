import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import numpy as np
from shapely.geometry import Polygon, Point
import math


def get_perfect_solution(main_triangle, divisor=2):
    epsilon = 0.001
    triangles = []
    x = 0
    y = 1

    # Need divisor * (1 / (1 - percent_shift)) = blow_up_factor
    percent_shift = 1 - divisor / blow_up_factor

    while True:
        height = y - y / blow_up_factor
        xs = list(np.arange(0, 1 - y + epsilon, y * percent_shift))
        if 1 - y - xs[-1] > epsilon:
            xs.append(1 - y)
        for x in xs:
            p1 = Point(x, y)
            p2 = Point(x, y - height)
            p3 = Point(x + height, y - height)
            new_triangle = Polygon([p1, p2, p3])
            triangles.append(new_triangle)

        if y - height <= main_triangle.exterior.coords[0][1]:
            break

        y /= divisor

    return triangles


def get_cost_for_perfect_sol(main_triangle, divisor=2):
    epsilon = 0.001
    triangles = []
    x = 0
    y = 1

    # Need divisor * (1 / (1 - percent_shift)) = blow_up_factor
    percent_shift = 1 - divisor / blow_up_factor

    total_cost = 0

    while True:
        height = y - y / blow_up_factor
        to_add = (1 - y + epsilon) / (y * percent_shift)
        total_cost += int(to_add) * y
        if to_add % 1 > epsilon:
            total_cost += y

        if y - height <= main_triangle.exterior.coords[0][1]:
            break

        y /= divisor

    return total_cost


def calculate_cost(triangles, main_triangle):
    # Calculate the total area not covered by the smaller triangles
    union_of_triangles = Polygon()
    for triangle in triangles:
        union_of_triangles = union_of_triangles.union(triangle)

    uncovered_area = main_triangle.difference(union_of_triangles).area
    assert uncovered_area <= 0.001

    y_coords_sum = sum([triangle.exterior.coords[0][1] for triangle in triangles])

    return y_coords_sum


def plot_solution(solution, main_triangle):
    fig, ax = plt.subplots()
    ax.add_patch(
        patches.Polygon(
            main_triangle.exterior.coords, closed=True, color="blue", fill=False
        )
    )
    for triangle in solution:
        ax.add_patch(
            patches.Polygon(
                triangle.exterior.coords, closed=True, color="red", alpha=0.5
            )
        )
    plt.show()


bottom_offsets = [0.0001, 0.0002, 0.0004, 0.0008, 0.001, 0.002, 0.004, 0.008]
# bottom_offsets = [0.002, 0.004]

for blow_up_factor in [3, 4, 5, 6, 7, 8]:
    data = []

    current_costs_for_bottom_offsets = []
    for divisor in np.arange(1.1, blow_up_factor, 0.0001):
        for bottom_offset in bottom_offsets:
            main_triangle = Polygon(
                [[0, bottom_offset], [1 - bottom_offset, bottom_offset], [0, 1]]
            )
            # # best_tessellation = simulated_annealing(main_triangle)
            # best_tessellation = get_perfect_solution(main_triangle, divisor=divisor)

            # cost = sum([triangle.exterior.coords[0][1] for triangle in best_tessellation])
            # # print(f"Cost for bottom offset {bottom_offset}, blow up factor {blow_up_factor}: {cost}; {len(best_tessellation)} triangles")
            # # plot_solution(best_tessellation, main_triangle)

            cost = get_cost_for_perfect_sol(main_triangle, divisor=divisor)
            data.append((divisor, bottom_offset, cost))

            if abs(divisor - 2) < 0.001:
                current_costs_for_bottom_offsets.append(cost)

        memory_costs_for_divisor = [x[2] for x in data if x[0] == divisor]
        # print(f"Average memory cost for worst case blow up factor {blow_up_factor} and divisor {divisor : .2f}: {np.mean(memory_costs_for_divisor)}")

    min_cost_per_bottom_offset = []
    for bottom_offset in bottom_offsets:
        min_cost = min([c for _, bo, c in data if bo == bottom_offset])
        min_cost_per_bottom_offset.append(min_cost)
        print(f"Minimum cost for bottom offset {bottom_offset}: {min_cost}")

    plt.plot(
        bottom_offsets,
        [
            c / m
            for c, m in zip(
                current_costs_for_bottom_offsets, min_cost_per_bottom_offset
            )
        ],
        label=f"Blow up factor {blow_up_factor}",
    )
    # plt.plot(bottom_offsets, current_costs_for_bottom_offsets, label=f"Current cost per bottom offset for blow up factor {blow_up_factor}")

plt.title("Potential Memory Savings")
plt.xlabel("Bottom Offset")
plt.ylabel("Memory Saving Ratio")
plt.legend()
plt.savefig("memory_savings.png", bbox_inches="tight")
