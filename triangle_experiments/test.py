import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backend_bases import MouseButton

from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import nearest_points

class ShapelyInteractiveTrianglePlot:
    def __init__(self):

        bottom_offset = 1 / 8
        self.slop_value = 2

        self.fig, self.ax = plt.subplots()

        self.ax.set_xlabel("Width")
        self.ax.set_ylabel("Starting Position")

        self.ax.set_xlim(bottom_offset - 0.2, 1.2 - bottom_offset)
        self.ax.set_ylim(bottom_offset - 0.2, 1.2)

        self.main_triangle = Polygon([[0, bottom_offset], [1 - bottom_offset, bottom_offset], [0, 1]])
        self.ax.add_patch(patches.Polygon(self.main_triangle.exterior.coords, closed=True, color='blue', fill=False))

        self.dash_triangle = patches.Polygon([[0, 0], [0, 0], [0, 0]], closed=True, color='blue', linestyle='dashed', fill=False)
        self.ax.add_patch(self.dash_triangle)

        self.triangles = MultiPolygon()

        self.y_cords = []
        self.x_cords = []

        self.area_text = self.ax.text(0.1, 0.95, '', transform=self.ax.transAxes)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

        # self.ax.set_yscale('log')
        # self.ax.set_ylim(10**-2, 10**1)
        # self.ax.set_xscale('log')
        # self.ax.set_xlim(10**-2, 10**1)


    def on_click(self, event):
        if event.button is MouseButton.LEFT:
            x, y = self.get_closest_point_in_triangle(event.xdata, event.ydata)
            self.add_triangle(x, y)

    def on_motion(self, event):
        x, y = self.get_closest_point_in_triangle(event.xdata, event.ydata)
        self.update_dashed_triangle(x, y)
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'u':
            self.undo()
        elif event.key == 'r':
            self.redo()

    def add_triangle(self, x, y):
        size = y - y / self.slop_value
        new_triangle = Polygon([[x, y], [x, y - size], [x + size, y - size]])
        self.y_cords.append(y)
        self.x_cords.append(x)
        print(f"{x},{y}")
        self.triangles = self.triangles.union(new_triangle)
        self.ax.add_patch(patches.Polygon(new_triangle.exterior.coords, closed=True, color='blue'))
        self.update_area_text()
        self.fig.canvas.draw()

    def update_dashed_triangle(self, x, y):
        size = y - y / self.slop_value
        self.dash_triangle.set_xy([[x, y], [x, y - size], [x + size, y - size]])

    def update_area_text(self):
        filled_area = self.main_triangle.intersection(self.triangles).area
        total_area = self.main_triangle.area
        percentage_filled = (filled_area / total_area) * 100
        self.area_text.set_text(f'Filled: {percentage_filled:.2f}% Cost {sum(self.y_cords):.2f}')

    def get_closest_point_in_triangle(self, x, y):
        if x is None or y is None:
            return 0, 0
        if self.is_inside_main_triangle(x, y):
            return x, y
        
        point = Point(x, y)
        closest_point = nearest_points(self.main_triangle, point)[0]
        return list(closest_point.coords)[0]


    def is_inside_main_triangle(self, x, y):
        return self.main_triangle.contains(Point(x, y))

    def show(self):
        plt.show()

# Create and show the shapely-based interactive plot
shapely_plot = ShapelyInteractiveTrianglePlot()
shapely_plot.show()
