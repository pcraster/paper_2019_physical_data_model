"""
Some operations used in the deer-biomass model
"""
import numpy as np
from scipy.signal import convolve2d
import random
import math


def window_total(
        array,
        radius):

    window = np.full((2 * radius + 1, 2 * radius + 1), 1.0, array.dtype)

    return convolve2d(array, window, mode="same", fillvalue=0)


def classify(
        value,
        min_value,
        max_value,
        nr_classes):

    assert min_value <= value < max_value
    assert max_value > min_value

    value_range = max_value - min_value
    class_range = value_range / nr_classes
    idx = int(math.floor((value - min_value) / class_range))

    assert idx < nr_classes, "value: {}, min_value: {}, max_value: {}, nr_classes: {}".format(
        value, min_value, max_value, nr_classes)

    return idx


def cell_indices(
        space_box,
        shape,
        cell_size,
        point):

    assert point[0] >= space_box[0] and \
        point[0] < space_box[0] + shape[1] * cell_size, \
        "point({}), space_box({}), shape({}), cell_size({})".format(
            point, space_box, shape, cell_size)
    assert point[1] >= space_box[1] and \
        point[1] < space_box[1] + shape[0] * cell_size, \
        "point({}), space_box({}), shape({}), cell_size({})".format(
            point, space_box, shape, cell_size)

    row = classify(point[1], space_box[1], space_box[3], shape[0])
    col = classify(point[0], space_box[0], space_box[2], shape[1])

    return row, col


def cell_total(
        points,
        values,
        space_box,
        shape,
        cell_size):
    """
    Sum the number of points located in each cell of the raster defined
    by the arguments passed in
    """

    result = np.full(shape, 0, np.uint64)

    for point, value in zip(points, values):
        row, col = cell_indices(space_box, shape, cell_size, point)
        result[row][col] += value

    return result


def clamp(
        value,
        min_value,
        max_value):
    return max(min(value, max_value), min_value)


def gaussian_move(
        current_locations,
        boolean,
        space_box,
        cell_size,
        shape,
        sigma):

    assert boolean.dtype == np.bool

    new_locations = []

    min_x, min_y, max_x, max_y = space_box

    for x, y in current_locations:

        # Pick a new location and clamp it to the area's extent
        new_x = clamp(random.normalvariate(x, sigma), min_x, max_x - 1e-3)
        new_y = clamp(random.normalvariate(y, sigma), min_y, max_y - 1e-3)
        row, col = cell_indices(
            space_box, shape, cell_size, [new_x, new_y])

        # This assumes that there are 'good' cells nearby
        while not boolean[row][col]:
            new_x = clamp(random.normalvariate(x, sigma), min_x, max_x - 1e-3)
            new_y = clamp(random.normalvariate(y, sigma), min_y, max_y - 1e-3)
            row, col = cell_indices(
                space_box, shape, cell_size, [new_x, new_y])

        assert min_x <= new_x < max_x
        assert min_y <= new_y < max_y
        new_locations.append([new_x, new_y])

    return np.array(new_locations, dtype=current_locations.dtype)
