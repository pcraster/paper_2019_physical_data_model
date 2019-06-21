#!/usr/bin/env python
import lue
from scipy.signal import convolve2d
import numpy as np
import docopt
import math
import os.path
import random
import sys


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
        shape):
    """
    Sum the number of points located in each cell of the raster defined
    by the arguments passed in
    """

    ### extent = [
    ###     space_box[2] - space_box[0],
    ###     space_box[3] - space_box[1],
    ### ]

    ### cell_size = [
    ###     extent[0] / shape[1],
    ###     extent[1] / shape[0],
    ### ]

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


# TODO Add to data model
carrying_capacity = 10.0
growth_rate = 0.02
d = 0.01

cell_size = 100  # m


usage = """\
Run deer model simulating the influence of grazing deer on biomass

Usage:
    {command} <dataset> [--area_shape=<shape>] [--nr_timesteps=<steps>]
        [--nr_deer=<deer>]

Options:
    dataset         Pathname of dataset to create
    --area_shape=<shape>  Shape of biomass area [default: 25x50]
    --nr_timesteps=<steps>  Number of timesteps for iteration [default: 25]
    --nr_deer=<deer>  Number of deer to simulate [default: 5]
    -h --help       Show this screen

The shape of the area is in 100x100 meter cells
""".format(
    command = os.path.basename(sys.argv[0]))


def initialize_dataset(
    dataset_pathname):
    """
    Initialize the dataset

    This will prepare the dataset for receiving the modelled system's state.
    """
    space_coordinate_type = np.float32
    space_coordinate_dtype = np.dtype(space_coordinate_type)

    # Create new dataset, truncating any existing one
    dataset = lue.create_dataset(dataset_pathname)

    # simulation ---------------------------------------------------------------
    # Add a simulation phenomenon, representing the collection of
    # simulations. There is only one simulation. In this phenomenon we
    # can store information that is global to this one simulation. In
    # this case, we add the time domain of the simulation, which we can
    # then share between all temporal property-sets.
    nr_simulations = 1
    simulation = dataset.add_phenomenon("simulation")

    # Origin of locations in time
    epoch = lue.Epoch(
        lue.Epoch.Kind.anno_domini, "2019-01-01", lue.Calendar.gregorian)

    # 3 hourly timesteps
    clock = lue.Clock(epoch, lue.Unit.hour, 3)

    time_extent = simulation.add_property_set(
        "time_extent",
        lue.TimeConfiguration(lue.TimeDomainItemType.cell), clock)

    # area ---------------------------------------------------------------------
    # Add phenomenon for representing the collection of areas. (Ιn this
    # model, there is only one area.)
    area = dataset.add_phenomenon("area")

    # Add property-set for area-related information that does not vary
    # through time
    constant = area.add_property_set("constant")

    # Add property for storing discretization information of the
    # fields. This information does not change through time.
    rank = 2
    space_discretization = constant.add_property(
        "space_discretization", dtype=lue.dtype.Count, shape=(rank,))

    # Add property-set for area-related information that varies through
    # time. The time domain of the simulation's time_extent property-set
    # is passed in for re-use.
    space_extent = area.add_property_set(
        "space_extent", time_extent.time_domain,
        lue.SpaceConfiguration(
            lue.Mobility.stationary,
            lue.SpaceDomainItemType.box),
        space_coordinate_dtype=space_coordinate_dtype, rank=rank)

    # Add properties for storing the area's biomass. Property
    # values per object are
    # rank-D arrays. The fact that they are indeed discretized through
    # space instead of just n-D values, is made explicit.

    biomass = space_extent.add_property(
        "biomass", dtype=np.dtype(np.float32),
        rank=rank,
        shape_per_object=lue.ShapePerObject.different,
        shape_variability=lue.ShapeVariability.constant)
    biomass.set_space_discretization(
        lue.SpaceDiscretization.regular_grid, space_discretization)

    ### capacity = space_extent.add_property(
    ###     "capacity", dtype=np.dtype(np.float32),
    ###     rank=rank,
    ###     shape_per_object=lue.ShapePerObject.different,
    ###     shape_variability=lue.ShapeVariability.constant)
    ### capacity.set_space_discretization(
    ###     lue.SpaceDiscretization.regular_grid, space_discretization)

    ### growth = space_extent.add_property(
    ###     "growth", dtype=np.dtype(np.float32),
    ###     rank=rank,
    ###     shape_per_object=lue.ShapePerObject.different,
    ###     shape_variability=lue.ShapeVariability.constant)
    ### growth.set_space_discretization(
    ###     lue.SpaceDiscretization.regular_grid, space_discretization)

    # deer ---------------------------------------------------------------------
    # Add phenomenon for representing the collection of simulated deer.
    deer = dataset.add_phenomenon("deer")

    # Add property-set for deer-related information that does not vary
    # through time
    constant = deer.add_property_set("constant")

    # Add property for storing each deer's sex
    sex_value_type = np.uint8
    sex = constant.add_property("sex", dtype=np.dtype(sex_value_type))

    # Add property-set for deer-related information that varies through
    # time. The time domain of the simulation's time_extent property-set
    # is passed in for re-use.
    location = deer.add_property_set(
        "location", time_extent.time_domain,
        lue.SpaceConfiguration(
            lue.Mobility.mobile,
            lue.SpaceDomainItemType.point),
        space_coordinate_dtype=space_coordinate_dtype, rank=rank)

    # Add property for storing each deer's weight
    weight_value_type = np.uint32
    weight = location.add_property(
        "weight", dtype=np.dtype(weight_value_type), shape=(),
        value_variability=lue.ValueVariability.variable)

    return dataset


def initialize_state(
    dataset,
    area_shape,
    nr_deer):
    """
    Write the initial state of the modelled system to the dataset

    This information could also come from input datasets, but in this
    case we just make it up. After calling this function, the dataset
    contains a single state. During the simulation this state is
    updated through time and appended to the dataset.
    """

    # simulation ---------------------------------------------------------------
    # - Add ID of the simulation
    # - Add time cell for initial state
    nr_time_boxes = 1
    simulation_id = 5  # Just pick an ID

    simulation = dataset.phenomena["simulation"]
    time_extent = simulation.property_sets["time_extent"]

    # For the current time cell, the index of the active set. For the
    # first location in time, this is always 0.
    time_extent.object_tracker.active_set_index.expand(1)[-1] = 0

    # For the current time cell, the IDs of the active objects. In this
    # case, this is always the one and only simulation object.
    time_extent.object_tracker.active_object_id.expand(1)[-1] = simulation_id

    # For the current time box, the number of cells. Since we are only
    # initializing here, there is only one cell. During the simulation
    # this value must be updated to represent the simulated number of
    # time cells.
    time_extent.time_domain.value.count.expand(1)[-1] = 1

    # Per time box a time extent. Since we are initializing here, there
    # is only one cell, with the duration of a single timestep.
    time_extent.time_domain.value.expand(1)[-1] = np.array(
        [0, 1], dtype=lue.dtype.TickPeriodCount).reshape(1, 2);

    # area ---------------------------------------------------------------------
    # - Add ID
    # - Add extent to space domain
    # - Add space discretization to property
    area_id = 25  # Just pick an ID
    rank = len(area_shape)

    area = dataset.phenomena["area"]
    area.object_id.expand(1)[-1] = area_id

    constant = area.property_sets["constant"]

    space_discretization = constant.properties["space_discretization"]

    # Add the discretization information of the one area
    space_discretization.value.expand(1)[-1] = np.array(
        [area_shape], dtype=lue.dtype.Count).reshape(1, rank)

    space_extent = area.property_sets["space_extent"]

    # For the current time cell, the index of the active set. For the
    # first location in time, this is always 0.
    space_extent.object_tracker.active_set_index.expand(1)[-1] = 0

    # For the current time cell, the IDs of the active objects. In this
    # case, this is always the one and only area object.
    space_extent.object_tracker.active_object_id.expand(1)[-1] = area_id

    # For the current time cell, for the one area, an index into the
    # value array of different_shape / constant_shape values (biomass).
    # In this case, one biomass object array will be written, whose
    # index will be 0, since it is the first value.
    space_extent.object_tracker.active_object_index.expand(1)[-1] = 0

    # For the one and only area a space box
    space_coordinate_dtype = space_extent.space_domain.value.dtype
    space_extent.space_domain.value.expand(1)[-1] = np.array(
        [0, 0, area_shape[1] * cell_size, area_shape[0] * cell_size],
        dtype=space_coordinate_dtype).reshape(1, 2 * rank)

    # For the area a 2D array with biomass. Initially, biomass values
    # are very low: [1, 2).
    biomass = space_extent.properties["biomass"]
    biomass_dtype = biomass.value.dtype
    biomass.value.expand(area_id, area_shape, 1)[0] = \
        1 + np.random.rand(*area_shape).astype(biomass_dtype)

    # biomass_value = np.random.rand(*area_shape).astype(biomass_dtype)
    # biomass_value = np.random.rand(*area_shape).astype(biomass_dtype)
    # biomass.value.expand(area_id, area_shape, 1)[0] = biomass_value


    # deer ---------------------------------------------------------------------
    # - ID
    deer = dataset.phenomena["deer"]

    # IDs of the deer objects. The order must correspond with the order in
    # which non-temporal deer information is written to the dataset.
    deer_ids = np.arange(nr_deer, dtype=lue.dtype.ID)
    deer.object_id.expand(nr_deer)[:] = deer_ids

    constant = deer.property_sets["constant"]

    sex = constant.properties["sex"]

    # Randomly assign sex to each deer
    sex_dtype = sex.value.dtype
    sex.value.expand(nr_deer)[:] = np.random.randint(
        low=0, high=2, size=nr_deer, dtype=sex_dtype)

    location = deer.property_sets["location"]

    # For the current time cell, the index of the active set. For the
    # first location in time, this is alway 0.
    location.object_tracker.active_set_index.expand(1)[-1] = 0

    # For the current time cell, the IDs of the active objects. In this
    # case, these are the IDs of the deer.
    location.object_tracker.active_object_id.expand(nr_deer)[-nr_deer:] = \
        deer_ids

    # For each deer a space point. Initially, they all start in the
    # center of the area.
    space_coordinate_dtype = location.space_domain.value.dtype
    center = 0.5 * np.array(area_shape[::-1]) * cell_size
    location.space_domain.value.expand(nr_deer)[-nr_deer:] = \
        np.array(nr_deer * [center], dtype=space_coordinate_dtype) \
            .reshape(nr_deer, rank)

    # For each deer a weight
    weight = location.properties["weight"]
    weight_dtype = weight.value.dtype
    weight.value.expand(nr_deer)[:] = np.random.randint(
        low=100, high=501, size=nr_deer, dtype=weight_dtype)


def simulate_new_states(
    dataset,
    nr_timesteps):
    """
    Simulate the state of the modelled system through time and write
    the results to the dataset

    The initial state is read from the dataset.
    """

    # Read initial state from the dataset

    # simulation ---------------------------------------------------------------
    # We now know how many time steps we need to fill with temporal
    # state information. Update the simulation's time domain accordingly.
    simulation = dataset.phenomena["simulation"]
    time_extent = simulation.property_sets["time_extent"]

    # For each time cell, the index of the active set. In this case,
    # each active set contains a single object, so the indices are a
    # strictly monotonically increasing range. We append to information
    # already present.
    nr_indices_present = time_extent.object_tracker.active_set_index.nr_indices
    time_extent.object_tracker.active_set_index.expand(nr_timesteps) \
        [-nr_timesteps:] = np.arange(
            nr_indices_present, nr_timesteps + nr_indices_present,
            dtype=np.dtype(np.uint64))

    # For each time cell, the IDs of the active objects. In this case,
    # this is always the one and only simulation object, whose ID is
    # already stored in the dataset.
    simulation_id = time_extent.object_tracker.active_object_id[-1]
    time_extent.object_tracker.active_object_id.expand(nr_timesteps) \
        [-nr_timesteps:] = np.full(
            nr_timesteps, simulation_id, dtype=lue.dtype.ID)

    # For each time box, the number of cells. In this case, we only have
    # a single time box.
    time_extent.time_domain.value.count[-1] += nr_timesteps

    # Per time box a time extent. There is only one time box, which
    # we are filling here. Move the box' end coordinate into the future,
    # by the number of timesteps simulated here.
    assert time_extent.time_domain.value[-1][0] == 0
    time_extent.time_domain.value[-1] += np.array(
        [0, nr_timesteps], dtype=lue.dtype.TickPeriodCount)

    area = dataset.phenomena["area"]
    space_extent = area.property_sets["space_extent"]
    area_id = space_extent.object_tracker.active_object_id[-1]
    biomass = space_extent.properties["biomass"]
    biomass_value = biomass.value[area_id]
    current_biomass_value = biomass_value[-1]

    nr_neighbours = window_total(
        np.full_like(current_biomass_value, fill_value=1), radius=1)
    area_shape = biomass_value.shape[1:]

    ### space_domain = space_extent.space_domain
    ### assert space_domain.configuration.mobility == lue.Mobility.stationary
    ### assert space_domain.configuration.item_type == lue.SpaceDomainItemType.box

    ### assert space_domain.value.nr_boxes == 1
    ### assert len(space_domain.value.array_shape) == 1  # A 1D array ...
    ### # ... of four values: x1, y1, x2, y2
    ### assert space_domain.value.array_shape[0] == 4  # Implies rank == 2

    area_extent = space_extent.space_domain.value[0]

    deer = dataset.phenomena["deer"]
    location = deer.property_sets["location"]
    active_deer_set_idx = location.object_tracker.active_set_index[-1]
    deer_ids = location.object_tracker.active_object_id[active_deer_set_idx:]
    nr_deer = len(deer_ids)
    current_deer_location = location.space_domain.value[-nr_deer:]
    deer_weight = location.properties["weight"]
    current_deer_weight = deer_weight.value[-nr_deer:]

    # Iterate through time and update the simulated temporal state. Each
    # new state is written to the dataset.
    for t in range(nr_timesteps):







        # For each timestep, do the folowing:
        #
        # deer
        # - Determine IDs of active deer and write to dataset
        # - Calculate positions in space and write to dataset
        # - Calculate weights and write to dataset
        #
        # area
        # - Determine IDs of active area and write to dataset
        # - Calculate biomass and write to dataset


        # We don't simulate birth and death, so the collection of deer
        # stays the same during the simulated period of time


        # deer -----------------------------------------------------------------

        # For the current time cell, the index of the active set
        location.object_tracker.active_set_index.expand(1)[-1] = \
            location.object_tracker.active_set_index[-1] + nr_deer

        # For the current time cell, the IDs of the active deer.
        location.object_tracker.active_object_id.expand(nr_deer)[-nr_deer:] = \
            deer_ids

        ### # Update deer locations
        ### # TODO Calculate deer locations
        ### new_deer_location = current_deer_location
        ### location.space_domain.value.expand(nr_deer)[-nr_deer:] = \
        ###     new_deer_location

        ### # Update deer weights
        ### # TODO Calculate deer weights
        ### new_deer_weight = current_deer_weight
        ### deer_weight.value.expand(nr_deer)[-nr_deer:] = new_deer_weight


        # area -----------------------------------------------------------------

        # For the current time cell, the index of the active set. There is
        # only one area, so we can just use the previous index and add one.
        space_extent.object_tracker.active_set_index.expand(1)[-1] = \
            space_extent.object_tracker.active_set_index[-1] + 1

        # For the current time cell, the IDs of the active objects. In this
        # case, this is always the one and only area object.
        space_extent.object_tracker.active_object_id.expand(1)[-1] = area_id

        # For the current time cell, for the one area, an index into the
        # value array of different_shape / constant_shape values (biomass).
        space_extent.object_tracker.active_object_index.expand(1)[-1] = \
            space_extent.object_tracker.active_object_index[-1] + 1



        capacity = 1 - (current_biomass_value / carrying_capacity)
        growth = capacity * growth_rate * current_biomass_value
        new_biomass_value = current_biomass_value + growth

        diffusion = \
            d * (
                window_total(new_biomass_value, radius=1) -
                nr_neighbours * new_biomass_value
            )

        new_biomass_value += diffusion


        # Iterate over all deer and update biomass in cell they are
        # located in
        # Create a field with for each cell the sum of a property of
        # the deer located in that cell
        deer_weight_per_cell = cell_total(
            current_deer_location, current_deer_weight,
            area_extent, area_shape)

        new_biomass_value = np.maximum(
            0.01,
            new_biomass_value - (deer_weight_per_cell * 0.0008))

        biomass_value.expand(1)[-1:] = new_biomass_value



        enough_food = new_biomass_value > 0.5

        # Update deer weights
        new_deer_weight = np.minimum(750, current_deer_weight * 1.003)
        deer_weight.value.expand(nr_deer)[-nr_deer:] = new_deer_weight


        # Update deer locations
        new_deer_location = gaussian_move(
            current_deer_location, enough_food, area_extent, area_shape,
            sigma=cell_size)
        location.space_domain.value.expand(nr_deer)[-nr_deer:] = \
            new_deer_location


        current_deer_weight = new_deer_weight
        current_deer_location = new_deer_location
        current_biomass_value = new_biomass_value

        sys.stdout.write('.')
        sys.stdout.flush()

    sys.stdout.write('\n')


def run_model(
        dataset_pathname,
        nr_timesteps,
        area_shape,
        nr_deer):

    # Both the initial state and the simulated states will be stored in
    # the dataset. This implies that there will be states at nr_timesteps
    # + 1 timesteps in the dataset.

    dataset = initialize_dataset(dataset_pathname)
    lue.assert_is_valid(dataset)

    initialize_state(dataset, area_shape, nr_deer)
    lue.assert_is_valid(dataset)

    simulate_new_states(dataset, nr_timesteps)
    lue.assert_is_valid(dataset)

    # BTW, simulating another nr_timesteps also works. Simulation just
    # appends to the last stored state.
    # simulate_new_states(dataset, nr_timesteps)
    # lue.assert_is_valid(dataset)

    lue.assert_is_valid(dataset_pathname)


if __name__ == "__main__":
    parsed_arguments = docopt.docopt(usage)

    parsed_dataset_pathname = os.path.abspath(parsed_arguments["<dataset>"])
    parsed_area_shape = tuple(
        int(e) for e in parsed_arguments["--area_shape"].split("x"))
    parsed_nr_timesteps = int(parsed_arguments["--nr_timesteps"])
    parsed_nr_deer = int(parsed_arguments["--nr_deer"])

    run_model(
        parsed_dataset_pathname, parsed_nr_timesteps, parsed_area_shape,
        parsed_nr_deer)
