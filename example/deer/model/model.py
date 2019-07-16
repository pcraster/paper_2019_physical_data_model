#!/usr/bin/env python
from operation import *
import lue
import numpy as np
import docopt
import math
import os.path
import random
import sys


# Global variables. These could be added / read from the dataset instead
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

    # park ---------------------------------------------------------------------
    # Add phenomenon for representing the collection of park areas. (Ιn
    # this model, there is only one park.)
    park = dataset.add_phenomenon("park")

    # Add property-set for park-related information that does not vary
    # through time
    constant = park.add_property_set("constant")

    # Add property for storing discretization information of the
    # fields. This information does not change through time.
    rank = 2
    space_discretization = constant.add_property(
        "space_discretization", dtype=lue.dtype.Count, shape=(rank,))

    # Add property-set for park-related information that varies through
    # time. The time domain of the simulation's time_extent property-set
    # is passed in for re-use.
    space_extent = park.add_property_set(
        "space_extent", time_extent.time_domain,
        lue.SpaceConfiguration(
            lue.Mobility.stationary,
            lue.SpaceDomainItemType.box),
        space_coordinate_dtype=space_coordinate_dtype, rank=rank)

    # Add properties for storing the park's biomass. Property
    # values per object are rank-D arrays. The fact that they are indeed
    # discretized through space instead of just n-D values, is made explicit.

    biomass = space_extent.add_property(
        "biomass", dtype=np.dtype(np.float32),
        rank=rank,
        shape_per_object=lue.ShapePerObject.different,
        shape_variability=lue.ShapeVariability.constant)
    biomass.set_space_discretization(
        lue.SpaceDiscretization.regular_grid, space_discretization)

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

    # park ---------------------------------------------------------------------
    # - Add ID
    # - Add extent to space domain
    # - Add space discretization to property
    area_id = 25  # Just pick an ID
    rank = len(area_shape)

    park = dataset.phenomena["park"]
    park.object_id.expand(1)[-1] = area_id

    constant = park.property_sets["constant"]

    space_discretization = constant.properties["space_discretization"]

    # Add the discretization information of the one area
    space_discretization.value.expand(1)[-1] = np.array(
        [area_shape], dtype=lue.dtype.Count).reshape(1, rank)

    space_extent = park.property_sets["space_extent"]

    # For the current time cell, the index of the active set. For the
    # first location in time, this is always 0.
    space_extent.object_tracker.active_set_index.expand(1)[-1] = 0

    # For the current time cell, the IDs of the active objects. In this
    # case, this is always the one and only park object.
    space_extent.object_tracker.active_object_id.expand(1)[-1] = area_id

    # For the current time cell, for the one park, an index into the
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

    park = dataset.phenomena["park"]
    space_extent = park.property_sets["space_extent"]
    area_id = space_extent.object_tracker.active_object_id[-1]
    biomass = space_extent.properties["biomass"]
    biomass_value = biomass.value[area_id]
    biomass_array = biomass_value[-1]

    nr_neighbours = window_total(
        np.full_like(biomass_array, fill_value=1), radius=1)
    area_shape = biomass_value.shape[1:]

    area_extent = space_extent.space_domain.value[0]

    deer = dataset.phenomena["deer"]
    location = deer.property_sets["location"]
    active_deer_set_idx = location.object_tracker.active_set_index[-1]
    deer_ids = location.object_tracker.active_object_id[active_deer_set_idx:]
    nr_deer = len(deer_ids)
    deer_location_array = location.space_domain.value[-nr_deer:]
    deer_weight = location.properties["weight"]
    deer_weight_array = deer_weight.value[-nr_deer:]

    # Iterate through time and update the simulated temporal state. Each
    # new state is written to the dataset.
    for t in range(nr_timesteps):

        # Model ----------------------------------------------------------------
        capacity = 1 - (biomass_array / carrying_capacity)
        growth = capacity * growth_rate * biomass_array
        biomass_array += growth

        diffusion = \
            d * (
                window_total(biomass_array, radius=1) -
                nr_neighbours * biomass_array
            )

        biomass_array += diffusion

        # Iterate over all deer and update biomass in cell they are
        # located in
        # Create a field with for each cell the sum of a property of
        # the deer located in that cell
        deer_weight_per_cell = cell_total(
            deer_location_array, deer_weight_array,
            area_extent, area_shape, cell_size)

        biomass_array = np.maximum(
            0.01,
            biomass_array - (deer_weight_per_cell * 0.0008))

        # Update deer weights
        deer_weight_array = np.minimum(750, deer_weight_array * 1.003)

        # Update deer locations
        enough_food = biomass_array > 0.5
        deer_location_array = gaussian_move(
            deer_location_array, enough_food, area_extent, cell_size,
            area_shape, sigma=cell_size)


        # I/O: write updated state to the data set -----------------------------

        # deer state
        # We don't simulate birth and death, so the collection of deer
        # stays the same during the simulated period of time

        # For the current time cell, the index of the active set
        location.object_tracker.active_set_index.expand(1)[-1] = \
            location.object_tracker.active_set_index[-1] + nr_deer

        # For the current time cell, the IDs of the active deer.
        location.object_tracker.active_object_id.expand(nr_deer)[-nr_deer:] = \
            deer_ids

        # Scalar deer weight per deer
        deer_weight.value.expand(nr_deer)[-nr_deer:] = deer_weight_array

        # Location in space per deer
        location.space_domain.value.expand(nr_deer)[-nr_deer:] = \
            deer_location_array

        # park state
        # For the current time cell, the index of the active set. There is
        # only one park, so we can just use the previous index and add one.
        space_extent.object_tracker.active_set_index.expand(1)[-1] = \
            space_extent.object_tracker.active_set_index[-1] + 1

        # For the current time cell, the IDs of the active objects. In this
        # case, this is always the one and only park object.
        space_extent.object_tracker.active_object_id.expand(1)[-1] = area_id

        # For the current time cell, for the one park, an index into the
        # value array of different_shape / constant_shape values (biomass).
        space_extent.object_tracker.active_object_index.expand(1)[-1] = \
            space_extent.object_tracker.active_object_index[-1] + 1

        # Here we write a 2D raster with biomass per cell per park
        biomass_value.expand(1)[-1:] = biomass_array


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

    random.seed(5)
    np.random.seed(6)

    run_model(
        parsed_dataset_pathname, parsed_nr_timesteps, parsed_area_shape,
        parsed_nr_deer)
