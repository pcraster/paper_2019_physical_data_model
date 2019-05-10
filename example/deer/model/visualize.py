#!/usr/bin/env python
import lue
import numpy as np
import docopt
import os.path
import sys


usage = """\
Visualize deer model output

Usage:
    {command} <dataset> [--output=<dir>]

Options:
    dataset         Pathname of dataset to read
    --output=<dir>  Pathname of directory to put results in [default: .]
    -h --help       Show this screen
""".format(
    command = os.path.basename(sys.argv[0]))


def visualize(
        dataset_pathname,
        output_pathname):

    dataset = lue.open_dataset(dataset_pathname)

    simulation = dataset.phenomena["simulation"]
    # print(simulation.property_sets["time_extent"].time_domain.value.shape[0])
    nr_timesteps = 5

    # Iterate over all timesteps
    for t in range(nr_timesteps):

        # Visualize locations of deer

        # Visualize biomass field

        pass




if __name__ == "__main__":
    arguments = docopt.docopt(usage)

    dataset_pathname = os.path.abspath(arguments["<dataset>"])
    output_pathname = os.path.abspath(arguments["--output"])

    visualize(dataset_pathname, output_pathname)
