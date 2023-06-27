#!/usr/bin/env python
import lue
import docopt
import json
import os.path
import sys


# - Current goals:
#     - Standard gravity: scalar, non-temporal, non-spatial
#     - Biomass: field, temporal, spatial, stationary
#     - Deer: temporal, spatial, mobile
# - Given a dataset with all kinds of data, how to select
#     - Data to visualize
#         - phenomenon
#         - property-set
#         - property
#     - IDs
#     - Locations in time
#     - Locations in space
#     - Visualization parameters (colours, ...)
# 
# {{{json
# {
#     "visualize": {
#         "datasets": [
#             {
#                 "pathname": "my_dataset.lue",
#                 "phenomena": [
#                     {
#                         "name": "my_phenomenon",
#                         "id": {
#                             "slice": {
#                                 "start": 0,
#                                 "end": 500,
#                                 "stride": 10
#                             }
#                         },
#                         "property_sets": [
#                             {
#                                 "name": "my_property_set",
#                                 "domain": {
#                                     "time": {
#                                         "slice": ...
#                                     },
#                                     "space": {
#                                         "extent": {
#                                             "min": [0, 0],
#                                             "max": [500, 500]
#                                         }
#                                     }
#                                 },
#                                 "properties": [
#                                     {
#                                         "name": "my_property",
#                                         "palette": {
#                                             ...
#                                         },
#                                         ...
#                                     }
#                                 ]
#                             }
#                         ]
#                     }
#                 ]
#             }
#         ]
#     }
# }
# }}}
# 
# 
#     - Keep track of extent in time and space
# - Read render JSON, if provided
#     - Iterate over renders, per render
#         - Generate output file
#     - Could be animation in the future too
# - Command to generate visualize.json given a LUE dataset. Can be used
#     to
#     - Visualize everything in a LUE dataset, without having to handcraft
#         a JSON.
#     - As a starting point for creating the JSON/visualization needed.


usage = """\
Visualize data stored in LUE a dataset

Usage:
    {command} [--render=renders] <settings>

Options:
    settings      JSON formatted file with definitions of data to visualize
    renders       JSON formatted file with definitions of renderings to make
    -h --help     Show this screen
""".format(
    command = os.path.basename(__file__))


def parse_id(
        id_json):

    # print(id_json)

    return None


def parse_time_domain(
        time_domain_json):

    return None


def parse_space_domain(
        space_domain_json):

    return None


def visualize_property(
        property_set,
        property_json):

    print(property_set)
    print(property_json)

    # TODO hier verder

    property_name = property_json["name"]
    property = property_set.properties[property_name]

    print(property)

    # TODO hier verder


def visualize_property_set(
        phenomenon,
        property_set_json):

    # TODO Optional!
    # domain_json = property_set_json["domain"]
    # time_domain = parse_time_domain(domain_json["time_domain"]
    # space_domain = parse_time_domain(domain_json["space_domain"]

    property_set_name = property_set_json["name"]
    property_set = phenomenon.property_sets[property_set_name]

    for property in property_set_json["properties"]:
        visualize_property(property_set, property)


def visualize_phenomenon(
        dataset,
        phenomenon_json):

    phenomenon_name = phenomenon_json["name"]
    phenomenon = dataset.phenomena[phenomenon_name]

    # TODO
    # id = parse_id(phenomenon_json["id"])

    for property_set in phenomenon_json["property_sets"]:
        visualize_property_set(phenomenon, property_set)


def visualize_dataset(
        dataset_json):
    """
    Visualize (part of) dataset
    """

    # - Iterate over phenomena, per phenomenon
    #     - Iterate over property-sets, per property-set
    #         - Iterator over properties, per property
    #             - ...

    dataset_pathname = os.path.expandvars(dataset_json["pathname"])
    dataset = lue.open_dataset(dataset_pathname)

    for phenomenon in dataset_json["phenomena"]:
        visualize_phenomenon(dataset, phenomenon)


def render(
        render_settings_json):
    """
    Generate renderings
    """
    assert render_settings_json

    # {
    # "renders": [
    #     {
    #         "pathname": "my_render.png",
    #         "camera": {
    #             "location": [-200, 300],
    #             "angle": ...
    #         }
    #     }
    # ]
    # }

    # TODO


def visualize(
        settings_json,
        render_settings_json):

    # We're in Blender now. Do whatever it takes to visualize the data
    # and (optionally) render images/videos).

    for dataset in settings_json["visualize"]["datasets"]:
        visualize_dataset(dataset)

    if render_settings_json:
        render(render_settings_json)



if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:]
    arguments = docopt.docopt(usage, argv=argv)

    render_settings_pathname = arguments["--render"]
    settings_pathname = arguments["<settings>"]

    settings_json = json.load(open(settings_pathname))
    render_settings_json = json.load(open(render_settings_pathname)) if \
        render_settings_pathname else None

    visualize(settings_json, render_settings_json)
