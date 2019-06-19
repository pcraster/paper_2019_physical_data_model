from lue_blender import *
import lue

import bmesh
import bpy
import mathutils

import seaborn as sns

import numpy as np

import os.path
import random
import sys


# 6. Colour each line differently
# 7. Set tube material of all lines

# - Add indication of time boxes (transparent slices?)
# - Is the deer's location representative for the whole time cell?
#    - In that case, draw connected sticks?


# bpy.ops.mesh
# bpy.ops.object
# bpy.ops.<meh>


### # TODO Verify api docs
### mesh = bpy.data.meshes.new("My mesh")


### # vertices:
### # - 1D array
### # - 3 values (numpy.float32) per vertex
### mesh.vertices.add(1)
### vertices = numpy.array([0.0, 0.0, 0.0], dtype=np.float32)
### mesh.vertices.foreach_set("co", vertices)
### 
### 
### # faces:
### # - 1D array
### # - 4 indices (numpy.uint32) per triangle/quad
### # - for a triangle set last index to 0
### # mesh.tessfaces.add(num_triangles)
### # mesh.tessfaces.foreach_set("vertices_raw", faces)


def add_track(
        collection,
        property_set,
        object_id,
        scale_z):

    time_domain = property_set.time_domain
    space_domain = property_set.space_domain

    # For each location in time, find the location in space of the object
    # with ID object_id
    # TODO Handle the case that an object is not present at a location in time

    object_tracker = property_set.object_tracker
    active_set_idxs = np.append(
        object_tracker.active_set_index[:],
        np.array(
            [object_tracker.active_object_id.nr_ids], dtype=lue.dtype.Index))
    set_idx = 0

    min_x, min_y, max_x, max_y, min_z, max_z = 6 * (None,)

    materials = create_materials(
        colors=sns.color_palette("Set1", 9), alpha=1.0)
    set_material_metallic(materials, 0.7)
    set_material_roughness(materials, 0.5)

    for b in range(time_domain.value.nr_boxes):
        for t in range(time_domain.value.nr_counts):
            nr_time_cells = time_domain.value.count[b]
            locations = np.empty(shape=(nr_time_cells, 3), dtype=np.float64)

            for c in range(nr_time_cells):
                # Given IDs of active set at current location in time,
                # find index of current object in the set

                set_begin_idx = int(active_set_idxs[set_idx])
                set_end_idx = active_set_idxs[set_idx + 1]
                object_ids = \
                    object_tracker.active_object_id[set_begin_idx:set_end_idx]

                object_idx = next(
                    (idx for idx, value in np.ndenumerate(object_ids)
                        if value == object_id), None)
                assert object_idx is not None
                object_idx = object_idx[0]

                x, y = space_domain.value[set_begin_idx + object_idx]
                z = (c + 0.5) * scale_z  # Connect centers of time cells

                # Add location of object to array
                locations[c] = [x, y, z]

                min_x = x if min_x is None else min(x, min_x)
                max_x = x if max_x is None else max(x, max_x)
                min_y = y if min_y is None else min(y, min_y)
                max_y = y if max_y is None else max(y, max_y)
                min_z = c if min_z is None else min(z, min_z)
                max_z = c if max_z is None else max(z, max_z)

                set_idx += 1

            add_polyline(collection, object_id, locations, materials)

    return min_x, min_y, max_x, max_y, min_z, max_z



    ### for i in range(len(active_set_idxs) - 1):
    ###     set_idx_begin = active_set_idxs[i]
    ###     set_idx_end = active_set_idxs[i + 1]
    ###     active_set_size = set_idx_end - set_idx_begin
    ###     object_ids = object_tracker.active_object_id[set_idx_begin:set_idx_end]

    ###     print(object_ids)

    ###     for object_id in object_ids:

    ###         add_track(collection, property_set, object_id)



    # assert False, "OK..."


    # # Visualize x, y, t tracks of each deer
    # # - Iterate over each deer id
    # # - Collect x, y, t for each one
    # # - Create

    # # TODO Make this x, y, t for each deer
    # coordinates = [
    #     [(1, 0, 1), (2, 0, 2), (3, 0, 3)],
    #     [(2, 2, 1), (2, 2, 2), (3, 2, 3)],
    # ]

    # for idx in range(len(coordinates)):
    #     add_polyline(collection, idx, coordinates[idx])


def assert_time_domain_as_expected(
        property_set):
    assert property_set.has_time_domain
    time_domain = property_set.time_domain
    assert time_domain.configuration.item_type == lue.TimeDomainItemType.cell
    assert time_domain.value.nr_boxes == 1
    assert time_domain.value.nr_counts == 1
    assert time_domain.value.count.nr_arrays == time_domain.value.nr_boxes


def add_tracks(
        collection,
        property_set,
        scale_z):

    """
    """
    assert_time_domain_as_expected(property_set)

    assert property_set.has_space_domain
    space_domain = property_set.space_domain
    assert space_domain.configuration.mobility == lue.Mobility.mobile
    assert space_domain.configuration.item_type == lue.SpaceDomainItemType.point
    assert space_domain.value.rank == 1  # Per (t, x, y, object) an x, y
    assert len(space_domain.value.array_shape) == 1  # A 1D array ...
    assert space_domain.value.array_shape[0] == 2  # ... of two values: x, y

    # These are the IDs of object for which information is stored. For
    # each of them, add a track to the collection.
    object_ids = set(property_set.object_tracker.active_object_id[:])
    print(object_ids)

    min_x, min_y, max_x, max_y, min_z, max_z = 6 * (None,)

    for object_id in object_ids:
        min_x_track, min_y_track, \
        max_x_track, max_y_track, \
        min_z_track, max_z_track = \
            add_track(collection, property_set, object_id, scale_z)

        ### min_x = min_x_track if min_x is None else min(min_x_track, min_x)
        ### max_x = max_x_track if max_x is None else max(max_x_track, max_x)
        ### min_y = min_y_track if min_y is None else min(min_y_track, min_y)
        ### max_y = max_y_track if max_y is None else max(max_y_track, max_y)
        ### min_z = min_z_track if min_z is None else min(min_z_track, min_z)
        ### max_z = max_z_track if max_z is None else max(max_z_track, max_z)

        min_x, max_x, min_y, max_y, min_z, max_z = update_extent(
            min_x, max_x, min_y, max_y, min_z, max_z,
            min_x_track, max_x_track, min_y_track, max_y_track,
            min_z_track, max_z_track)

    return min_x, min_y, max_x, max_y, min_z, max_z


def add_field(
        collection,
        property_set,
        property_name,
        scale_z):

    assert_time_domain_as_expected(property_set)
    time_domain = property_set.time_domain

    assert property_set.has_space_domain
    space_domain = property_set.space_domain
    assert space_domain.configuration.mobility == lue.Mobility.stationary
    assert space_domain.configuration.item_type == lue.SpaceDomainItemType.box

    assert space_domain.value.nr_boxes == 1
    assert len(space_domain.value.array_shape) == 1  # A 1D array ...
    # ... of four values: x1, y1, x2, y2
    assert space_domain.value.array_shape[0] == 4  # Implies rank == 2

    min_x, min_y, max_x, max_y = space_domain.value[0]
    min_z, max_z = 2 * (None,)

    property = property_set.properties[property_name]
    assert property.space_is_discretized

    discretization = property.space_discretization_property()

    assert isinstance(discretization, lue.same_shape.Property)
    assert discretization.value.nr_arrays == 1
    assert len(discretization.value.array_shape) == 1
    assert discretization.value.array_shape[0] == 2  # nr_rows, nr_cols

    nr_rows, nr_cols = discretization.value[0]

    # Given the location and discretization of the raster, generate a
    # collection of planes: for each raster cell a plane with its
    # own colour.

    # Here we assume/know that there is only one object. For that object,
    # we need to output rasters for each location in time.
    object_tracker = property_set.object_tracker
    active_set_idxs = np.append(
        object_tracker.active_set_index[:],
        np.array(
            [object_tracker.active_object_id.nr_ids], dtype=lue.dtype.Index))
    set_idx = 0

    min_z = 0
    max_z = 100

    assert len(set(object_tracker.active_object_id[:])) == 1
    object_id = object_tracker.active_object_id[0]

    all_arrays = property.value[object_id][:]
    min_value = np.min(all_arrays)
    max_value = np.max(all_arrays)
    del all_arrays

    for b in range(time_domain.value.nr_boxes):
        for t in range(time_domain.value.nr_counts):
            nr_time_cells = time_domain.value.count[b]

            for c in range(0, nr_time_cells):

                # A 2D array for the current location in time can be
                # found in the property value
                array = property.value[object_id][c]

                grid_data = bpy.data.meshes.new(
                    "grid_data-{}-{}".format(property_name, c))
                grid_object = bpy.data.objects.new(
                    "grid_object-{}-{}".format(property_name, c), grid_data)

                collection.objects.link(grid_object)

                # Create empty bmesh
                grid_mesh = bmesh.new()

                # Create point grid. To end up with nr_rows x nr_cols cells,
                # add one to each dimension.
                bmesh.ops.create_grid(
                    grid_mesh, x_segments=nr_cols+1, y_segments=nr_rows+1,
                    size=1.0)

                assign_colors_to_grid_cells(
                    grid_data,
                    sns.cubehelix_palette(8, rot=-0.4),
                    # sns.color_palette(),
                    grid_mesh, array, min_value, max_value)

                z = (c + 0.5) * scale_z  # Connect centers of time cells

                extents = [
                    max_x - min_x,
                    max_y - min_y,
                    0]
                grid_object.scale = (
                    extents[0] / 2,
                    extents[1] / 2,
                    1.0)
                grid_object.location = (
                    min_x + extents[0] / 2,
                    min_y + extents[1] / 2,
                    z)

                # Convert bmesh to Blender representation
                grid_mesh.to_mesh(grid_data)
                grid_mesh.free()


                min_z = c if min_z is None else min(z, min_z)
                max_z = c if max_z is None else max(z, max_z)

                if c % 10 != 0:
                    # Initially, hide most layers
                    grid_object.hide_set(True)
                    grid_object.hide_render = True

    return min_x, min_y, max_x, max_y, min_z, max_z


# - Open dataset
# - Find deer location property set

start_with_empty_file()
scene = bpy.context.scene

dataset_name = "deer.lue"
dataset_pathname = os.path.join("/tmp/deer", dataset_name)
dataset = lue.open_dataset(dataset_pathname)
# dataset_collection = scene.collection
dataset_collection = bpy.data.collections.new(dataset_name)
scene.collection.children.link(dataset_collection)


# Stuff specific for a phenomenon ----------------------------------------------
# Possibly specific for property-set? Or even domain / property?
min_x, min_y, max_x, max_y, min_z, max_z = 6 * (None,)

# Create a new collection to add objects to and add the collection to
# the scene

# ------------------------------------------------------------------------------
scale_z = 25.0

phenomenon_name = "deer"
phenomenon = dataset.phenomena[phenomenon_name]
phenomenon_collection = bpy.data.collections.new(phenomenon_name)
dataset_collection.children.link(phenomenon_collection)

property_set_name = "location"
property_set = phenomenon.property_sets[property_set_name]
property_set_collection = bpy.data.collections.new(property_set_name)
phenomenon_collection.children.link(property_set_collection)

min_x_new, min_y_new, \
max_x_new, max_y_new, \
min_z_new, max_z_new = \
    add_tracks(property_set_collection, property_set, scale_z)
min_x, max_x, min_y, max_y, min_z, max_z = update_extent(
    min_x, max_x, min_y, max_y, min_z, max_z,
    min_x_new, max_x_new, min_y_new, max_y_new, min_z_new, max_z_new)

# ------------------------------------------------------------------------------
phenomenon_name = "area"
phenomenon = dataset.phenomena[phenomenon_name]
phenomenon_collection = bpy.data.collections.new(phenomenon_name)
dataset_collection.children.link(phenomenon_collection)

property_set_name = "space_extent"
property_set = phenomenon.property_sets[property_set_name]
property_set_collection = bpy.data.collections.new(property_set_name)
phenomenon_collection.children.link(property_set_collection)

property_name = "biomass"
property_collection = bpy.data.collections.new(property_name)
property_set_collection.children.link(property_collection)

min_x_new, min_y_new, \
max_x_new, max_y_new, \
min_z_new, max_z_new = \
    add_field(property_collection, property_set, property_name, scale_z)
min_x, max_x, min_y, max_y, min_z, max_z = update_extent(
    min_x, max_x, min_y, max_y, min_z, max_z,
    min_x_new, max_x_new, min_y_new, max_y_new, min_z_new, max_z_new)


### # Add a cube
### cube_data = bpy.data.meshes.new("cube_data")
### cube_object = bpy.data.objects.new("cube_object", cube_data)
### collection.objects.link(cube_object)
### 
### cube_bmesh = bmesh.new()
### bmesh.ops.create_cube(cube_bmesh, size=1.0)
### cube_bmesh.to_mesh(cube_data)
### cube_bmesh.free()


# Add a vertex
### vertex_data = bpy.data.meshes.new("vertex")
### vertex = bpy.data.objects.new("vertex", vertex_data)
### collection.objects.link(vertex)
### 
### vertex_bmesh = bmesh.new()
### bmesh.ops.create_cube(vertex_bmesh, size=1.0)
### vertex_bmesh.to_mesh(vertex_data)
### vertex_bmesh.free()


# vertex_coordinates = [(0, 0, 0), (1, 1, 1)]  # 2 verts made with XYZ coords
# 
# # Add a new mesh
# mesh_data = bpy.data.meshes.new("mesh_data")
# 
# # Add a new object using the mesh
# mesh_object = bpy.data.objects.new("mesh_object", mesh_data)
# 
# # Put the object into the scene
# collection.objects.link(mesh_object)
# 
# edge_bmesh = bmesh.new()
# 
# # Add vertices
# vertices = [edge_bmesh.verts.new(v) for v in vertex_coordinates]
# # for v in vertex_coordinates:
# #     edge_bmesh.verts.new(v)
# 
# # Add edges
# edge_bmesh.edges.new((vertices[0], vertices[1]))
# 
# print(dir(edge_bmesh))
# # 'calc_loop_triangles', 'calc_volume', 'clea$', 'copy', 'edges', 'faces', 'free', 'from_mesh', 'from_object', 'is_valid', 'is_wrapped', 'loops', 'normal_update', 'select_flush', 'select_flush_mode', 'select_history', 'select_mode', 'to_mesh', 'transform', 'verts'
# 
# # Make the bmesh the object's mesh
# edge_bmesh.to_mesh(mesh_data)
# edge_bmesh.free()






# The solution was simple enough. I changed
# 
# polyline.points.add(len(coords))
# to
# 
# polyline.points.add(len(coords)-1)



### mesh = bpy.data.meshes.new("mesh")  # add a new mesh
### obj = bpy.data.objects.new("MyObject", mesh)  # add a new object using the mesh
### 
### scene = bpy.context.scene
### scene.objects.link(obj)  # put the object into the scene (link)
### scene.objects.active = obj  # set as the active object in the scene
### obj.select = True  # select object

### mesh = bpy.context.object.data
### bm = bmesh.new()
### 
### for v in verts:
###     bm.verts.new(v)  # add a new vert
### 
### # make the bmesh the object's mesh
### bm.to_mesh(mesh)  





# v1 = vertex_data.verts.new((2.0, 2.0, 2.0))
# v2 = vertex_data.verts.new((-2.0, 2.0, 2.0))
# v3 = vertex_data.verts.new((-2.0, -2.0, 2.0))
# 
# vertex_data.edges.new((v1, v2))
# vertex_data.edges.new((v2, v3))

# bmesh.update_edit_mesh(bpy.context.object.data)

### obj = bpy.context.object
### me = obj.data
### bm = bmesh.from_edit_mesh(me)
### 
### v1 = bm.verts.new((2.0, 2.0, 2.0))
### v2 = bm.verts.new((-2.0, 2.0, 2.0))
### v3 = bm.verts.new((-2.0, -2.0, 2.0))
### 
### bm.edges.new((v1, v2))
### bm.edges.new((v2, v3))
### 
### bmesh.update_edit_mesh(obj.data)


# /Stuff specific for a phenomenon ---------------------------------------------

# Global stuff -----------------------------------------------------------------

collection = bpy.context.collection

# print(dir(collection))
# print(dir(collection.objects))
# print(dir(collection.all_objects))
# assert False

### # Scale z (time in this case) to make its extent similar to the extents
### # in x and y.
### print(dir(collection))
### print(dir(collection.objects))

# print(dir(collection))
# print(collection.objects)
# print(dir(collection.all_objects))


### # TODO Calculate scale in z
### for object in dataset_collection.all_objects:
###     print(object.type)
###     if object.type in ["CURVE", "MESH"]:
###         object.scale[2] = 25


scene_extent = [
    max_x - min_x,
    max_y - min_y,
    max_z - min_z]
scene_origin = [
    min_x + (scene_extent[0] / 2),
    min_y + (scene_extent[1] / 2),
    min_z + (scene_extent[2] / 2)]
max_extent = max(scene_extent)

print("scene_extent: {}".format(scene_extent))
print("data extent:\n  x: {} - {}\n  y: {} - {}\n  z: {} - {}".format(
    min_x, max_x, min_y, max_y, min_z, max_z))
print("scene_origin: {}".format(scene_origin))
print("max_extent: {}".format(max_extent))




# Add a light
light_data = bpy.data.lights.new("light_data", type="SUN")
light = bpy.data.objects.new("light_object", light_data)
collection.objects.link(light)

# Orient light at scene
light_location = mathutils.Vector([
    min_x + 2 * scene_extent[0],
    min_y - 2 * scene_extent[1],
    min_z + 2 * scene_extent[2]])
light.location = light_location


# Add a camera
camera_data = bpy.data.cameras.new("camera_data")
camera_object = bpy.data.objects.new("camera_object", camera_data)
# TODO Make this dependent on the extent of the data and the distance
#     of the camera from the scene
camera_object.data.clip_end = 3 * max_extent
collection.objects.link(camera_object)

# Orient camera at scene
# TODO Determine good location, given extent of object and camera angles
camera_location = mathutils.Vector([
    min_x + 1.3 * scene_extent[0],
    min_y - 1.3 * scene_extent[1],
    min_z + 3 * scene_extent[2]])
camera_object.location = camera_location
camera_object.data.lens = 100.0
# camera_object.rotation_euler = mathutils.Euler((0.9, 0.0, 1.1))

# Update layer, otherwise the camera's world matrix is not set yet
bpy.context.view_layer.update()

look_at(camera_object, mathutils.Vector(scene_origin))

scene.camera = camera_object


# Add an empty at the base of the scene, in the middle and do a 'view
# selected' on it. That way, when the interface starts, we can hover around
# the objects in the scene instead of the origin.
empty_data = None
empty_object = bpy.data.objects.new("empty_object", empty_data)
collection.objects.link(empty_object)

empty_object.empty_display_size = 2
empty_object.empty_display_type = "PLAIN_AXES"
empty_location = [
    min_x + 0.5 * scene_extent[0],
    min_y + 0.5 * scene_extent[1],
    min_z,
]
empty_object.location = mathutils.Vector(empty_location)
# empty_object.hide_set(True)


bounding_box_object = add_bounding_box(
    collection, min_x, max_x, min_y, max_y, min_z, max_z)
bounding_box_object.hide_set(True)
bounding_box_object.hide_render = True

configure_interface(clip_end=3*max_extent)

# /Global stuff ----------------------------------------------------------------



# print_state()

# TODO Fix rendering of objects shadowed by upper objects (sun light, cycles)
# Given sun light:
# - For each grid object:
#     - Object | Cycles settings | ray visibility | shadow -> off
# Other option: attach a light to the camera? Is that possible?

render_still(bpy.context.scene, "PNG", "deer.png", "BLENDER_WORKBENCH")
save_as("deer.blend")

# TODO Auto-save blender file



























### obj = bpy.context.object
### me = obj.data
### bm = bmesh.from_edit_mesh(me)
### 
### v1 = bm.verts.new((2.0, 2.0, 2.0))
### v2 = bm.verts.new((-2.0, 2.0, 2.0))
### v3 = bm.verts.new((-2.0, -2.0, 2.0))
### 
### bm.edges.new((v1, v2))
### bm.edges.new((v2, v3))
### 
### bmesh.update_edit_mesh(obj.data)



### # High level
### 
### # vertices = [(0, 0, 0), (1, 2, 3), ...]
### # edges = [(0, 1), ...]
### # faces = [(0, 1, 2), (3, 4, 5, 6, 7), ...]
### # mesh = bpy.data.meshes.new("My mesh")
### # mesh.from_pydata(vertices, edges, faces)
### # mesh.update()
### # # Add mesh to scene...
### 
### # from_pydata(vertices, edges, faces)
### #     Make a mesh from a list of vertices/edges/faces.
### #     Until we have a nicer way to make geometry, use this.



### To tackle the problem requires several steps.
### 
### Avoid using ops.
### Create the cube.
### Avoid using ops.
### Create a camera, only cameras can render images.
### Avoid using ops.
### Create a light, otherwise your unlit cube is going to be black.
### Render an image (this can only by done with bpy.ops).
### First import the necessary modules. Store the current scene inside a variable, that way we can access it later on.
### 
### import bpy
### import mathutils
### 
### scene = bpy.context.scene
### A method for creating the cube is found here. I have blatantly copied it. We will have to import the bmesh module.
### 
### # Create the cube
### mesh = bpy.data.meshes.new('cube')
### ob = bpy.data.objects.new('cube', mesh)
### 
### scene.objects.link(ob)
### 
### import bmesh
### bm = bmesh.new()
### bmesh.ops.create_cube(bm, size=1.0)
### bm.to_mesh(mesh)
### bm.free()
### Creating the light need a lamp data-block and an object block as well. light.location changes the position of the lamp object. If it is at (0, 0, 0), it will not affect the result, because it will be stuck inside the cube.
### 
### # Create a light
### light_data = bpy.data.lamps.new('light', type='POINT')
### light = bpy.data.objects.new('light', light_data)
### scene.objects.link(light)
### light.location = mathutils.Vector((3, -4.2, 5))
### The camera has to be repositioned and reoriented.
### 
### # Create the camera
### cam_data = bpy.data.cameras.new('camera')
### cam = bpy.data.objects.new('camera', cam_data)
### scene.objects.link(cam)
### scene.camera = cam
### 
### cam.location = mathutils.Vector((6, -3, 5))
### cam.rotation_euler = mathutils.Euler((0.9, 0.0, 1.1))
### Set the render settings of the current scene, and render it. write_still is necessary if you want the result to be written to the filepath. There will be no visual feedback during the render process.
### 
### # render settings
### scene.render.image_settings.file_format = 'PNG'
### scene.render.filepath = "F:/image.png"
### bpy.ops.render.render(write_still = 1)
