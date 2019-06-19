from . debug import *
import bmesh
import numpy as np
import math


def save_as(
        pathname):

    bpy.ops.wm.save_as_mainfile(filepath=pathname, check_existing=False)


def render_still(
        scene,
        format,
        pathname,
        engine):

    # engine must be one of
    # - BLENDER_EEVEE
    # - CYCLES
    # - BLENDER_WORKBENCH

    scene.render.engine = engine
    scene.render.film_transparent = True

    scene.render.image_settings.file_format = format
    scene.render.filepath = pathname
    bpy.ops.render.render(write_still=True)


def start_with_empty_file():
    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.context.preferences.view.show_splash = False


def look_at(
        camera_object,
        location):

    camera_location = camera_object.matrix_world.to_translation()

    direction = location - camera_location

    # Direct the camera's '-Z', and use its 'Y' as up
    rot_quat = direction.to_track_quat("-Z", "Y")

    camera_object.rotation_euler = rot_quat.to_euler()


def add_polyline(
        collection,
        id,
        coordinates,
        materials):

    # Create the curve data-block
    curve_data = bpy.data.curves.new(
        "curve_data-{:02}".format(id), type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.fill_mode = "FULL"
    curve_data.resolution_u = 4
    # curve_data.bevel_depth = 0.05
    # TODO pass this in. Depends on data extent things.
    curve_data.bevel_depth = 25
    # curve_data.bevel_depth = 0.01

    # Map coordinates to spline
    polyline = curve_data.splines.new("POLY")
    polyline.points.add(len(coordinates) - 1)

    for i in range(len(coordinates)):
        x, y, z = coordinates[i]
        polyline.points[i].co = (x, y, z, 1.0)

    # Create object
    curve_object = bpy.data.objects.new(
        "curve_object-{:02}".format(id), curve_data)

    curve_object.data.materials.append(
        materials[id % type(id)(len(materials))])

    # Attach to scene and validate context
    collection.objects.link(curve_object)


def configure_interface(
        clip_end):

    # Manipulate the 3D view and camera
    # We cannot use bpy.context.screen because it doesn't exist yet (None). Maybe
    # because the Blender interface is still starting up. Here, we find the
    # screen in the bpy.data instead of bpy.context. Apparently, this screen
    # is used when creating the interface.
    for screen in bpy.data.screens:
        if screen.name == "Layout":
            for area in screen.areas:
                if area.type == "VIEW_3D":
                    for space in area.spaces:
                        if space.type == "VIEW_3D":
                            # Enlarge the clipping plane. Given the default,
                            # we likely won't see anything.
                            space.clip_end = clip_end

                            # Start with looking through the camera
                            space.region_3d.view_perspective = "CAMERA"

                            # Lock camera to view
                            space.lock_camera = True




### w = 1
### 
### def MakeFilledPolyLine(collection, objname, curvename, cLists):
###     curvedata = bpy.data.curves.new(name=curvename, type='CURVE')
###     curvedata.dimensions = '2D'  
### 
###     odata = bpy.data.objects.new(objname, curvedata)
###     odata.location = (0,0,0) # object origin
### 
###     collection.objects.link(odata)
###     # bpy.context.scene.objects.link(odata)
### 
###     for cList in cLists:
###         polyline = curvedata.splines.new('POLY')
###         polyline.points.add(len(cList)-1)
###         for num in range(len(cList)):
###             # --------------------- = x            , y            , z, w
###             polyline.points[num].co = cList[num][0], cList[num][1], cList[num][2], w
### 
###         polyline.order_u = len(polyline.points)-1
###         polyline.use_endpoint_u = True
###         polyline.use_cyclic_u = True  # this closes each loop
### 
###     return odata


### vectors = [
###     [[0,0], [10,0], [10,10], [0,10]], 
###     [[1,1], [1,2], [2,2], [2,1]]
### ]
### MakeFilledPolyLine("NameOfMyCurveObject", "NameOfMyCurve", vectors)


def add_plane(
        collection,
        min_x,
        max_x,
        min_y,
        max_y,
        z):


    # TODO Generalize add_polyline to make it usable from here
    # - Define material
    # - Pass some id / name
    # - ...

    return

    ### vectors = [
    ###     [
    ###         [min_x, min_y, z],
    ###         [min_x, max_y, z],
    ###         [max_x, max_y, z],
    ###         [max_x, min_y, z],
    ###     ]
    ### ]

    ### obj = MakeFilledPolyLine(collection, "my_curve_object", "my_curve", vectors)
    ### return obj

    ### plane_data = bpy.data.meshes.new("plane_data")
    ### plane_object = bpy.data.objects.new("plane_object", plane_data)

    ### material = bpy.data.materials.new("plane")
    ### material.diffuse_color = (0.0, 0.0, 0.0, 0.4)
    ### material.metallic = 0.0
    ### material.roughness = 0.5
    ### plane_object.data.materials.append(material)

    ### collection.objects.link(plane_object)

    ### plane_bmesh = bmesh.new()

    ### bmesh.ops.create_cube(plane_bmesh, size=1.0)
    ### # meh = bpy.ops.mesh.primitive_plane_add(size=1.0)

    ### extents = [
    ###     max_x - min_x,
    ###     max_y - min_y,
    ###     0]
    ### plane_object.scale = (
    ###     extents[0] / 2,
    ###     extents[1] / 2,
    ###     1.0)
    ### plane_object.location = (
    ###     min_x + extents[0] / 2,
    ###     min_y + extents[1] / 2,
    ###     z)

    ### plane_bmesh.to_mesh(plane_data)
    ### plane_bmesh.free()

    ### return plane_object


def add_bounding_box(
        collection,
        min_x,
        max_x,
        min_y,
        max_y,
        min_z,
        max_z):

    bounding_box_data = bpy.data.meshes.new("bounding_box_data")
    bounding_box_object = bpy.data.objects.new(
        "bounding_box_object", bounding_box_data)

    material = bpy.data.materials.new("bounding_box")
    material.diffuse_color = (0.0, 0.0, 0.0, 0.4)
    material.metallic = 0.0
    material.roughness = 0.5
    bounding_box_object.data.materials.append(material)

    collection.objects.link(bounding_box_object)

    bounding_box_bmesh = bmesh.new()

    bmesh.ops.create_cube(bounding_box_bmesh, size=1.0)

    extents = [
        max_x - min_x,
        max_y - min_y,
        max_z - min_z]

    bounding_box_object.scale = extents
    bounding_box_object.location = (
        min_x + extents[0] / 2, min_y + extents[1] / 2, min_z + extents[2] / 2)

    bounding_box_bmesh.to_mesh(bounding_box_data)
    bounding_box_bmesh.free()

    return bounding_box_object


def update_extent(
    min_x_cur, max_x_cur, min_y_cur, max_y_cur, min_z_cur, max_z_cur,
    min_x_new, max_x_new, min_y_new, max_y_new, min_z_new, max_z_new):

    min_x = min_x_new if min_x_cur is None else min(min_x_cur, min_x_new)
    max_x = max_x_new if max_x_cur is None else max(max_x_cur, max_x_new)
    min_y = min_y_new if min_y_cur is None else min(min_y_cur, min_y_new)
    max_y = max_y_new if max_y_cur is None else max(max_y_cur, max_y_new)
    min_z = min_z_new if min_z_cur is None else min(min_z_cur, min_z_new)
    max_z = max_z_new if max_z_cur is None else max(max_z_cur, max_z_new)

    return min_x, max_x, min_y, max_y, min_z, max_z


def create_materials(
        colors,
        alpha):
    materials = []

    for color in colors:
        material = bpy.data.materials.new("")
        material.diffuse_color = *color, alpha
        materials.append(material)

    return materials


def set_material_metallic(
        materials,
        metallic):

    for material in materials:
        material.metallic = 0.7


def set_material_roughness(
        materials,
        roughness):

    for material in materials:
        material.roughness = 0.7


def classify(
        value,
        min_value,
        max_value,
        nr_classes):

    assert min_value <= value <= max_value

    value_range = max_value - min_value

    if value_range == 0:
        return 0

    class_range = value_range / nr_classes
    idx = (value - min_value) / class_range

    return int(math.floor(idx))


def assign_colors_to_grid_cells(
        grid_data,
        palette,
        grid_mesh,
        array,
        min_value,
        max_value):

    assert array.size > 0
    assert len(palette) > 0

    nr_materials = len(palette)

    materials = create_materials(palette, alpha=1.0)

    for material in materials:
        grid_data.materials.append(material)

    grid_mesh.faces.ensure_lookup_table()

    nr_rows, nr_cols = array.shape
    i = 0

    for r in range(nr_rows):
        for c in range(nr_cols):
            # Assign a material based on the cell's value
            cell_value = array[r][c]
            material_idx = classify(
                cell_value, min_value, max_value, nr_materials)
            grid_mesh.faces[i].material_index = material_idx

            i += 1
