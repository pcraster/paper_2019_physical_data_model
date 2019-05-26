from . debug import *
import bmesh


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
        coordinates):

    # Create the curve data-block
    curve_data = bpy.data.curves.new("curve_data-{}".format(id), type="CURVE")
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
        "curve_object-{}".format(id), curve_data)

    # Setup a material
    material = bpy.data.materials.new("line_material-{}".format(id))
    material.diffuse_color = (0.1, 0.7, 0.1, 1.0)
    material.metallic = 0.7
    material.roughness = 0.5
    # material.use_shadeless = True
    curve_object.data.materials.append(material)

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