import lue
import bmesh
import bpy
import itertools
import mathutils
import sys
# import numpy as np


def flatten_list(
        nested_list):
    return list(itertools.chain(*nested_list))


def indentation(
        level,
        tabsize=4):
    return level * tabsize * " "


def indent_lines(
        lines,
        indent_level):

    return ["{}{}".format(indentation(indent_level), line) for line in lines]


def format_section(
        header,
        lines,
        indent_level,
        label=None):

    lines = indent_lines(lines, 1)
    lines = [label if label is not None else header] + lines
    lines = indent_lines(lines, indent_level)

    return lines


def describe_cameras(
        cameras,
        indent_level):

    lines = ["{}".format(camera) for camera in cameras]

    return format_section("cameras", lines, indent_level)


def describe_collection(
        collection,
        indent_level):

    # 'all_objects', 'animation_data_clear', 'animation_data_create', 'bl_rna', 'children', 'copy', 'evaluated_get', 'hide_render', 'hide_select', 'hide_viewport', 'instance_offset', 'is_evaluated', 'is_library_indirect', 'library', 'make_local', 'objects', 'original', 'override_create', 'override_static', 'preview', 'rna_type', 'tag', 'update_tag', 'use_fake_user', 'user_clear', 'user_of_id', 'user_remap', 'users', 'users_dupli_group'

    lines = flatten_list([
        ["name: {}".format(collection.name)],
        ["name_full: {}".format(collection.name_full)],
        describe_objects(collection.all_objects, indent_level,
            label="all_objects"),
        describe_objects(collection.objects, indent_level)
    ])

    return format_section("collection", lines, indent_level)


def describe_collections(
        collections,
        indent_level):

    lines = flatten_list([
        describe_collection(collection, indent_level) for
            collection in collections
    ])

    return format_section("collections", lines, indent_level)


def describe_curves(
        curves,
        indent_level):

    lines = ["{}".format(curve) for curve in curves]

    return format_section("curves", lines, indent_level)


def describe_images(
        images,
        indent_level):

    lines = ["{}".format(image) for image in images]

    return format_section("images", lines, indent_level)


def describe_layer_collection(
        collection,
        indent_level):

    # 'bl_rna', 'children', 'collection', 'exclude', 'has_objects', 'has_selected_objects', 'hide_viewport', 'holdout', 'indirect_only', 'is_visible', 'rna_type'

    lines = flatten_list([
        ["name: {}".format(collection.name)],
        describe_collection(collection.collection, indent_level)
    ])

    return format_section("layer_collection", lines, indent_level)


def describe_lights(
        lights,
        indent_level):

    lines = ["{}".format(light) for light in lights]

    return format_section("lights", lines, indent_level)


def describe_materials(
        materials,
        indent_level):

    lines = ["{}".format(material) for material in materials]

    return format_section("materials", lines, indent_level)


def describe_meshes(
        meshes,
        indent_level):

    lines = ["{}".format(mesh) for mesh in meshes]

    return format_section("meshes", lines, indent_level)


def describe_objects(
        objects,
        indent_level,
        label=None):

    lines = ["{}".format(object) for object in objects]

    return format_section("objects", lines, indent_level, label)


def describe_scene(
        scene,
        indent_level):

    # 'active_clip', 'alembic_export', 'animation_data', 'animation_data_clear', 'animation_data_create', 'audio_distance_model', 'audio_doppler_factor', 'audio_doppler_speed', 'audio_volume', 'background_set', 'bl_rna', 'camera', 'copy', 'cursor', 'cycles', 'cycles_curves', 'display', 'display_settings', 'eevee', 'evaluated_get', 'frame_current', 'frame_current_final', 'frame_end', 'frame_float', 'frame_preview_end', 'frame_preview_start', 'frame_set', 'frame_start', 'frame_step', 'frame_subframe', 'gravity', 'grease_pencil', 'is_evaluated', 'is_library_indirect', 'is_nla_tweakmode', 'keying_sets', 'keying_sets_all', 'library', 'lock_frame_selection_to_range', 'make_local', 'node_tree', 'original', 'override_create', 'override_static', 'preview', 'ray_cast', 'render', 'rigidbody_world', 'rna_type', 'safe_areas', 'sequence_editor', 'sequence_editor_clear', 'sequence_editor_create', 'sequencer_colorspace_settings', 'show_keys_from_selected_only', 'show_subframe', 'statistics', 'sync_mode', 'tag', 'timeline_markers', 'tool_settings', 'transform_orientation_slots', 'unit_settings', 'update_tag', 'use_audio', 'use_audio_scrub', 'use_fake_user', 'use_gravity', 'use_nodes', 'use_preview_range', 'use_stamp_note', 'user_clear', 'user_of_id', 'user_remap', 'users', 'uvedit_aspect', 'view_layers', 'view_settings', 'world'

    lines = flatten_list([
        ["name: {}".format(scene.name)],
        ["name_full: {}".format(scene.name_full)],
        describe_collection(scene.collection, indent_level),
        describe_objects(scene.objects, indent_level),
    ])

    return format_section("scene", lines, indent_level)


def describe_scenes(
        scenes,
        indent_level):

    lines = flatten_list([
        describe_scene(scene, indent_level) for scene in scenes
    ])

    return format_section("scenes", lines, indent_level)


def describe_data(
        data,
        indent_level):
    """
    Blend file data
    """
    # 'actions', 'armatures', 'batch_remove', 'bl_rna', 'brushes', 'cache_files', 'filepath', 'fonts', 'grease_pencils', 'is_dirty', 'is_saved', 'lattices', 'libraries', 'lightprobes', 'linestyles', 'masks', 'metaballs', 'movieclips', 'node_groups', 'paint_curves', 'palettes', 'particles', 'rna_type', 'screens', 'shape_keys', 'sounds', 'speakers', 'texts', 'textures', 'use_autopack', 'user_map', 'version', 'window_managers', 'workspaces', 'worlds'

    lines = \
        describe_cameras(data.cameras, indent_level) + \
        describe_collections(data.collections, indent_level) + \
        describe_curves(data.curves, indent_level) + \
        describe_images(data.images, indent_level) + \
        describe_lights(data.lights, indent_level) + \
        describe_materials(data.materials, indent_level) + \
        describe_meshes(data.meshes, indent_level) + \
        describe_objects(data.objects, indent_level) + \
        describe_scenes(data.scenes, indent_level)

    return format_section("data", lines, indent_level)


# def describe_active_object(
#         object,
#         indent_level):
# 
#     # 'active_material', 'active_material_index', 'active_shape_key', 'active_shape_key_index', 'animation_data', 'animation_data_clear', 'animation_data_create', 'animation_visualization', 'bl_rna', 'bound_box', 'cache_release', 'calc_matrix_camera', 'camera_fit_coords', 'children', 'closest_point_on_mesh', 'collision', 'color', 'constraints', 'convert_space', 'copy', 'cycles', 'cycles_visibility', 'data', 'delta_location', 'delta_rotation_euler', 'delta_rotation_quaternion', 'delta_scale', 'dimensions', 'display', 'display_bounds_type', 'display_type', 'empty_display_size', 'empty_display_type', 'empty_image_depth', 'empty_image_offset', 'empty_image_side', 'evaluated_get', 'face_maps', 'field', 'find_armature', 'grease_pencil_modifiers', 'hide_get', 'hide_render', 'hide_select', 'hide_set', 'hide_viewport', 'holdout_get', 'image_user', 'indirect_only_get', 'instance_collection', 'instance_faces_scale', 'instance_type', 'is_deform_modified', 'is_evaluated', 'is_from_instancer', 'is_from_set', 'is_instancer', 'is_library_indirect', 'is_modified', 'library', 'local_view_get', 'local_view_set', 'location', 'lock_location', 'lock_rotation', 'lock_rotation_w', 'lock_rotations_4d', 'lock_scale', 'make_local', 'material_slots', 'matrix_basis', 'matrix_local', 'matrix_parent_inverse', 'matrix_world', 'mode', 'modifiers', 'motion_path', 'name', 'name_full', 'original', 'override_create', 'override_static', 'parent', 'parent_bone', 'parent_type', 'parent_vertices', 'particle_systems', 'pass_index', 'pose', 'pose_library', 'preview', 'proxy', 'proxy_collection', 'ray_cast', 'rigid_body', 'rigid_body_constraint', 'rna_type', 'rotation_axis_angle', 'rotation_euler', 'rotation_mode', 'rotation_quaternion', 'scale', 'select_get', 'select_set', 'shader_effects', 'shape_key_add', 'shape_key_clear', 'shape_key_remove', 'show_all_edges', 'show_axis', 'show_bounds', 'show_empty_image_orthographic', 'show_empty_image_perspective', 'show_in_front', 'show_instancer_for_render', 'show_instancer_for_viewport', 'show_name', 'show_only_shape_key', 'show_texture_space', 'show_transparent', 'show_wire', 'soft_body', 'tag', 'to_mesh', 'to_mesh_clear', 'track_axis', 'type', 'up_axis', 'update_from_editmode', 'update_tag', 'use_dynamic_topology_sculpting', 'use_empty_image_alpha', 'use_fake_user', 'use_instance_faces_scale', 'use_instance_vertices_rotation', 'use_shape_key_edit_mode', 'user_clear', 'user_of_id', 'user_remap', 'users', 'users_collection', 'users_scene', 'vertex_groups', 'visible_get'
#     # lines = ["{}".format(object) for object in objects]
# 
#     lines = [
#         "type: {}".format(object.type)
#     ]
# 
#     return format_section("active_object", lines, indent_level)


def describe_context(
        context,
        indent_level):
    """
    User context
    """
    # 'active_base', 'active_bone', 'active_gpencil_frame', 'active_gpencil_layer', 'active_object', 'active_operator', 'active_pose_bone', 'area', 'bl_rna', 'blend_data', 'collection', 'copy', 'edit_object', 'editable_bases', 'editable_bones', 'editable_gpencil_layers', 'editable_gpencil_strokes', 'editable_objects', 'engine', 'evaluated_depsgraph_get', 'gizmo_group', 'gpencil_data', 'gpencil_data_owner', 'image_paint_object', 'layer_collection', 'mode', 'object', 'objects_in_mode', 'objects_in_mode_unique_data', 'particle_edit_object', 'pose_object', 'preferences', 'region', 'region_data', 'rna_type', 'scene', 'screen', 'sculpt_object', 'selectable_bases', 'selectable_objects', 'selected_bases', 'selected_bones', 'selected_editable_bases', 'selected_editable_bones', 'selected_editable_fcurves', 'selected_editable_objects', 'selected_editable_sequences', 'selected_objects', 'selected_pose_bones', 'selected_pose_bones_from_active_object', 'selected_sequences', 'sequences', 'space_data', 'tool_settings', 'vertex_paint_object', 'view_layer', 'visible_bases', 'visible_bones', 'visible_gpencil_layers', 'visible_objects', 'visible_pose_bones', 'weight_paint_object', 'window', 'window_manager', 'workspace'
    # 'area', 'bl_rna', 'blend_data', 'collection', 'copy', 'engine', 'evaluated_depsgraph_get', 'gizmo_group', 'layer_collection', 'mode', 'preferences', 'region', 'region_data', 'rna_type', 'scene', 'screen', 'space_data', 'tool_settings', 'view_layer', 'window', 'window_manager', 'workspace'

    lines = \
        describe_collection(context.collection, indent_level) + \
        describe_layer_collection(context.layer_collection, indent_level) + \
        describe_scene(context.scene, indent_level)

    return format_section("context", lines, indent_level)


def print_state():

    lines = \
        describe_data(bpy.data, indent_level=0) + \
        describe_context(bpy.context, indent_level=0)

    print("\n".join(lines))


print_state()


# - How to write current cube to an output file?
# - How to start with empty/clean blender session, without data, popups

# 1. Add vertices. Can we visualize this?
# 2. Connect vertices to edges. Can we visualize this?
# 3. Turn line into curve
# 4. Create circle-curve
# 5. Visualize line as a tube
# 6. Colour each line differently
# 7. Set tube material of all lines

# - Clear scene
# - Add camera
# - Add light
# - Add indication of time boxes (transparent slices?)
# - Is the deer's location representative for the whole time cell?
#    - In that case, draw connected sticks?


# bpy.ops.mesh
# bpy.ops.object
# bpy.ops.<meh>




### # TODO Verify api docs
### mesh = bpy.data.meshes.new("My mesh")
### 
### print_state()


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


# bpy.ops.wm.read_factory_settings(use_empty=True)



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


start_with_empty_file()



scene = bpy.context.scene

# Stuff specific for a phenomenon ----------------------------------------------
# Possibly specific for property-set? Or even domain / property?
# Create a new collection to add objects to and add the collection to
# the scene
collection_name = "deer"
collection = bpy.data.collections.new(collection_name)
scene.collection.children.link(collection)


# # Add a cube
# cube_data = bpy.data.meshes.new("cube_data")
# cube = bpy.data.objects.new("cube_object", cube_data)
# collection.objects.link(cube)
# 
# cube_bmesh = bmesh.new()
# bmesh.ops.create_cube(cube_bmesh, size=1.0)
# cube_bmesh.to_mesh(cube_data)
# cube_bmesh.free()


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



def add_polyline(
        id,
        coordinates):

    # Create the curve data-block
    curve_data = bpy.data.curves.new("curve_data-{}".format(id), type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = 2

    # Map coordinates to spline
    polyline = curve_data.splines.new("POLY")
    polyline.points.add(len(coordinates) - 1)

    for i, coordinate in enumerate(coordinates):
        x, y, z = coordinate
        polyline.points[i].co = (x, y, z, 1)

    # Create object
    curve_object = bpy.data.objects.new(
        "curve_object-{}".format(id), curve_data)
    curve_data.bevel_depth = 0.01

    # Attach to scene and validate context
    collection.objects.link(curve_object)



# TODO Make this x, y, t for each deer
coordinates = [
    [(1, 0, 1), (2, 0, 2), (3, 0, 3)],
    [(2, 2, 1), (2, 2, 2), (3, 2, 3)],
]

for idx in range(len(coordinates)):
    add_polyline(idx, coordinates[idx])




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


scene_extent = [3, 3, 3]  # TODO
cell_extent = [1, 1, 1]  # TODO

# Scene extent in cells
normalized_scene_extent = [
    se / ce for se, ce in zip(scene_extent, cell_extent)]


# Global stuff -----------------------------------------------------------------
collection = bpy.context.collection


# Add a light
light_data = bpy.data.lights.new("light_data", type="SUN")
light = bpy.data.objects.new("light_object", light_data)
collection.objects.link(light)

normalized_light_location = [3, -4.2, 5]
light_location = [
    l * e for l, e in zip(normalized_light_location, normalized_scene_extent)]
light.location = mathutils.Vector(light_location)


# Add a camera
camera_data = bpy.data.cameras.new("camera_data")
camera = bpy.data.objects.new("camera_object", camera_data)
collection.objects.link(camera)

normalized_camera_location = [6, -3, 5]
camera_location = [
    l * e for l, e in zip(normalized_camera_location, normalized_scene_extent)]
camera.location = mathutils.Vector(camera_location)
camera.rotation_euler = mathutils.Euler((0.9, 0.0, 1.1))

scene.camera = camera

# /Global stuff ----------------------------------------------------------------



print_state()

render_still(bpy.context.scene, "PNG", "image.png", "BLENDER_WORKBENCH")










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
