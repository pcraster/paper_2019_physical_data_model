import bpy
import itertools


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


