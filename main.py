import bpy
import sys
import os

# path of the directory that blend file is inside
dir = os.path.dirname(bpy.data.filepath)

'''
 sys.path is a built-in variable within the sys module.
 It contains a list of directories that the interpreter will search in for the required module.
 We add dir to sys.path to avoid import issues.
 
'''
if not dir in sys.path:
    sys.path.append(dir)

import sampler
from sampler import *

# this next part forces a reload in case you edit the source after you first start the blender session
import importlib
importlib.reload(sampler)

from math import *
from mathutils import Matrix, Vector

# Change units to milimeters
bpy.data.scenes["Scene"].unit_settings.scale_length = 0.001 
bpy.data.scenes["Scene"].unit_settings.length_unit = 'MILLIMETERS'

# Change grid scale to mm in the 3D-View
for area in bpy.data.screens["Scripting"].areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.overlay.grid_scale = 0.01
                break

# Select all the objects in the scene and delete them
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()


# Create the plate cube
def add_base(width, height, depth, position):
    base_width = width  # mm
    base_height = height / width  # changed in the scale
    base_depth = depth / width  # changed in the scale

    bpy.ops.mesh.primitive_cube_add(
        size=base_width, enter_editmode=False,
        align='WORLD', location=position,
        scale=(1, base_height, base_depth)
    )

    base = bpy.context.object  # Pointer to object we have created
    origin_to_bottom(base)  # Move the origin to the buttom center
    base.location = position  # Set the location
    return bpy.context.object


# Cone instance
def add_cone(r_cap, r_base, height, position):
    bpy.ops.mesh.primitive_cone_add(radius1=r_base, radius2=r_cap, depth=height,
                                    enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    cone = bpy.context.object
    origin_to_bottom(cone)
    cone.location = position
    return cone  # Object after operator is active so you can access it by this or "bpy.context.active_object"


# Function to transfer the origin to the buttom-center
def origin_to_bottom(ob, matrix=Matrix()):
    me = ob.data
    mw = ob.matrix_world
    local_verts = [matrix @ Vector(v[:]) for v in ob.bound_box]
    o = sum(local_verts, Vector()) / 8
    o.z = min(v.z for v in local_verts)
    o = matrix.inverted() @ o
    me.transform(Matrix.Translation(-o))
    mw.translation = mw @ o


# Select just one object by passing the refrence
def select_object(obj):
    for object in bpy.context.selected_objects:  # deselect all objects in the scene
        object.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


# engrave (carve) a text onto the base object
def add_text(base_object, text, size, engrave_level):
    font_curve = bpy.data.curves.new(type="FONT", name="Font Curve")
    font_curve.body = text
    font_curve.extrude = .1 * engrave_level # The amount of engrave into the text
    font_obj = bpy.data.objects.new(name="Text", object_data=font_curve)
    font_obj.scale[0] = size # Change size of the writing
    font_obj.scale[1] = size 

    bpy.data.collections["Collection"].objects.link(font_obj)

    select_object(font_obj)
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS', center='MEDIAN')
    font_obj.location = base_object.location
    font_obj.rotation_euler = (radians(180), 0, 0)
    bpy.ops.object.convert(target="MESH")  # convert to mesh
    select_object(base_object)
    bpy.ops.object.modifier_add(type='BOOLEAN')
    bpy.context.object.modifiers["Boolean"].object = font_obj
    bpy.ops.object.modifier_apply(modifier="Boolean")
    select_object(font_obj)
    bpy.ops.object.delete()


# The function that generates the textures automatically.
def Generator(base_shape, base_width, base_height,base_depth, cone_cap_diameter, cone_base_diameter,cone_height, points, text, text_font, text_depth, join, location):
       
    '''  
       
       parameters:
       
       base_shape (str): 'circular' or 'rectangular'
       base_width, base_heigth, base_depth (float): parameters of the base
       cone_cap_diameter, cone_base_diameter, cone_height (float): parameters of the cone
       points ([(x1, y1), (x2, y2), ...]): list of points to place cones
       text (str): text to engrave
       text_font, text_depth (float): font and depth of the engraved text
       join (bool): rigid all if it's TRUE
       location: position to place texture
       
        returns:

            None
       
    '''
    # Create a new collection
    collection = bpy.data.collections.new('Texture')
    bpy.context.scene.collection.children.link(collection)

    # Activate the created collection
    layer_collection = bpy.context.view_layer.layer_collection.children[collection.name]
    bpy.context.view_layer.active_layer_collection = layer_collection

    # Add Base
    base = add_base(base_width, base_height, base_depth, location)

    # Add Cones
    cone_cap_radius = round(cone_cap_diameter / 2, 2)
    
    cone_base_radius= round(cone_base_diameter / 2, 2)
    
    for point in points:
        Cone = add_cone(cone_cap_radius, cone_base_radius , cone_height, (point[0], point[1], base_depth))

    # Engrave Text
    print(text)

    add_text(base, text, text_font, text_depth)

    # Join all objects in the collection together
    if join:
        select_object(base)
        for obj in bpy.data.collections[collection.name].all_objects:
            obj.select_set(True)
        bpy.ops.object.join()
    
    if base_shape == "circular":
        
        # create the cylinder
        cylinder = bpy.ops.mesh.primitive_cylinder_add( radius = base_width / 2 , depth = 100, location = (0,0,0) )
        cylinder = bpy.data.objects['Cylinder']
        
        # define boolean modifire
        boolean_modifier = cylinder.modifiers.new(type="BOOLEAN", name="bool")
        boolean_modifier.object = base
        boolean_modifier.operation = 'INTERSECT'
        boolean_modifier.solver = 'EXACT'
        boolean_modifier.use_self = True
        
        # apply modifire
        target_obj = bpy.context.active_object
        for modifier in target_obj.modifiers:
            bpy.ops.object.modifier_apply(modifier=modifier.name)
        
        # remove the rectangular object
        select_object(base)
        bpy.ops.object.delete()


        
    

