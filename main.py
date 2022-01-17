import bpy
import sys
import os

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)

import sampler
from sampler import *

# this next part forces a reload in case you edit the source after you first start the blender session
import importlib

importlib.reload(sampler)

from math import *
from mathutils import Matrix, Vector

# Change unit to milimeter
bpy.data.scenes["Scene"].unit_settings.scale_length = 0.001
bpy.data.scenes["Scene"].unit_settings.length_unit = 'MILLIMETERS'

# Change grid scale to mm
for area in bpy.data.screens["Scripting"].areas:
    if area.type == 'VIEW_3D':
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.overlay.grid_scale = 0.01
                break

# Switch from Edit mode to Object mode
# if bpy.context.object.mode == "EDIT":
#    bpy.ops.object.mode_set(mode= "OBJECT")


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
def add_cone(r, h, position):
    bpy.ops.mesh.primitive_cone_add(radius1=.25, radius2=r, depth=h,
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


# Select just one object by wrting the name, e.g.
def select_object(obj):
    for object in bpy.context.selected_objects:  # deselect all objects in the scene
        object.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


# Transfering origins of all objects in the scene to buttom-center
# for o in bpy.context.scene.objects:
#    if o.type == 'MESH':
#        origin_to_bottom(o)

# engrave (carve) a text onto the base object
def add_text(base_object, text):
    font_curve = bpy.data.curves.new(type="FONT", name="Font Curve")
    font_curve.body = text
    font_curve.extrude = .1
    font_obj = bpy.data.objects.new(name="Text", object_data=font_curve)
    font_obj.scale[0] = 2
    font_obj.scale[1] = 2

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


def Generator(base_width, base_height, base_depth, cone_diameter, cone_height, cone_shape, points, text, join,
              location):
    # Create a new collection
    collection = bpy.data.collections.new('Texture')
    bpy.context.scene.collection.children.link(collection)

    # Activate the created collection
    layer_collection = bpy.context.view_layer.layer_collection.children[collection.name]
    bpy.context.view_layer.active_layer_collection = layer_collection

    # Add Base
    Base = add_base(base_width, base_height, base_depth, location)

    # Add Cones
    cone_diameter = round(cone_diameter, 2)
    cone_radius = cone_diameter / 2
    for point in points:
        #        print(point)
        Cone = add_cone(cone_radius, cone_height, (point[0], point[1], base_depth))

    # Engrave Text
    print(text)

    add_text(Base, text)

    # Join all objects in the collection together
    if join:
        select_object(Base)
        for obj in bpy.data.collections[collection.name].all_objects:
            #            print(f"Len of obj collection is: {len(bpy.data.collections[collection.name].all_objects)}")
            obj.select_set(True)
        bpy.ops.object.join()


class sample_generator:

    def aniso(self, width, height, space, center):
        samples = []

        current_center = (int(width / 2), int(height / 2))
        dx = center[0] - current_center[0]
        dy = center[1] - current_center[1]

        nx = int(width / space)
        ny = int(height / space)

        for i in range(nx):
            for j in range(ny):
                point = (i * space + dx, j * space + dy)
                print(point)
                samples.append(point)

        return samples

# pd = PoissonDisc(30, 10, 1, 30)
# print(pd)
# center = (0,0)
# pd.sample(center)
# points_list = pd.samples_t
# print("# of Poisson samples: ", len(pd.samples_t))

# Generator(base_width = 30, base_height = 10, base_depth = 1,
#     cone_diameter = .2, cone_height = 1,
#      cone_shape = "", points = pd.samples_t ,
#       text = True, join = True
#      )