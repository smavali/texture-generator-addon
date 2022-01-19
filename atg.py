bl_info = {
    "name": "Texture Generator",
    "author": "Sina Mavali",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Texture Generator",
    "description": "",
    "warning": "",
    "wiki_url": "",
    "category": "3D View"}

import bpy
import time
import sys
import os
from math import *
from mathutils import Matrix, Vector
import numpy as np
from pathlib import Path


# path of the directory that blend file is inside
dir = os.path.dirname(bpy.data.filepath)


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

def export():
    dir = os.path.dirname(bpy.data.filepath)
    print(dir)

    context = bpy.context
    scene = context.scene
    viewlayer = context.view_layer

    path = dir + '/stl files'
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)

    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(path)


    obs = [o for o in scene.objects if o.type == 'MESH']
    print (obs)
    bpy.ops.object.select_all(action='SELECT')    

    path = Path(path)
    for ob in obs:
        viewlayer.objects.active = ob
        ob.select_set(True)
        stl_path = path / f"{ob.name}.stl"
        print(stl_path)
        bpy.ops.export_mesh.stl(
                filepath=str(stl_path),
                use_selection=True)
        ob.select_set(False)


class Random:
    def random_sampler(n, width, height, thresh, center):
        """
        params:
            n: number of samples
            width & height: dimentions of sample surface
            thresh: distance from edges
        return:
            list of points
        """
        x_center = center[0]
        y_center = center[1]

        xy_min = [x_center - width / 2 + thresh, y_center - height / 2 + thresh]
        xy_max = [x_center + width / 2 - thresh, y_center + height / 2 - thresh]
        data_np = np.random.uniform(low=xy_min, high=xy_max, size=(n, 2))
        data = tuple(map(tuple, data_np))

        return list(data)


class PoissonDisc:
    def __init__(self, width=50, height=50, r=1, k=30):
        self.width, self.height = width, height
        self.r = r
        self.k = k
        self.samples_t = []

        # Cell side length
        self.a = r / np.sqrt(2)
        # Number of cells in the x- and y-directions of the grid
        self.nx, self.ny = int(width / self.a) + 1, int(height / self.a) + 1

        self.reset()

    def reset(self):
        """Reset the cells dictionary."""

        # A list of coordinates in the grid of cells
        coords_list = [(ix, iy) for ix in range(self.nx)
                       for iy in range(self.ny)]
        # Initilalize the dictionary of cells: each key is a cell's coordinates
        # the corresponding value is the index of that cell's point's
        # coordinates in the samples list (or None if the cell is empty).
        self.cells = {coords: None for coords in coords_list}

    def get_cell_coords(self, pt):
        """Get the coordinates of the cell that pt = (x,y) falls in."""

        return int(pt[0] // self.a), int(pt[1] // self.a)

    def get_neighbours(self, coords):
        """Return the indexes of points in cells neighbouring cell at coords.
        For the cell at coords = (x,y), return the indexes of points in the
        cells with neighbouring coordinates illustrated below: ie those cells
        that could contain points closer than r.

                                     ooo
                                    ooooo
                                    ooXoo
                                    ooooo
                                     ooo

        """

        dxdy = [(-1, -2), (0, -2), (1, -2), (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
                (-2, 0), (-1, 0), (1, 0), (2, 0), (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1),
                (-1, 2), (0, 2), (1, 2), (0, 0)]
        neighbours = []
        for dx, dy in dxdy:
            neighbour_coords = coords[0] + dx, coords[1] + dy
            if not (0 <= neighbour_coords[0] < self.nx and
                    0 <= neighbour_coords[1] < self.ny):
                # We're off the grid: no neighbours here.
                continue
            neighbour_cell = self.cells[neighbour_coords]
            if neighbour_cell is not None:
                # This cell is occupied: store the index of the contained point
                neighbours.append(neighbour_cell)
        return neighbours

    def point_valid(self, pt):
        """Is pt a valid point to emit as a sample?

        It must be no closer than r from any other point: check the cells in
        its immediate neighbourhood.

        """

        cell_coords = self.get_cell_coords(pt)
        for idx in self.get_neighbours(cell_coords):
            nearby_pt = self.samples[idx]
            # Squared distance between candidate point, pt, and this nearby_pt.
            distance2 = (nearby_pt[0] - pt[0]) ** 2 + (nearby_pt[1] - pt[1]) ** 2
            if distance2 < self.r ** 2:
                # The points are too close, so pt is not a candidate.
                return False
        # All points tested: if we're here, pt is valid
        return True

    def get_point(self, refpt):
        """Try to find a candidate point near refpt to emit in the sample.

        We draw up to k points from the annulus of inner radius r, outer radius
        2r around the reference point, refpt. If none of them are suitable
        (because they're too close to existing points in the sample), return
        False. Otherwise, return the pt.

        """

        i = 0
        while i < self.k:
            rho, theta = (np.random.uniform(self.r, 2 * self.r),
                          np.random.uniform(0, 2 * np.pi))
            pt = refpt[0] + rho * np.cos(theta), refpt[1] + rho * np.sin(theta)
            if not (0 <= pt[0] < self.width and 0 <= pt[1] < self.height):
                # This point falls outside the domain, so try again.
                continue
            if self.point_valid(pt):
                return pt
            i += 1
        # We failed to find a suitable point in the vicinity of refpt.
        return False

    def sample(self, center, thresh):
        """Poisson disc random sampling in 2D.

        Draw random samples on the domain width x height such that no two
        samples are closer than r apart. The parameter k determines the
        maximum number of candidate points to be chosen around each reference
        point before removing it from the "active" list.

        """
        thresh = thresh
        width = self.width
        height = self.height

        current_center = (self.width / 2, self.height / 2)
        dx = center[0] - current_center[0]
        dy = center[1] - current_center[1]

        # Pick a random point to start with.
        pt = (np.random.uniform(0, self.width),
              np.random.uniform(0, self.height))
        self.samples = [pt]
        # Our first sample is indexed at 0 in the samples list...
        self.cells[self.get_cell_coords(pt)] = 0
        # and it is active, in the sense that we're going to look for more
        # points in its neighbourhood.
        active = [0]

        # As long as there are points in the active list, keep looking for
        # samples.
        while active:
            # choose a random "reference" point from the active list.
            idx = np.random.choice(active)
            refpt = self.samples[idx]
            # Try to pick a new point relative to the reference point.
            pt = self.get_point(refpt)
            if pt:
                # Point pt is valid: add it to samples list and mark as active
                pt_t = (pt[0] + dx, pt[1] + dy)  # Transform point to the new center
                if pt[0] > thresh and pt[1] > thresh and pt[0] < width - thresh and pt[1] < height - thresh:
                    self.samples_t.append(pt_t)
                self.samples.append(pt)
                nsamples = len(self.samples) - 1
                active.append(nsamples)
                self.cells[self.get_cell_coords(pt)] = nsamples
            else:
                # We had to give up looking for valid points near refpt, so
                # remove it from the list of "active" points.
                active.remove(idx)

        return self.samples_t


class Aniso:

    def sample(self, width, height, spacing, center):

        points = []
        x_center = center[0]
        y_center = center[1]

        dx = x_center - width / 2
        dy = y_center - height / 2

        n_x = int(width / spacing)
        n_y = int(height / spacing)
        width_margin = (width - spacing * n_x) / 2
        height_margin = (height - spacing * n_y) / 2

        for i in range(n_x):
            for j in range(n_y):
                point = (dx + width_margin + i * spacing, dy + height_margin * j)
                points.append(point)

        return points


class R2:
    # Returns a pair of deterministic pseudo-random numbers
    # based on seed i=0,1,2,...
    def getU(self, i):
        useRadial = True  # user-set parameter

        # Returns the fractional part of (1+1/x)^y
        def fractionalPowerX(x, y):
            n = x * y
            a = np.zeros(n).astype(int)
            s = np.zeros(n).astype(int)
            a[0] = 1
            for j in range(y):
                c = np.zeros(n).astype(int)
                s[0] = a[0]
                for i in range(n - 1):
                    z = a[i + 1] + a[i] + c[i]
                    s[i + 1] = z % x
                    c[i + 1] = z / x  # integer division!
                a = np.copy(s)
            f = 0;
            for i in range(y):
                f += a[i] * pow(x, i - y)
            return f

        #
        u = np.zeros(2)
        v = np.zeros(2)
        v = [fractionalPowerX(2, i + 1), fractionalPowerX(3, i + 1)]
        if useRadial:
            u = [pow(v[0], 0.5) * np.cos(2 * np.pi * v[1]), pow(v[0], 0.5) * np.sin(2 * np.pi * v[1])]
        else:
            u = [v[0], v[1]]
        return u

    # Returns the i-th term of the canonical R2 sequence
    # for i = 0,1,2,...
    def r2(self, i):
        g = 1.324717957244746  # special math constant
        a = [1.0 / g, 1.0 / (g * g)]
        return [a[0] * (i + 1) % 1, a[1] * (1 + i) % 1]

    # Returns the i-th term of the jittered R2 (infinite) sequence.
    # for i = 0,1,2,...
    def jitteredR2(self, i):
        lambd = 1.0  # user-set parameter
        useRandomJitter = False  # user parameter option
        delta = 0.76  # empirically optimized parameter
        i0 = 0.300  # special math constant
        p = np.zeros(2)
        u = np.zeros(2)
        p = self.r2(i)
        if useRandomJitter:
            u = np.random.random(2)
        else:
            u = self.getU(i)
        k = lambd * delta * pow(np.pi, 0.5) / (
                4 * pow(i + i0, 0.5))  # only this line needs to be modified for point sequences
        j = [k * x for x in u]  # multiply array x by scalar k
        pts = [sum(x) for x in zip(p, j)]  # element-wise addition of arrays p and j
        return [s % 1 for s in pts]  # element-wise %1 for s


class func_generator(bpy.types.Operator):
    bl_idname = "gen.func"
    bl_label = "Generate"
    position = (0, 0, 0)

    def execute(self, context):

        # Hard coded location of generated texture
        x, y, z = self.position

        # Reading the input from UI Panel
        scene_data = bpy.data.scenes["Scene"]
        
        base_size = scene_data.base_size
        base_diameter = scene_data.base_diameter
        
        spacing = scene_data.enterelement_space
        cone_cap_diameter = scene_data.cone_cap_diameter
        cone_base_diameter = scene_data.cone_base_diameter
        cone_h = scene_data.cone_height
        
        distribution = scene_data.dist_enum
        plate_shape = scene_data.plate_shape_enum
        
        num_particles = scene_data.num_particles
        
        bool_text = scene_data.bool_text
        font_text = scene_data.font_text
        depth_text = scene_data.depth_text
        eng_text = scene_data.text_input
        
        join_all = scene_data.join
        export_stl = scene_data.stl
        
        if plate_shape == '1':
            width = base_size[0]
            height = base_size[1]
            depth = base_size[2]
            
        if plate_shape == '2':
            width = base_diameter[0]
            height = base_diameter[0]
            depth = base_diameter[1]
        
        points_list = []
        text = ""

        print(width, height, depth, spacing, cone_cap_diameter, cone_base_diameter, cone_h)

        if distribution == '1':
            pd = PoissonDisc(width, height, spacing, 30)
            print(pd)
            center = (x, y)
            pd.sample(center, thresh=.25)
            points_list = pd.samples_t
            text = f" {round(spacing, 2)}, {round(cone_cap_diameter, 2)}"
            print("# of Poisson samples: ", len(pd.samples_t))  # poisson samples


        elif distribution == '2':
            points_list = sampler.Random.random_sampler(num_particles, width, height, 1, (x, y))  # random samples
            text = f"{num_particles}, {round(cone_cap_diameter, 2)}"
            print("# of Random samples: ", len(points_list))

        elif distribution == '3':
            points_list = sampler.Aniso.sample(width, height, spacing, (x, y))  # aniso smapler
            text = f"{spacing}"
            print("# of Aniso samples: ", len(points_list))  # Aniso samples

        # generating texture
        t0 = time.time()

        # Modify engrave text by user inputs
        if bool_text:
            text = eng_text

        if plate_shape == '1':
            Generator(base_shape = "", base_width=width, base_height=height, base_depth=depth,
                      cone_cap_diameter=cone_cap_diameter, cone_base_diameter=cone_base_diameter, cone_height=cone_h, points=points_list,
                      text=text, text_font=font_text, text_depth=depth_text, join=join_all, location=(x, y, z)
                      )
        if plate_shape == '2':
            Generator(base_shape = "circular", base_width=width, base_height=height, base_depth=depth,
                      cone_cap_diameter=cone_cap_diameter, cone_base_diameter=cone_base_diameter, cone_height=cone_h,
                      points=points_list, text=text, text_font=font_text, text_depth=depth_text, join=join_all, location=(x, y, z)
                      )
                      
        if export_stl:
            export.export()
            

        t1 = time.time()
        total = t1 - t0

        self.report({'INFO'}, f"Generation process took {total} seconds!")
        return {'FINISHED'}


class panel(bpy.types.Panel):
    bl_idname = "PANEL_PT_panel"
    bl_label = "Texture Generator"
    bl_category = "ATG"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"

    scene = bpy.types.Scene
    float_vec = bpy.props.FloatVectorProperty
    float_ = bpy.props.FloatProperty
    int_ = bpy.props.IntProperty
    enum = bpy.props.EnumProperty
    boolean_ = bpy.props.BoolProperty
    string_ = bpy.props.StringProperty

    scene.base_size = float_vec(name="",
                                min=1, max=200,
                                default=(20.0, 20.0, 2),
                                soft_min=1, soft_max=200,
                                subtype="XYZ_LENGTH")
                                
    scene.base_diameter = float_vec(name="",
                                size = 2,
                                min=1, max=200,
                                default=(20.0, 1),
                                soft_min=1, soft_max=200,
                                subtype="XYZ_LENGTH")

    scene.enterelement_space = float_(name="Lambda",
                                      min=0, max=2, soft_min=.5, soft_max=2,
                                      default=0.75, subtype="DISTANCE")

    scene.num_particles = int_(name="Number of Textones:",
                               min=0, max=10000, soft_min=0, soft_max=10000,
                               default=10)

    scene.cone_cap_diameter = float_(name="Cap Diameter",
                                min=0, max=2, default=0.20,
                                subtype="DISTANCE")
                                
    scene.cone_base_diameter = float_(name="Base Diameter",
                                min=0, max=5, default=0.5,
                                subtype="DISTANCE")

    scene.cone_height = float_(name="Height",
                               min=.1, max=2, default=1.0,
                               subtype="DISTANCE")

    scene.dist_enum = enum(
        items=[("1", "Poisson Disk", ""),
               ("2", "Random", ""),
               ("3", "Aniso", ""),
               ("4", "Custom", ""),
               ],
        name="",
        description="Select the distribution of textones",
        default="1",
        update=None,
    )
    
    scene.plate_shape_enum = enum(
        items=[("1", "Rectangular", ""),
               ("2", "Circular", ""),
               ],
        name="",
        description="Select the shape of the plate",
        default="1",
        update=None,
    )

    scene.bool_text = boolean_(
        name="Engrave Desired Text", default=False,
        options={'ANIMATABLE'}
    )
    
    scene.font_text = float_(name="Font",
                               min=0, max=10, default=3, step=50)
                               
    scene.depth_text = float_(name="Depth",
                               min=0, max=10, default=4, step=50)

    scene.text_input = string_(
        name="",
        description=":",
        default="",
        maxlen=25,
    )

    scene.join = boolean_(
        name="Join All", default=True,
        options={'ANIMATABLE'}
    )
    
    scene.stl = boolean_(
        name="Export .STL", default=True,
        options={'ANIMATABLE'}
    )
    
    
    print(scene.enterelement_space)

    def draw(self, context):
        
        layout = self.layout
        row = layout.row()
        col = layout.column()
        
        
        
        scene = context.scene
        scene_data = bpy.data.scenes["Scene"]
        
        box = layout.box()
        box.label(text="Base :")
        
        box.prop(scene, "plate_shape_enum")
        if scene_data.plate_shape_enum == '1':
            box.prop(scene, "base_size")
        if scene_data.plate_shape_enum == '2':
            box.prop(scene, "base_diameter")

        
        box = layout.box()
        box.label(text="Distribution:")
        box.prop(scene, "dist_enum")
    
        if scene_data.dist_enum == '1' or scene_data.dist_enum == '3':
            box.prop(scene, "enterelement_space")
        if scene_data.dist_enum == '2':
            box.prop(scene, "num_particles")
         
        box = layout.box()
        box.label(text="Cone:")
        box.prop(scene, "cone_cap_diameter")
        box.prop(scene, "cone_base_diameter")
        box.prop(scene, "cone_height")
        
        
        box = layout.box()
        box.label(text="Engrave Text:")
        row = box.row()
        row.prop(scene, 'font_text')
        row = box.row()
        row.prop(scene, 'depth_text')
        box.prop(scene, "bool_text")

        if scene_data.bool_text == True:
            row = layout.row()
            row.prop(scene, "text_input")

        box = layout.box()
        row = box.row()
        row.prop(scene, "join")
        row.prop(scene, "stl")

        layout.operator(func_generator.bl_idname)


classes = (func_generator, panel)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()






