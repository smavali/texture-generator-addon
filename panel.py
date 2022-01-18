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
import main
import sampler
from sampler import *
import importlib

importlib.reload(main)
importlib.reload(sampler)
from main import *
import time


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
        join_all = scene_data.join
        eng_text = scene_data.text_input
        bool_text = scene_data.bool_text
        
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
        if bool_text == True:
            text = eng_text

        if plate_shape == '1':
            Generator(base_shape = "", base_width=width, base_height=height, base_depth=depth,
                      cone_cap_diameter=cone_cap_diameter, cone_base_diameter=cone_base_diameter, cone_height=cone_h,
                      cone_shape="", points=points_list,
                      text=text, join=join_all, location=(x, y, z)
                      )
        if plate_shape == '2':
            Generator(base_shape = "circular", base_width=width, base_height=height, base_depth=depth,
                      cone_cap_diameter=cone_cap_diameter, cone_base_diameter=cone_base_diameter, cone_height=cone_h,
                      cone_shape="", points=points_list,
                      text=text, join=join_all, location=(x, y, z)
                      )

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
        box.prop(scene, "bool_text")

        if scene_data.bool_text == True:
            row = layout.row()
            row.prop(scene, "text_input")

        row = layout.row()
        row.prop(scene, "join")

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






