import bpy
import os
from pathlib import Path

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