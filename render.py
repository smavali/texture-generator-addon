import bpy
import mathutils
from math import *


# scene = bpy.context.scene

# scene.render.image_settings.file_format = 'PNG'
# scene.render.filepath = "F:/image.png"
# bpy.ops.render.render(write_still = 1)


def rotate_and_render(output_dir, output_file_pattern_string='render%d.jpg', rotation_steps=32, rotation_angle=360.0,
                      subject=bpy.context.object):
    import os
    original_rotation = subject.rotation_euler
    for step in range(0, rotation_steps):
        subject.rotation_euler[2] = radians(step * (rotation_angle / rotation_steps))
        bpy.context.scene.render.filepath = os.path.join(output_dir, (output_file_pattern_string % step))
        bpy.ops.render.render(write_still=True)
    subject.rotation_euler = original_rotation


rotate_and_render('/Users/sina', 'render%d.jpg')
