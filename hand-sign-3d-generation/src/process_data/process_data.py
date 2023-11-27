import json
import os

import bpy
import numpy as np
import json
import os

# Third-party imports
import bpy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Local application/library specific imports
from .utils import HAND_BONES, HAND_BONES_CONNECTIONS


class ProcessBVH:
    def __init__(self, bvh_path):
        self.bvh_path = bvh_path
        self.armature_name = None
        self.import_bvh()
        self.find_armature()
        bpy.context.scene.frame_end =2147483647
        self.max_frame_end = bpy.context.scene.frame_end
        print(self.max_frame_end)

    def import_bvh(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_anim.bvh(filepath=self.bvh_path)

    def set_frame(self, frame):
        bpy.context.scene.frame_set(frame)

    def find_armature(self):
        for obj in bpy.context.scene.objects:
            if obj.type == 'ARMATURE':
                self.armature_name = obj.name
                break

    def get_all_joint_names(self):
        joint_names = []
        armature = bpy.data.objects[self.armature_name]
        for bone in armature.pose.bones:
            joint_names.append(bone.name)
        return joint_names

    def get_bone_location(self, bone_name, frame):
        self.set_frame(frame)
        armature = bpy.data.objects[self.armature_name]
        return armature.pose.bones[bone_name].head

    def visualize_joint_locations(self, frame_to_visualize, save_path=None, use_plotly=False, debug=False):
        joint_names = self.get_all_joint_names()
        joint_locations = {}
        for joint_name in joint_names:
            print(joint_name)
            if joint_name not in HAND_BONES:
                continue
            location = self.get_bone_location(joint_name, frame_to_visualize)
            joint_locations[joint_name] = location
            print(
                f"{joint_name} Location at Frame {frame_to_visualize}: X={location.x}, Y={location.y}, Z={location.z}")

        if use_plotly:
            fig = go.Figure()
            for joint_name, location in joint_locations.items():
                fig.add_trace(go.Scatter3d(
                    x=[location.x], y=[location.y], z=[location.z],
                    mode='markers',
                    marker=dict(size=5),
                    name=joint_name
                ))
            for connection in HAND_BONES_CONNECTIONS:
                if connection[0] in joint_locations and connection[1] in joint_locations:
                    loc0 = joint_locations[connection[0]]
                    loc1 = joint_locations[connection[1]]
                    fig.add_trace(go.Scatter3d(
                        x=[loc0.x, loc1.x], y=[
                            loc0.y, loc1.y], z=[loc0.z, loc1.z],
                        mode='lines', line=dict(color='blue', width=2)
                    ))
            fig.update_layout(scene=dict(aspectmode="data"))
            fig.update_layout(
                title=f'Scatter Plot - Frame {frame_to_visualize}')
            if save_path:
                fig.write_html(save_path + '.html')
            fig.show()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for location in joint_locations.values():
                ax.scatter(location.x, location.y, location.z, 'o')
            for connection in HAND_BONES_CONNECTIONS:
                if connection[0] in joint_locations and connection[1] in joint_locations:
                    loc0 = joint_locations[connection[0]]
                    loc1 = joint_locations[connection[1]]
                    ax.plot([loc0.x, loc1.x], [loc0.y, loc1.y],
                            [loc0.z, loc1.z], 'blue')
            ax.set_title(f'3D Axes Plot - Frame {frame_to_visualize}')
            if debug:
                plt.imshow(fig)
                plt.show()
            return fig


if __name__ == '__main__':
    # Replace with the path to your BVH file
    BVH_PATH = "/Users/aleksandrsimonyan/Desktop/hand_sign_generation_project/datasets/BVH/3D_alphabet_11_15_2023_BVH.bvh"
    bpy.context.scene.frame_end = 2147483647  # Set the end frame to the desired value
    FRAME_TO_VISUALIZE = 3400  # Frame number to visualize
    # Replace with your desired save path and file name without extension
    SAVE_PATH = "/Users/aleksandrsimonyan/Desktop/hand_sign_generation_project/process"

    bvh_reader = ProcessBVH(BVH_PATH)

    # Visualize as a 3D scatter plot using Plotly and save as HTML
    bvh_reader.visualize_joint_locations(FRAME_TO_VISUALIZE, SAVE_PATH)

#    bvh_reader.visualize_joint_locations(FRAME_TO_VISUALIZE, SAVE_PATH, plot_type='axes3d')
