# Author: weiwei
import open3d as o3d
import numpy as np


def visualize_kpt_3d(kpt_3d, size=5):
    kpt3d_list = []
    for i, cur_kpt in enumerate(kpt_3d):
        cubic = o3d.geometry.TriangleMesh.create_box(size, size, size)
        cubic.vertices = o3d.utility.Vector3dVector(np.asarray(cubic.vertices) + cur_kpt[None, :])
        cubic.paint_uniform_color([1, 0, 0])
        kpt3d_list.append(cubic)

    o3d.visualization.draw_geometries(kpt3d_list)

