# Author: weiwei
import math

import numpy as np
from scipy.spatial.transform import Rotation as R


def pose_inv(rt):
    rm = rt[0:3, 0:3]
    tvec = rt[0:3, 3]
    inv_rt = np.eye(4)
    inv_rt[0:3, 0:3] = rm.T
    inv_rt[0:3, 3] = np.matmul(-rm.T, tvec)
    return inv_rt[0:rt.shape[0], 0:rt.shape[1]]


def rotation_matrix_mean(r_list, iteration_count=20, eps=1e-10):
    """
    https://www.zhihu.com/question/439497100

    :param r_list:
    :param iteration_count:
    :param eps:
    :return:
    """
    m = r_list[0]
    for i in range(iteration_count):
        w = np.mean([R.from_matrix(m.T.dot(x)).as_rotvec() for x in r_list], axis=0)
        m = m.dot(R.from_rotvec(w).as_matrix())
        if np.linalg.norm(w) < eps:  # early stop
            break
    return m


def rotation_matrix_mean_weight(r_list, r_weights, iteration_count=20, eps=1e-10):
    """
    https://www.zhihu.com/question/439497100

    :param r_list:
    :param iteration_count:
    :param eps:
    :return:
    """
    m = r_list[0]
    for i in range(iteration_count):
        w = np.sum([R.from_matrix(m.T.dot(x)).as_rotvec() * w for x, w in zip(r_list, r_weights)], axis=0) / np.sum(r_weights)
        m = m.dot(R.from_rotvec(w).as_matrix())
        if np.linalg.norm(w) < eps:  # early stop
            break
    return m


def avg_rotation(x1, x2, w1, w2):
    m = (w1 * R.from_matrix(x1).as_rotvec() + w2 * R.from_matrix(x2).as_rotvec()) / (w1 + w2)
    return R.from_rotvec(m).as_matrix()


if __name__ == '__main__':
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    from lib.utils import visualize_utils
    from lib.utils.base_utils import project
    import matplotlib.pyplot as plt

    box = np.array([[0, 0, 0],
           [1, 0, 0],
           [0, 1, 0],
           [1, 1, 0],
           [0, 0, 1],
           [1, 0, 1],
           [0, 1, 1],
           [1, 1, 1]])

    K = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]])
    r1 = R.from_euler('z', 0, degrees=True).as_matrix()
    r2 = R.from_euler('z', 90, degrees=True).as_matrix()
    t = [[50], [50], [50]]
    rt1 = np.column_stack([r1, t])
    rt2 = np.column_stack([r2, t])

    alpha_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    alpha_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
    fig = visualize_utils.create_figure(ion=False)
    ax = fig.add_subplot(1, 1, 1)
    label = ['1', '2', '3', '4', '5', '6', '7', '8']
    color = ['g', 'b', 'r', 'k', 'y', 'gray']
    for c, alpha in zip(color, alpha_list):
        r3 = rotation_matrix_mean_weight([r1, r2], [alpha, 1-alpha])
        rt3 = np.column_stack([r3, t])
        # visualize_utils.draw_keypoints_and_labels(ax, project(box, K, rt1), '.g', label, 'g')
        # visualize_utils.draw_keypoints_and_labels(ax, project(box, K, rt2), '.y', label, 'y')
        visualize_utils.draw_keypoints_and_labels(ax, project(box, K, rt3), '.', label, c)
        # visualize_utils.draw_bbox3d_projection(ax, project(box, K, rt1), 'g')
        # visualize_utils.draw_bbox3d_projection(ax, project(box, K, rt2), 'r')
        visualize_utils.draw_bbox3d_projection(ax, project(box, K, rt3), c)
        # plt.show()
        # visualize_utils.wait_and_clf()
        # plt.waitforbuttonpress()
        # plt.show()
    plt.show()
