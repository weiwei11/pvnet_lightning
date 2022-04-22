# Author: weiwei
import numpy as np

# from lib.utils.renderer import opengl_utils


def equal_round(a: np.array, b: np.array, delta: float) -> np.array:
    """
    :param a: array
    :param b: array
    :param delta: tolerance used.
    :return: bool array
    """
    return np.abs(a - b) < delta


def estimate_visible_points_mask(depth, xy, z, delta=20):
    rows = depth.shape[0]
    cols = depth.shape[1]

    # point location
    row_low = np.int32(np.floor(xy[:, 1]))
    col_low = np.int32(np.floor(xy[:, 0]))
    row_high = np.int32(np.ceil(xy[:, 1]))
    col_high = np.int32(np.ceil(xy[:, 0]))

    # points in image
    row_low_mask = (row_low >= 0) & (row_low < rows)
    col_low_mask = (col_low >= 0) & (col_low < cols)
    row_high_mask = (row_high >= 0) & (row_high < rows)
    col_high_mask = (col_high >= 0) & (col_high < cols)

    visible_mask = np.zeros((xy.shape[0], ), dtype=np.bool)
    idx_mask = row_low_mask & col_low_mask
    visible_mask[idx_mask] = visible_mask[idx_mask] | equal_round(z[idx_mask], depth[row_low[idx_mask], col_low[idx_mask]], delta)
    idx_mask = row_low_mask & col_high_mask
    visible_mask[idx_mask] = visible_mask[idx_mask] | equal_round(z[idx_mask], depth[row_low[idx_mask], col_high[idx_mask]], delta)
    idx_mask = row_high_mask & col_low_mask
    visible_mask[idx_mask] = visible_mask[idx_mask] | equal_round(z[idx_mask], depth[row_high[idx_mask], col_low[idx_mask]], delta)
    idx_mask = row_high_mask & col_high_mask
    visible_mask[idx_mask] = visible_mask[idx_mask] | equal_round(z[idx_mask], depth[row_high[idx_mask], col_high[idx_mask]], delta)

    return visible_mask


def estimate_visible_mask(im_size, opengl, K, pose, points_xyz, delta=20, return_more=False):
    depth = opengl.render(im_size, 100, 10000, K, pose[:, :3], pose[:, 3:])

    xyz = np.dot(points_xyz, pose[:, :3].T) + pose[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    z = xyz[:, 2]

    visible_mask = estimate_visible_points_mask(depth, xy, z, delta)
    if return_more:
        return visible_mask, xyz, depth
    else:
        return visible_mask


def estimate_visible_mask_for_all(im_size, model, K_list, pose_list, points_xyz, delta=20, scale=1000.0, return_xyz=False, return_depth=False):
    assert len(K_list) == len(pose_list)

    # scale = 1000.0
    model['pts'] = model['pts'] * scale
    # opengl = opengl_utils.NormalRender(model, im_size)
    opengl = opengl_utils.DepthRender(model, im_size)

    result_list = []
    for (K, pose) in zip(K_list, pose_list):
        pose[:, 3:] = pose[:, 3:] * scale
        ret = estimate_visible_mask(im_size, opengl, K, pose, points_xyz * scale, delta, return_xyz or return_depth)
        if return_xyz and return_depth:
            ret = (ret[0], ret[1] / scale, ret[2])
        elif return_xyz and not return_xyz:
            ret = (ret[0], ret[1] / scale)
        elif not return_xyz and return_depth:
            ret = (ret[0], ret[2])
        result_list.append(ret)

    model['pts'] = model['pts'] / scale  # recover data

    return result_list


def estimate_inside_mask(uv, mask, pixel_error):
    h, w = mask.shape[:2]
    inside_mask = np.zeros((uv.shape[0],), dtype=np.bool)
    # pixel inside mask with error
    for i in range(-pixel_error, pixel_error + 1):
        for j in range(-pixel_error, pixel_error + 1):
            row, col = uv[:, 1] + i, uv[:, 0] + j
            idx_mask = (row >= 0) & (col >= 0) & (row < h) & (col < w)
            inside_mask[idx_mask] = inside_mask[idx_mask] | (mask[row[idx_mask], col[idx_mask]] > 0)

    # pixel outside image
    row, col = uv[:, 1], uv[:, 0]
    idx_mask = (row < 0) | (col < 0) | (row >= h) | (col >= w)
    inside_mask[idx_mask] = False

    return inside_mask


def estimate_inside_mask_v2(uv, mask, pixel_error):
    h, w = mask.shape[:2]
    # pixel inside mask with error
    x, y = np.meshgrid(range(-pixel_error, pixel_error + 1), range(-pixel_error, pixel_error + 1))
    grid = np.column_stack([x.flatten(), y.flatten()])
    uv_all = uv[:, None, :] + grid[None, :, :]
    uv_all = np.reshape(uv_all, (-1, 2))
    row, col = uv_all[:, 1], uv_all[:, 0]

    idx_mask = (row >= 0) & (col >= 0) & (row < h) & (col < w)
    inside_mask = np.zeros((uv_all.shape[0],), dtype=np.bool)
    inside_mask[idx_mask] = mask[row[idx_mask], col[idx_mask]] > 0
    inside_mask = np.any(inside_mask.reshape((uv.shape[0], -1)), -1)

    # pixel outside image
    row, col = uv[:, 1], uv[:, 0]
    idx_mask = (row < 0) | (col < 0) | (row >= h) | (col >= w)
    inside_mask[idx_mask] = False

    return inside_mask


def points2d2points3d(points_xy, depth, K):
    if points_xy.shape[0] == 0:
        return None

    xyz = np.concatenate([np.round(points_xy), np.ones((points_xy.shape[0], 1))], axis=1)
    xyz = xyz * depth[np.int32(xyz[:, 1]), np.int32(xyz[:, 0]), None]
    xyz = xyz.dot(np.linalg.inv(K).T)
    return xyz


def transform_point3d(points_xyz, rt):
    return points_xyz @ rt[:3, :3].T + rt[:3, 3][None, :]


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy
