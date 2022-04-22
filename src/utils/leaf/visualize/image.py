# Author: weiwei
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as patches, pyplot as plt
from matplotlib.patches import ConnectionPatch


def create_figure(figsize=(19.20, 10.80), ion=True):
    fig = plt.figure(figsize=figsize)
    if ion:
        plt.ion()
    return fig


def wait_and_clf(timeout=-1):
    plt.waitforbuttonpress(timeout)
    plt.clf()


def visualize_visible_points(depth, image, visible_mask, xy, figsize=(19.20, 10.80)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 2, 1)
    # ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image)
    ax.plot(xy[:, 0], xy[:, 1], '.r')
    ax.plot(xy[visible_mask, 0], xy[visible_mask, 1], '+b')
    ax1 = fig.add_subplot(1, 2, 2)
    ax1.imshow(depth)
    ax1.plot(xy[:, 0], xy[:, 1], '.r')
    ax1.plot(xy[visible_mask, 0], xy[visible_mask, 1], '+b')
    return fig


def draw_bbox3d_projection(axes, bbox3d_projection, edgecolor, fill=False, linewidth=1):
    p = axes.add_patch(patches.Polygon(xy=bbox3d_projection[[0, 1, 3, 2, 0, 4, 6, 2]], fill=fill, linewidth=linewidth, edgecolor=edgecolor))
    p = axes.add_patch(patches.Polygon(xy=bbox3d_projection[[5, 4, 6, 7, 5, 1, 3, 7]], fill=fill, linewidth=linewidth, edgecolor=edgecolor))
    return p


def draw_bbox3d_projection_v2(axes, bbox3d_projection, edgecolor, fill=False, linewidth=1):
    p = axes.add_patch(patches.Polygon(xy=bbox3d_projection[[0, 1, 3, 2, 0, 4, 6, 2]], fill=True, linewidth=linewidth, edgecolor=edgecolor, alpha=0.2))
    p = axes.add_patch(patches.Polygon(xy=bbox3d_projection[[5, 4, 6, 7, 5, 1, 3, 7]], fill=fill, linewidth=linewidth, edgecolor=edgecolor))
    return p


def draw_keypoints(axes, keypoints_xy, fmt):
    p = axes.plot(keypoints_xy[:, 0], keypoints_xy[:, 1], fmt)
    return p


def draw_text(axes, positions, text_list, color, alpha=0.5):
    for pos, text in zip(positions, text_list):
        axes.text(pos[0]+1, pos[1]-1, text, bbox={'facecolor': color, 'alpha': alpha, 'pad': 1})


def draw_keypoints_and_labels(axes, xy, xy_fmt, labels, l_fmt):
    p = draw_keypoints(axes, xy, xy_fmt)
    if labels is None:
        labels = list(map(str, range(1, len(xy)+1)))
    draw_text(axes, xy, labels, l_fmt)
    return p


def draw_vector(axes, xy, uv, *args, **kw):
    q = axes.quiver(xy[:, 0], xy[:, 1], uv[:, 0], uv[:, 1], *args, **kw, angles='xy', scale=1, scale_units='xy', width=0.001)
    return q


def draw_bbox2d(axes, bbox2d, text, edgecolor, fill=False, linewidth=1, bbox_fmt='xyxy'):
    """
    Draw 2D bounding box
    :param axes:
    :param bbox2d: four value for bounding box
    :param edgecolor:
    :param fill:
    :param linewidth:
    :param bbox_fmt: the format of bounding box,
                        - 'xyxy' means two points,
                        - 'ltrb' means left, top, right, botton
                        - 'xywh' means a point, width and height
    """
    if bbox_fmt == 'xyxy' or bbox2d == 'ltrb':
        left = bbox2d[0]
        top = bbox2d[1]
        right = bbox2d[2]
        bottom = bbox2d[3]
    elif bbox_fmt == 'xywh':
        left = bbox2d[0]
        top = bbox2d[1]
        right = bbox2d[2] + left
        bottom = bbox2d[3] + right
    else:
        raise ValueError('the bbox_fmt is wrong!')
    p = axes.add_patch(
        patches.Polygon(xy=[[left, top], [left, bottom], [right, bottom], [right, top], [left, top]], fill=fill,
                        linewidth=linewidth, edgecolor=edgecolor))
    axes.text(left+1, top-1, text, bbox={'facecolor': edgecolor, 'alpha': 0.5, 'pad': 1})
    return p


def draw_bboxes2d(ax, bboxes2d, text_list, edgecolor_list, fill=False, linewidth=1, bbox_fmt='xyxy'):
    n = len(bboxes2d)
    p_list = []
    for i in range(n):
        p = draw_bbox2d(ax, bboxes2d[i], text_list[i], edgecolor_list[i], fill, linewidth, bbox_fmt)
        p_list.append(p)
    return p_list


def visualize_fields_errors(xy_fields_error, fig=None, hist_size=150):
    """
    Visualize fields and plot histogram for x and y
    :param xy_fields_error: np.array, shape is (h, w)
    :param fig:
    :param hist_size: size of histogram on figure
    :return:

    >>> xy_fields = np.random.randn(480, 640)
    >>> _ = visualize_fields_errors(xy_fields)
    >>> plt.show()
    """
    rows, cols = xy_fields_error.shape[:2]
    x_hist = np.sum(xy_fields_error, axis=0)
    y_hist = np.sum(xy_fields_error, axis=1)

    x_span = y_span = hist_size

    if fig is None:
        fig = plt.figure(figsize=((cols+y_span)/100, (rows+x_span)/100))
    gs = fig.add_gridspec(2, 2, width_ratios=(cols, y_span), height_ratios=(x_span, rows),
                          left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.01, hspace=0.01)
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    # no labels
    ax_histx.tick_params(axis='x', labelbottom=False)
    ax_histy.tick_params(axis='y', labelleft=False)

    # plot
    ax_histy.barh(np.arange(rows), y_hist, align='center')
    ax_histx.bar(np.arange(cols), x_hist, align='center')
    ax.imshow(xy_fields_error)
    return fig


def visualize_images(images, titles, fig):
    k = len(images)
    m = math.ceil(math.sqrt(k))
    n = math.ceil(k / m)
    ax_list = []
    for i in range(k):
        ax = fig.add_subplot(n, m, i+1)
        ax.imshow(images[i])
        ax.set_title(titles[i])
        ax_list.append(ax)
    return ax_list


def visualize_points_pairs(points_pairs_list, axes_list, arrowstyle='-'):
    assert len(points_pairs_list) == len(axes_list)
    for ((p0, p1), (ax1, ax2)) in zip(points_pairs_list, axes_list):
        con_list = list(map(lambda x: ConnectionPatch(x[0], x[1], 'data', 'data', arrowstyle=arrowstyle, axesA=ax2, axesB=ax1,
                                                      color=np.random.uniform(size=(3,)), shrinkA=0, shrinkB=0), zip(p1, p0)))
        list(map(lambda x: ax2.add_artist(x), con_list))


def draw_mask(mask, ax, color, alpha=0.2):
    """

    :param mask:
    :param ax:
    :param color:
    :param alpha:
    :return:

    >>> mask = np.zeros((480, 640))
    >>> mask[100:200, 100: 300] = 1
    >>> _, ax = plt.subplots(1)
    >>> _ = draw_mask(mask, ax, [255, 0, 0], alpha=0.2)
    >>> plt.show()
    """
    color_mask = mask[:, :, None] * np.array(color)[None, None, :]
    im = ax.imshow(color_mask.astype(np.uint8), alpha=alpha)
    return im


if __name__ == '__main__':
    import doctest
    doctest.testmod()
