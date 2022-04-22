# Author: weiwei
from .image import *


def visualize_pose(ax, image, result, show_image=True):
    if show_image:
        ax.imshow(image)

    corner_2d_gt, corner_2d_pred = result['corner_2d_gt'], result['corner_2d_pred']
    kpt_2d_gt, kpt_2d_pred = result['kpt_2d_gt'], result['kpt_2d_pred']

    draw_bbox3d_projection_v2(ax, corner_2d_gt, edgecolor='g')
    draw_bbox3d_projection_v2(ax, corner_2d_pred, edgecolor='b')
    # fields_errors = np.sum((((output['vertex'][0] - batch['vertex'][0]).abs() * batch['mask'][0]).detach().cpu().numpy()), axis=0)
    # visualize_fields_errors(fields_errors)
    draw_keypoints(ax, kpt_2d_gt, '.g')
    draw_keypoints(ax, kpt_2d_pred, '.b')
    text_list = list(range(len(kpt_2d_gt)))
    draw_text(ax, kpt_2d_gt, text_list, 'g')
    draw_text(ax, kpt_2d_pred, text_list, 'b')


def visualize_mask(ax, image, mask, color, alpha=0.5, show_image=True):
    if show_image:
        color = np.array(color) * 255
        ax.imshow((image * alpha + (1 - alpha) * color * mask[:, :, None]) / 255)
    else:
        ax.imshow(mask)


def visualize_bbox2d(ax, image, result, show_image=True):
    if show_image:
        ax.imshow(image)

    bbox2d_gt, bbox2d_pred = result['bbox2d_gt'], result['bbox2d_pred']
    cls_name_gt, cls_name_pred = result['cls_name_gt'], result['cls_name_pred']
    conf_pred = result['conf_pred']

    draw_bboxes2d(ax, bbox2d_gt, [f'{cn}: gt' for cn in cls_name_gt],
                  ['g' for i in range(len(cls_name_gt))])
    if bbox2d_pred is not None:
        draw_bboxes2d(ax, bbox2d_pred,
                      [f'{cn}: {cconf}' for cn, cconf in zip(cls_name_pred, conf_pred)],
                      ['r' for i in range(len(cls_name_pred))])
