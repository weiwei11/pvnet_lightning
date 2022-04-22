# Author: weiwei
import numpy as np

from src.evaluators.base_evaluator import BaseEvaluator
import torch

from lib.vendor.bop_toolkit.bop_toolkit_lib import inout

from src.utils.leaf.metrics import Compose
from src.utils.leaf.metrics.sixd import *
from src.utils.leaf.metrics.segmentation import AveragePrecision


class PoseEvaluator(BaseEvaluator):

    def __init__(self, datasource):
        super().__init__()

        obj_cls = datasource.obj_cls
        obj_sym = datasource.is_symmetric
        model = datasource.load_model()['pts']
        diameter = datasource.model_diameter
        model_scale = datasource.model_scale

        # metric
        self.pose_metric = PoseCompose('pose_metric', model, obj_sym, False,
                                       Projection2d('proj2d', model),
                                       # Projection2d('proj2d_5to640', model, 5 / 640 * max(self.height, self.width)),
                                       ADD('add', model, diameter=diameter, percentage=0.1,
                                           symmetric=obj_sym),
                                       ADD('add5', model, diameter=diameter, percentage=0.05,
                                           symmetric=obj_sym),
                                       ADD('add2', model, diameter=diameter, percentage=0.02,
                                           symmetric=obj_sym),
                                       Cmd('cmd5', 5, 5, model_scale),
                                       ADDAUC('add_auc', model, 0.1, unit_scale=model_scale, symmetric=obj_sym),
                                       )
        self.icp_metric = PoseCompose('icp_metric', model, obj_sym, False,
                                      Projection2d('icp_proj2d', model),
                                      ADD('icp_add', model, diameter=diameter, percentage=0.1, symmetric=obj_sym),
                                      Cmd('icp_cmd5', 5, 5),
                                      )
        self.mask_metric = Compose('mask_metric', False, AveragePrecision('ap', 0.7))


    def evaluate(self, result_dict):
        # img_id = int(result_dict['img_id'])
        # kpt_2d_pred = result_dict['kpt_2d_pred']

        # kpt_3d = result_dict['kpt_3d']
        K = np.array(result_dict['K'])

        pose_gt = np.array(result_dict['pose_gt'])
        pose_pred = result_dict['pose_pred']

        self.pose_metric({'predict_pose': pose_pred, 'target_pose': pose_gt, 'K': K})
        if 'mask_pred' in result_dict:
            self.mask_metric({'predict_mask': result_dict['mask_pred'], 'target_mask': result_dict['mask_gt']})

        if 'pose_pred_icp' in result_dict:
            pose_pred_icp = result_dict['pose_pred_icp']
            self.icp_metric({'predict_pose': pose_pred_icp, 'target_pose': pose_gt})

    def summarize(self):
        result_dict = {}
        result_dict.update(self.pose_metric.summarize())
        result_dict.update(self.mask_metric.summarize())
        for k, v in result_dict.items():
            print(f'{k}: {v}')

        for k, v in self.icp_metric.summarize().items():
            print(f'{k}: {v}')

        self.pose_metric.reset()
        self.icp_metric.reset()
        self.mask_metric.reset()
        return result_dict
