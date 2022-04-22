# Author: weiwei

import numpy as np

from .metric import BaseMetric, filter_parameters, Compose
from .functional.sixd import projection_2d, add, cm_degree, add_error, add_auc, nearest_point_distance, angular_error, \
    translation_error
# from leaf.metrics.metric import BaseMetric, filter_parameters, Compose
# from leaf.metrics.functional.sixd import projection_2d, add, cm_degree, add_error, add_auc, nearest_point_distance, angular_error, \
#     translation_error


class PoseCompose(Compose):
    def __init__(self, name='', model=None, symmetric=False, out_as_in=False, *metrics):
        super().__init__(name)
        self.model = model
        self.symmetric = symmetric
        self.out_as_in = out_as_in
        self.metrics = metrics

    def _compute_common_data(self, predict_pose, target_pose, K):
        model_pred = np.dot(self.model, predict_pose[:, :3].T) + predict_pose[:, 3]
        model_target = np.dot(self.model, target_pose[:, :3].T) + target_pose[:, 3]

        proj_pred = np.dot(model_pred, K.T)
        proj_pred = proj_pred[:, :2] / proj_pred[:, 2:]
        proj_target = np.dot(model_target, K.T)
        proj_target = proj_target[:, :2] / proj_target[:, 2:]

        # add error
        if self.symmetric:
            add_err = np.mean(nearest_point_distance(model_pred, model_target))
        else:
            add_err = np.mean(np.linalg.norm(model_pred - model_target, axis=-1))

        # projection error
        proj_err = np.mean(np.linalg.norm(proj_pred - proj_target, axis=-1))

        # angular error
        angular_err = angular_error(predict_pose, target_pose)

        # translation error
        translation_err = translation_error(predict_pose, target_pose)

        return {'add_err': add_err, 'projection_err': proj_err, 'angular_err': angular_err, 'translation_err': translation_err}

    def __call__(self, data, data_mode='mix'):
        result_dict = {}
        if data_mode == 'mix':
            res = self._compute_common_data(data['predict_pose'], data['target_pose'], data['K'])
            data.update(res)
            for m in self.metrics:
                if not self.out_as_in:
                    res = filter_parameters(m, data)
                else:
                    res = filter_parameters(m, {**data, **result_dict})
                result_dict.update({m.name: res})
        elif data_mode == 'seq':
            res = self._compute_common_data(data['predict_pose'], data['target_pose'], data['K'])
            data.update(res)
            for m, d in zip(self.metrics, data):
                if isinstance(d, dict):
                    res = m(**d)
                else:
                    res = m(*d)
                result_dict.update({m.name: res})
        else:
            raise ValueError('data_mode must be mix or seq')
        return result_dict


class Projection2d(BaseMetric):
    """
    2D projection

    :param name: name of the metric
    :param model: shape (N, 3), 3D points cloud of object
    :param threshold: default is 5 pixel

    >>> import numpy as np
    >>> pose_pred = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_target = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> K = np.array([[320, 0, 320], [0, 320, 240], [0, 0, 1]])
    >>> model_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> proj_metric = Projection2d(model=model_xyz, threshold=5)
    >>> proj_metric(pose_pred, pose_target, K)
    True
    >>> proj_metric.summarize()
    1.0
    """
    def __init__(self, name='Projection2d', model=None, threshold=5):

        super().__init__(name)
        self.model = model
        self.threshold = threshold

    def __call__(self, predict_pose, target_pose, K, projection_err=None):
        if projection_err is None:
            result = projection_2d(predict_pose, target_pose, K, self.model, self.threshold)
        else:
            result = projection_err < self.threshold

        self.result_list.append(result)

        return result


class ADD(BaseMetric):
    """
    ADD

    :param name: name of the metric
    :param model: shape (N, 3), 3D points cloud of object
    :param symmetric: whether the object is symmetric or not
    :param threshold: distance threshold, 'threshold' and 'model_diameter percentage' is not compatible
    :param diameter: the diameter of object
    :param percentage: percentage of model diameter

    >>> import numpy as np
    >>> pose_pred = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_target = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> model_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> model_diameter = np.sqrt(3)
    >>> add_metric = ADD(model=model_xyz, symmetric=False, threshold=None, diameter=model_diameter, percentage=0.1)
    >>> add_metric(pose_pred, pose_target)
    True
    >>> add_metric.summarize()
    1.0
    """
    def __init__(self, name='ADD', model=None, symmetric=False, threshold=None, diameter=None, percentage=0.1):
        super().__init__(name)
        self.model = model
        self.symmetric = symmetric
        self.threshold = threshold if threshold is not None else diameter * percentage

    def __call__(self, predict_pose, target_pose, add_err=None):
        if add_err is None:
            result = add(predict_pose, target_pose, self.model, self.symmetric, self.threshold)
        else:
            result = add_err < self.threshold
        self.result_list.append(result)

        return result


class MeanRotationError(BaseMetric):
    """
    Mean Rotation Error

    :param name: name of the metric

    >>> import numpy as np
    >>> pose_pred = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_target = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_pred1 = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 2]])
    >>> pose_target1 = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> re_metric = MeanRotationError()
    >>> re_metric(pose_pred, pose_target)
    0.0
    >>> re_metric(pose_pred1, pose_target1)
    0.0
    >>> re_metric.summarize()
    0.0
    """
    def __init__(self, name='Re'):
        super().__init__(name)

    def __call__(self, predict_pose, target_pose, angular_err=None):
        if angular_err is None:
            result = angular_error(predict_pose, target_pose)
        else:
            result = angular_err
        self.result_list.append(result)

        return result


class MeanTranslationError(BaseMetric):
    """
    Mean Translation Error

    :param name: name of the metric
    :param unit_scale: scale for meter

    >>> import numpy as np
    >>> pose_pred = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_target = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_pred1 = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 2]])
    >>> pose_target1 = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> te_metric = MeanTranslationError(unit_scale=1.0)
    >>> te_metric(pose_pred, pose_target)
    0.0
    >>> te_metric(pose_pred1, pose_target1)
    1.0
    >>> te_metric.summarize()
    0.5
    """
    def __init__(self, name='Te', unit_scale=1.0):
        super().__init__(name)
        self.unit_scale = unit_scale

    def __call__(self, predict_pose, target_pose, translation_err=None):
        if translation_err is None:
            result = translation_error(predict_pose, target_pose) * self.unit_scale
        else:
            result = translation_err
        self.result_list.append(result)

        return result


class Cmd(BaseMetric):
    """
    Degree and cm

    :param name: name of the metric
    :param cm_threshold: unit is centimeter
    :param degree_threshold:
    :param unit_scale: scale for meter

    >>> import numpy as np
    >>> pose_pred = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_target = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_pred1 = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 2]])
    >>> pose_target1 = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> cmd_metric = Cmd(cm_threshold=5, degree_threshold=5, unit_scale=1.0)
    >>> cmd_metric(pose_pred, pose_target)
    True
    >>> cmd_metric(pose_pred1, pose_target1)
    False
    >>> cmd_metric.summarize()
    0.5
    """
    def __init__(self, name='Cmd', cm_threshold=5, degree_threshold=5, unit_scale=1.0):
        super().__init__(name)
        self.cm_threshold = cm_threshold
        self.degree_threshold = degree_threshold
        self.unit_scale = unit_scale

    def __call__(self, predict_pose, target_pose, angular_err=None, translation_err=None):
        if angular_err is None or translation_err is None:
            pred = predict_pose.copy()
            target = target_pose.copy()
            pred[:3, 3] = pred[:3, 3] * self.unit_scale
            target[:3, 3] = target[:3, 3] * self.unit_scale
            result = cm_degree(pred, target, self.cm_threshold, self.degree_threshold)
        else:
            result = translation_err * self.unit_scale < 0.01 * self.cm_threshold and angular_err < self.degree_threshold
        self.result_list.append(result)

        return result


class ADDAUC(BaseMetric):
    """
    ADD AUC

    :param name: name of the metric
    :param model: shape (N, 3), 3D points cloud of object
    :param max_threshold: max error threshold, so threshold is [0, max]
    :param unit_scale: scale for meter unit
    :param symmetric: whether the object is symmetric or not

    >>> import numpy as np
    >>> model_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> pose_pred = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_target = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_pred1 = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 2]])
    >>> pose_target1 = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> pose_pred2 = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 3]])
    >>> pose_target2 = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])
    >>> add_auc_metric = ADDAUC(model=model_xyz, max_threshold=0.1, unit_scale=1.0, symmetric=False)
    >>> add_auc_metric(pose_pred, pose_target)
    >>> add_auc_metric(pose_pred1, pose_target1)
    >>> add_auc_metric(pose_pred2, pose_target2)
    >>> add_auc_metric.summarize()
    0.3333333333333333
    """
    def __init__(self, name='ADD_AUC', model=None, max_threshold=None, unit_scale=1.0, symmetric=False):
        super().__init__(name)
        self.model = model
        self.max_threshold = max_threshold
        self.unit_scale = unit_scale
        self.symmetric = symmetric

    def __call__(self, predict_pose, target_pose, add_err=None):
        if add_err is None:
            result = add_error(predict_pose, target_pose, self.model, self.symmetric)
        else:
            result = add_err
        self.result_list.append(result)

    def summarize(self):
        result = add_auc(self.result_list, self.max_threshold, self.unit_scale)
        return result


if __name__ == '__main__':
    import doctest
    doctest.testmod()
