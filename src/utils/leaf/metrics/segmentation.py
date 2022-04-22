# Author: weiwei

from .metric import BaseMetric
from .functional.segmentation import mask_iou


class AveragePrecision(BaseMetric):
    def __init__(self, name='AP', threshold=0.7):
        super().__init__(name)
        self.threshold = threshold

    def __call__(self, predict_mask, target_mask):
        iou = mask_iou(predict_mask, target_mask)
        self.result_list.append(iou > self.threshold)

        return iou > self.threshold
