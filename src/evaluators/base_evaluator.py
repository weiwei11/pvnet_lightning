# Created by ww at 2021/5/4
from src.utils.wind.file_util import makedirs


class BaseEvaluator:
    def __init__(self, *args, **kwargs):
        pass

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError()

    def summarize(self):
        return {}
