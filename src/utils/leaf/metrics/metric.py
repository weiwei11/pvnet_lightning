# Author: weiwei

import inspect
import numpy as np


def filter_parameters(func, params_dict):
    sig = inspect.signature(func)
    filter_key = set(sig.parameters.keys()) & set(params_dict.keys())
    params = {k: params_dict[k] for k in filter_key}
    return func(**params)


class BaseMetric:
    def __init__(self, name=''):
        self.result_list = []
        self.name = name

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

    def summarize(self):
        result_mean = np.mean(self.result_list)
        return result_mean

    def reset(self):
        self.result_list = []


class Compose(BaseMetric):
    def __init__(self, name='', out_as_in=False, *metrics):
        super().__init__(name)
        self.out_as_in = out_as_in
        self.metrics = metrics

    def __call__(self, data, data_mode='mix'):
        result_dict = {}
        if data_mode == 'mix':
            for m in self.metrics:
                if not self.out_as_in:
                    res = filter_parameters(m, data)
                else:
                    res = filter_parameters(m, {**data, **result_dict})
                result_dict.update({m.name: res})
        elif data_mode == 'seq':
            for m, d in zip(self.metrics, data):
                if isinstance(d, dict):
                    res = m(**d)
                else:
                    res = m(*d)
                result_dict.update({m.name: res})
        else:
            raise ValueError('data_mode must be mix or seq')
        return result_dict

    def summarize(self, data_mode='dict'):
        if data_mode == 'dict':
            s = {m.name: m.summarize() for m in self.metrics}
        elif data_mode == 'list':
            s = [[m.name, m.summarize()] for m in self.metrics]
        else:
            raise ValueError('data_mode must be dict or list')
        return s

    def reset(self):
        for m in self.metrics:
            m.reset()

    def get_params(self):
        return [inspect.signature(m).parameters for m in self.metrics]
