# @author: ww
from functools import wraps
from timeit import default_timer
import sys


class StackTimer(object):
    # _name_stack = []
    _time_stack = []
    _begin_stack = []
    _end_stack = []
    _diff = None
    _begin_line = None
    _end_line = None

    @classmethod
    def tic(cls):
        # cls._name_stack.append(name)
        cls._begin_stack.append(sys._getframe().f_back.f_lineno)
        cls._time_stack.append(default_timer())

    @classmethod
    def toc(cls):
        cls._diff = default_timer() - cls._time_stack.pop()
        cls._end_stack.append(sys._getframe().f_back.f_lineno)
        cls._begin_line = cls._begin_stack.pop()
        cls._end_line = cls._end_stack.pop()
        return cls._diff

    @classmethod
    def print_time(cls, format_str=None, stdout=print):
        # name = cls._name_stack.pop()
        if format_str is not None:
            stdout(format_str.format(cls._diff))
        # elif name is not None:
        #     stdin(f'{name} cost time: {cls._diff}')
        else:
            stdout(f'{sys._getframe().f_back.f_code.co_name}({cls._begin_line}-{cls._end_line}) cost time: {cls._diff}sec')


def print_runtime(func, format_str=None, stdout=print):
    timer = StackTimer
    if format_str is None:
        format_str = func.__name__ + ' cost time: {}sec'

    @wraps(func)
    def wapper(*args, **kwargs):
        timer.tic()
        result = func(*args, **kwargs)
        timer.toc()
        timer.print_time(format_str, stdout)

        return result

    return wapper


if __name__ == '__main__':
    timer = StackTimer

    def test_timer1():
        timer.tic()
        j = 0
        for i in range(1000):
             j = j + i
        timer.toc()
        # timer.print_time()

    def test_timer2():
        timer.tic()
        j = 0
        for i in range(1000):
            j = j + i
        timer.toc()
        timer.print_time('test_time2 time: {}s')

    test_timer1()
    test_timer2()
    print(timer._time_stack, timer._begin_stack, timer._end_stack)
