# Author: weiwei

def merge_list(*args):
    a = []
    [a.extend(x) for x in args]
    return a
