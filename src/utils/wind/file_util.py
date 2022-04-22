# @author: ww
import os
import re

import yaml


def read_yaml(yaml_file):
    """
    Read yaml file

    :param yaml_file:
    :return:

    >>> read_yaml('test_resource/test_read.yml')
    {'date': datetime.date(2019, 11, 6), 'pkg': {'python': {'version': '3.6.8', 'date': '{{ date }}'}, 'django': {'version': "{% if pkg.python.version|first == '2' %}1.8{% else %}2.2.6{% endif %}"}}}
    """
    with open(yaml_file) as f:
        data_str = f.read()
        data = yaml.safe_load(data_str)

    return data


def write_yaml(yaml_file, data):
    """
    Write yaml file

    :param yaml_file:
    :param data:
    :return:

    >>> import datetime
    >>> data = {'date': datetime.date(2019, 11, 6), 'pkg': {'python': {'version': '3.6.8', 'date': '{{ date }}'}, 'django': {'version': "{% if pkg.python.version|first == '2' %}1.8{% else %}2.2.6{% endif %}"}}}
    >>> write_yaml('test_resource/test_write.yaml', data)
    """
    with open(yaml_file, 'w') as f:
        yaml.dump(data, f)


def makedirs(path, verbose=True, stdout=print):
    """
    A wrap function for os.makedirs

    :param path:
    :param verbose: show message
    :param stdout:
    :return:

    >>> _ = os.system('rm -R testmakedirs')
    >>> makedirs('testmakedirs', True)
    make dirs: testmakedirs
    >>> makedirs('testmakedirs', True)
    """
    if not os.path.exists(path):
        os.makedirs(path)
        if verbose:
            stdout(f'make dirs: {path}')


def make_parent_dirs(path, verbose=True, stdout=print):
    """

    :param path:
    :param verbose:
    :param stdout:
    :return:
    """
    p, d = os.path.split(path)
    makedirs(p, verbose, stdout)


def match_files(path, pattern):
    """
    Match pattern str and return match files or dirs

    :param path:
    :param pattern:
    :return:

    >>> path = './'
    >>> pattern = 'file_*\.py'
    >>> match_files(path, pattern)
    []
    >>> path = './'
    >>> pattern = 'file_.*\.py'
    >>> match_files(path, pattern)
    ['./file_util.py']
    """
    file_pattern = re.compile(pattern)

    # list file and match specified pattern
    filename_list = os.listdir(path)
    match_result = list(map(file_pattern.match, filename_list))
    match_result = list(filter(lambda x: x[0] is not None, zip(match_result, filename_list)))
    match_filenames = list(map(lambda x: x[1], match_result))
    match_file_paths = list(map(lambda x: os.path.join(path, x), match_filenames))
    return match_file_paths


def match_paths(root, path_pattern):
    """
    Match pattern str and return match paths

    :param root:
    :param path_pattern:
    :return:

    >>> root = ''
    >>> pattern = 'file_.*'
    >>> match_paths(root, pattern)
    ['file_util.py']
    """
    root = os.getcwd() if root == '' else root

    if path_pattern == '':  # end
        return []

    split_list = path_pattern.split('/', maxsplit=1)  # a/b/c -> [a b/c]
    if len(split_list) == 1:
        split_list.append('')

    head, tail_pattern = split_list
    files = os.listdir(root)
    pattern = re.compile(head)
    match_result = list(filter(lambda x: pattern.fullmatch(x) is not None, files))
    if len(match_result) == 0:  # match fail
        return None

    # match all child paths
    all_match_paths = []
    for cur_file in match_result:
        child_match_paths = match_paths(os.path.join(root, cur_file), tail_pattern)
        if child_match_paths is None:
            continue
        elif len(child_match_paths) == 0:  # child match fail, so skip
            all_match_paths.append(cur_file)
        else:
            all_match_paths.extend(map(lambda x: os.path.join(cur_file, x), child_match_paths))
    return all_match_paths


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    # print(read_yaml('test_resource/test_read.yml'))
    # makedirs('a')
