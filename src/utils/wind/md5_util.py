# Author: weiwei
import os
import glob
from hashlib import md5


def generate_str_md5(s: str, encoding='utf-8'):
    """
    Generate md5 str for str object

    :param s: str
    :param encoding: str encoding, default is 'utf-8'
    :return: md5 of str

    >>> s = 'abcdefghijklmnopqrstuvwxyz'
    >>> generate_str_md5(s, 'utf-8')
    'c3fcd3d76192e4007dfb496cca67e13b'
    """
    m = md5(s.encode('utf-8'))
    # m.update(s)
    str_md5 = m.hexdigest()
    return str_md5


def generate_file_md5(filename):
    """
    Generate md5 str for file

    :param filename: file path
    :return: md5 of file

    >>> f = './LICENSE'
    >>> generate_file_md5(f)
    '08c536e577c5736f6ca90dc4d5bd7a26'
    """
    m = md5(open(filename, 'rb').read())
    file_md5 = m.hexdigest()
    return file_md5


def generate_file_status_md5(filename, mode='simple'):
    """
    Generate md5 for file status information

    :param filename:
    :param mode: 'simple' or 'all'
    :return: md5 of file status

    >>> generate_file_status_md5('./LICENSE')
    'a94f91fd96b9f8e666964dcc9f0f52e4'
    >>> generate_file_status_md5('./LICENSE', 'all')
    '9146c0b3761e180cf24de4e200972756'
    """
    if not os.path.exists(filename):
        raise FileExistsError('{} not exist!'.format(filename))

    file_info = os.stat(filename)
    if mode == 'simple':
        status_md5 = generate_str_md5(f'{file_info.st_size}{file_info.st_mtime}')
    elif mode == 'all':
        status_md5 = generate_str_md5(str(file_info))
    else:
        raise ValueError('The mode must be \'simple\' or \'all\'')

    return status_md5


def generate_files_status_md5(filename_list, mode='simple'):
    """
    Generate md5 for many files

    :param filename_list:
    :param mode: 'simple' or 'all'
    :return: list of md5 of file status

    >>> file_list = glob.glob('./test_resource/*')
    >>> print(file_list)
    ['./test_resource/test_write.yaml', './test_resource/test_config.yml', './test_resource/test_read.yml']
    >>> generate_files_status_md5(file_list)
    ['aa56bc3a1e18d4fc597346d9c4b738e9', 'f397d56519d6bb19ba96338cb9d70901', '3a2304556dec7c829bda2ddd37ec1f29']
    >>> generate_files_status_md5(file_list, 'all')
    ['ecff29e9a10b5db8550763221056c414', '8171c82b8c6b3c2ab52e45044e500c21', 'd9dfaecc52b49287a74b29af8bf8cafe']
    """
    status_md5_list = list(map(lambda x: generate_file_status_md5(x, mode), filename_list))
    return status_md5_list


def save_md5_sum_file(filename, md5_str_list, md5_name_list):
    """
    Save md5 sum file

    :param filename: path of md5 sum file
    :param md5_str_list: list of md5 str
    :param md5_name_list: list of name of md5 str
    :return:

    >>> file_list = glob.glob('./test_resource/*')
    >>> print(file_list)
    ['./test_resource/test_write.yaml', './test_resource/test_config.yml', './test_resource/test_read.yml']
    >>> md5_list = generate_files_status_md5(file_list)
    >>> print(md5_list)
    ['aa56bc3a1e18d4fc597346d9c4b738e9', 'f397d56519d6bb19ba96338cb9d70901', '3a2304556dec7c829bda2ddd37ec1f29']
    >>> save_md5_sum_file('./test_md5/test_md5.md5', md5_list, file_list)
    """
    with open(filename, 'w') as f:
        for md5_str, md5_name in zip(md5_str_list, md5_name_list):
            f.write(f'{md5_str} {md5_name}\n')


def read_md5_sum_file(filename):
    """
    Read md5 sum file

    :param filename: path of md5 sum file
    :return: md5_str_list, md5_name_list

    >>> md5_list, md5_names = read_md5_sum_file('./test_md5/test_md5.md5')
    >>> print(md5_list)
    ['aa56bc3a1e18d4fc597346d9c4b738e9', 'f397d56519d6bb19ba96338cb9d70901', '3a2304556dec7c829bda2ddd37ec1f29']
    >>> print(md5_names)
    ['./test_resource/test_write.yaml', './test_resource/test_config.yml', './test_resource/test_read.yml']
    """
    with open(filename, 'r') as f:
        line_list = f.readlines()
    data_list = list(map(lambda line: line.split(), line_list))
    md5_str_list = list(map(lambda x: x[0], data_list))
    md5_name_list = list(map(lambda x: x[1], data_list))
    return md5_str_list, md5_name_list


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    # s = 'abcdefghijklmnopqrstuvwxyz'
    # s_md5 = generate_md5(s)
    # print(s_md5)
