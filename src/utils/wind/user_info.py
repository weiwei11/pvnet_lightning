# @author: ww

import getpass
import os
import socket


def get_user_name():
    """
    Get user name
    :return:

    >>> get_user_name()
    'user'
    """
    return getpass.getuser()


def get_computer_name():
    """
    Get computer name
    :return:

    >>> get_computer_name()
    'user-computer'
    """
    return socket.gethostname()


def get_home_path():
    """
    Get absolute path of home
    :return:

    >>> get_home_path()
    '/home/user'
    """
    return os.path.expanduser('~')


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    # print(socket.gethostname())
    # print(getpass.getuser())
    # print(os.path.expanduser('~'))
