# @author: ww


import socket
from urllib.request import urlopen
import json


def get_local_ip():
    """
    Get local ip address
    :return:

    >>> get_local_ip()
    '192.168.1.123'
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
    finally:
        s.close()
    return local_ip


def get_net_ip():
    """
    Get internet ip address
    :return:

    >>> get_net_ip()
    '182.101.62.226'
    """
    network_ip = json.load(urlopen('http://jsonip.com'))['ip']
    return network_ip


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    # print(get_local_ip())
    # print(get_net_ip())
