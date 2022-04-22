# @author: ww

import subprocess


def get_hash():
    """
    Get latest commit hash
    :return:

    >>> get_hash()
    '0650ffee018272f630fea535df44e75f0e3224f4'
    """
    cmd_str = 'git log -1 --format="%H"'
    exitcode, hash_str = subprocess.getstatusoutput(cmd_str)
    if exitcode != 0:
        hash_str = ''
    return hash_str


def get_last_file_hash(filename):
    """
    Get the latest commit hash for file
    :param filename:
    :return:

    >>> get_last_file_hash('git_info.py')
    '0650ffee018272f630fea535df44e75f0e3224f4'
    """
    cmd_str = 'git log -1 --format="%H" {}'.format(filename)
    exitcode, hash_str = subprocess.getstatusoutput(cmd_str)
    if exitcode != 0:
        hash_str = ''
    return hash_str


def get_author():
    """
    Get latest commit author name
    :return:

    >>> get_author()
    'GitHub'
    """
    cmd_str = 'git log -1 --format="%cn"'
    exitcode, commit_author = subprocess.getstatusoutput(cmd_str)
    if exitcode != 0:
        commit_author = ''
    return commit_author


def get_datetime():
    """
    Get latest commit datetime
    :return:

    >>> get_datetime()
    '2021-01-25 19:22:09 +0800'
    """
    cmd_str = 'git log -1 --format="%cd" --date=iso'
    exitcode, datetime = subprocess.getstatusoutput(cmd_str)
    if exitcode != 0:
        datetime = ''
    return datetime


def get_commit_title():
    """
    Get latest commit title
    :return:

    >>> get_commit_title()
    'Initial commit'
    """
    cmd_str = 'git log -1 --format="%s"'
    exitcode, commit_title = subprocess.getstatusoutput(cmd_str)
    if exitcode != 0:
        commit_title = ''
    return commit_title


def get_diff():
    """
    Get git diff
    :return:

    >>> get_diff()
    ''
    """
    cmd_str = 'git diff'
    exitcode, diff_str = subprocess.getstatusoutput(cmd_str)
    if exitcode != 0:
        diff_str = ''
    return diff_str


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    # cmd = 'git log -1 --format="%H %cn %cd %s" --date=iso'
    # print(subprocess.getstatusoutput(cmd))
