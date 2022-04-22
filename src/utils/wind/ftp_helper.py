# @author: ww

import os
import sys
from ftplib import FTP
from tqdm import tqdm


class FTPHelper(object):

    def __init__(self):
        self.ftp = FTP()
        self.buf_size = 1024
        self.is_dir = False

    def connect(self, ftp_server, port, username, password):
        try:
            self.ftp.connect(ftp_server, port)
            self.ftp.login(username, password)
        except Exception:
            raise IOError('FTP connection failed!')

    def download_file(self, remote_path, local_path):
        with open(local_path, 'wb') as f:
            self.ftp.retrbinary(f'RETR {remote_path}', f.write, self.buf_size)
        return True

    def upload_file(self, local_path, remote_path):
        with open(local_path, 'rb') as f:
            self.ftp.storbinary(f'STOR {remote_path}', f, self.buf_size)
        return True

    def download_dir(self, remote_path, local_path):
        if not os.path.exists(local_path):
            os.makedirs(local_path)

        self.ftp.cwd(remote_path)
        for file_info in self.get_dir(remote_path):
            local_file = os.path.join(local_path, file_info['name'])
            if file_info['type'] == 'dir':
                if not os.path.exists(local_file):
                    os.makedirs(local_file)
                self.download_dir(file_info['name'], local_file)
            else:
                self.download_file(file_info['name'], local_file)
        self.ftp.cwd('..')

    def download_dir2(self, remote_path, local_path, verbose=True):
        if not os.path.exists(local_path):
            os.makedirs(local_path)

        if verbose:
            file_list = tqdm(self.list_dirs(remote_path), desc=f'Download {remote_path} to {local_path}')
        else:
            file_list = self.list_dirs(remote_path)

        for file_info in file_list:
            # local_file = os.path.join(local_path, file_info['name'])
            local_file = file_info['path'].replace(remote_path, local_path)
            if file_info['type'] == 'dir':
                if not os.path.exists(local_file):
                    os.makedirs(local_file)
            else:
                self.download_file(file_info['path'], local_file)

    def disconnect(self):
        self.ftp.quit()

    def get_dir(self, remote_path):
        file_list = []
        info_list = []
        self.ftp.dir('.', info_list.append)
        for file_info in info_list:
            file_info = file_info.split(' ')
            if file_info[0].startswith('d'):
                file_list.append({'name': file_info[-1], 'type': 'dir'})
            else:
                file_list.append({'name': file_info[-1], 'type': 'file'})
        return file_list

    def list_dirs(self, remote_path):
        file_list = []
        info_list = []
        self.ftp.dir(remote_path, info_list.append)
        for file_info in info_list:
            file_info = file_info.split(' ')
            name = file_info[-1]
            path = os.path.join(remote_path, name)
            if file_info[0].startswith('d'):
                file_list.append({'name': name, 'type': 'dir', 'path': path})
                file_list.extend(self.list_dirs(path))
            else:
                file_list.append({'name': name, 'type': 'file', 'path': path})
        return file_list


def download_dir(remote_path, local_path, remote_ip, port, username, password, verbose=True):
    ftp = FTPHelper()
    ftp.connect(remote_ip, port, username, password)
    ftp.download_dir2(remote_path, local_path)
    ftp.disconnect()
    return True


if __name__ == '__main__':
    ftp = FTPHelper()
    ftp.connect('127.0.0.1', 21, 'user', 'user')
    ftp.download_dir('/home/user/files', '/home/user/Desktop/files')
    ftp.download_dir2('/home/user/files', '/home/user/Desktop/files')
    ftp.disconnect()
