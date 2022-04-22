# Author: weiwei
import os
import stat

import paramiko
from tqdm import tqdm


class SFTPHelper(object):
    def __init__(self):
        self.sftp = None

    def connect(self, username, password, remote_ip, remote_port=22):
        try:
            client = paramiko.Transport((remote_ip, remote_port))
            client.connect(username=username, password=password)
            self.sftp = paramiko.SFTPClient.from_transport(client)
        except IOError:
            raise IOError('SFTP connection failed!')
        return self.sftp

    def exists(self, path):
        flag = True
        try:
            stat = self.sftp.lstat(path)
            flag = True
        except IOError:
            flag = False
        return flag

    def makedirs(self, path):
        head, tail = os.path.split(path)
        if not tail:
            head, tail = path.split(head)
        if head and tail and not self.exists(head):
            self.makedirs(head)
        self.sftp.mkdir(path)

    def list_remote_dir(self, remote_dir):
        all_files = []
        files = self.sftp.listdir_attr(remote_dir)
        for x in files:
            filename = os.path.join(remote_dir, x.filename)
            if stat.S_ISDIR(x.st_mode):
                all_files.extend(self.list_remote_dir(filename))
            else:
                all_files.append(filename)
        return all_files

    def list_local_dir(self, local_dir):
        all_files = []
        files = os.listdir(local_dir)
        for x in files:
            filename = os.path.join(local_dir, x)
            if os.path.isdir(filename):
                all_files.extend(self.list_remote_dir(filename))
            else:
                all_files.append(filename)
        return all_files

    def put_dir(self, local_dir, remote_dir, verbose=False, skip_existed=False):
        local_files = self.list_local_dir(local_dir)
        if verbose:
            local_files = tqdm(local_files, desc=f'copy {local_dir} to {remote_dir}')
        for f in local_files:
            r = f.replace(local_dir, remote_dir)
            if skip_existed and self.exists(r):
                continue
            p, n = os.path.split(r)
            if not self.exists(p):
                self.makedirs(p)
            self.sftp.put(f, r)

    def put_file(self, local_file, remote_dir, skip_existed=False):
        r = local_file.replace(local_file, remote_dir)
        if skip_existed and self.exists(r):
            return

        p, n = os.path.split(r)
        if not self.exists(p):
            self.makedirs(p)
        self.sftp.put(local_file, r)

    def get_dir(self, remote_dir, local_dir, verbose=False, skip_existed=False):
        remote_files = self.list_remote_dir(remote_dir)
        if verbose:
            remote_files = tqdm(remote_files, desc=f'copy {local_dir} from {remote_dir}')
        for f in remote_files:
            l = f.replace(remote_dir, local_dir)
            if skip_existed and os.path.exists(l):
                continue
            p, n = os.path.split(l)
            if not os.path.exists(p):
                os.makedirs(p)
            self.sftp.get(f, l)

    def get_file(self, remote_file, local_dir, skip_existed=False):
        l = remote_file.replace(remote_file, local_dir)
        if skip_existed and os.path.exists(l):
            return
        p, n = os.path.split(l)
        if not os.path.exists(p):
            os.makedirs(p)
        self.sftp.get(remote_file, l)

    def close(self):
        self.sftp.close()


if __name__ == '__main__':
    local_root = '/home/user/Desktop'
    remote_root = '/media/user/File/'
    username, password = 'user', 'user'
    remote_ip = '127.0.0.1'

    file_path = 'user'

    # create sftp
    sftp = SFTPHelper()
    sftp.connect(username, password, remote_ip)

    remote_file = os.path.join(remote_root, file_path)
    local_file = os.path.join(local_root, file_path)
    sftp.put_dir(local_file, remote_file, verbose=True)
    # sftp.get_dir(remote_file, local_file, verbose=True)
