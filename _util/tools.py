from paramiko import SSHClient
from scp import SCPClient
import utilx.path_ext as pathx
import utilx.io_ext as iox
import utilx.str_ext as strx
from os import path
import os

from _util.general_ext import eprint, hprint_message, eprint_message


def scp_upload(src_dir, host, username, password, dst_dir, pattern='*', recursive=True, server_path_sep='/', ignore_error='silent', ignore_unchanged_files=True, hash_block_size=65536, **kwargs):
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(hostname=host,
                username=username,
                password=password, **kwargs)

    scp = SCPClient(ssh.get_transport())

    src_dir = path.expandvars(src_dir)
    if ignore_unchanged_files:
        filehash_path = path.join(src_dir, f"{strx.hash_str('/'.join((host, username, dst_dir, str(pattern), str(recursive))))}_scp_file_hashes")
        filehash_dict = iox.pickle_load(filehash_path, compressed=True) if path.exists(filehash_path) else {}

    for local_file, file_name in pathx.iter_files_by_pattern(dir_or_dirs=src_dir, pattern=pattern, recursive=recursive, full_path=pathx.FullPathMode.FullPathRelativePathTuple):
        if ignore_unchanged_files:
            filehash = iox.hash_file(local_file, block_size=hash_block_size)
            if filehash == filehash_dict.get(local_file, None):
                continue

        remote_file = path.join(dst_dir, file_name)
        if server_path_sep != os.sep:
            remote_file = remote_file.replace(os.sep, server_path_sep)
        if ignore_error is True:
            try:
                scp.put(local_file, remote_file)
            except Exception as err:
                eprint_message(title='failed', content=local_file)
                print(err)
                continue
        elif ignore_error == 'silent':
            try:
                scp.put(local_file, remote_file)
            except:
                continue
        else:
            scp.put(local_file, remote_file)

        hprint_message(title='success', content=local_file)
        if ignore_unchanged_files:
            filehash_dict[local_file] = filehash

    scp.close()

    if ignore_unchanged_files:
        iox.pickle_save(filehash_path, filehash_dict, compressed=True)

# scp_upload(src_dir='.', host='dgx-1.hpc.temple.edu', username='tuf72841', password='Tuf119817_666', dst_dir='/home/tuf72841/PyModels2')
