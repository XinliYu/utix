from paramiko import SSHClient
from utix.external.scp import SCPClient
import utix.pathex as pathex
import utix.ioex as ioex
import utix.strex as strex
from os import path
import os
import shutil

from utix.general import eprint, hprint_message, eprint_message, iter__, str2val


def scp_upload(src_dir, host, username, password, dst_dir, pattern='*', recursive=True, server_path_sep='/', ignore_error='silent', ignore_unchanged_files=True, hash_block_size=65536, ssh_timout=15.0, **kwargs):
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(hostname=host,
                username=username,
                password=password,
                timeout=ssh_timout,
                **kwargs)

    scp = SCPClient(ssh.get_transport())

    src_dir = path.expandvars(src_dir)
    if ignore_unchanged_files:
        filehash_path = path.join(src_dir, f"{strex.hash_str('/'.join((host, username, dst_dir, str(pattern), str(recursive))))}_scp_file_hashes")
        filehash_dict = ioex.pickle_load(filehash_path, compressed=True) if path.exists(filehash_path) else {}

    for local_file, file_name in pathex.iter_files_by_pattern(dir_or_dirs=src_dir, pattern=pattern, recursive=recursive, full_path=pathex.FullPathMode.FullPathRelativePathTuple):
        if ignore_unchanged_files:
            filehash = ioex.hash_file(local_file, block_size=hash_block_size)
            if filehash == filehash_dict.get(local_file, None):
                continue

        remote_file = path.join(dst_dir, file_name)
        if server_path_sep != os.sep:
            remote_file = remote_file.replace(os.sep, server_path_sep)

        try:
            scp.put(local_file, remote_file)
        except Exception as err:
            if "No such file or directory" in str(err):
                ssh.exec_command(f"mkdir -p {path.dirname(remote_file)}")
                try:
                    scp.put(local_file, remote_file)
                except Exception as err:
                    if ignore_error is True:
                        eprint_message(title='failed', content=local_file)
                        print(type(err), err)
                        continue
                    elif ignore_error == 'silent':
                        continue
                    else:
                        raise err
            else:
                if ignore_error is True:
                    eprint_message(title='failed', content=local_file)
                    print(type(err), err)
                    continue
                elif ignore_error == 'silent':
                    continue
                else:
                    raise err

        hprint_message(title='success', content=local_file)
        if ignore_unchanged_files:
            filehash_dict[local_file] = filehash

    scp.close()

    if ignore_unchanged_files:
        ioex.pickle_save(filehash_dict, filehash_path, compressed=True)

# scp_upload(src_dir='E:\Data\graph', host='dgx-1.hpc.temple.edu', username='tuf72841', password='Tuf119817_666', dst_dir='/home/tuf72841/graph')
