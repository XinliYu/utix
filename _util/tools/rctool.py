import shutil
from os import path

from utix.general import iter__, hprint_message
from utix.pathex import get_main_name


def install_bashrc(rcdir, mainrc, other, target='~/.bashrc', verbose=__debug__):
    """
    Assembles multiple bashrc files.
    This script supports splitting a large complicated bashrc file to multiple files in a directory, and then assemble them back into one file when needed.
    Use this function with the `install_bashrc.sh` bash script to achieve easier management of bashrc files.
    :param rcdir:
    :param mainrc:
    :param other:
    :param target:
    :param verbose:
    :return:
    """
    if isinstance(other, str):
        other = [x.strip() for x in other.split(',')]
    mainrc = path.join(rcdir, mainrc)
    if verbose:
        hprint_message(title='mainrc', content=mainrc)
        hprint_message(title='others', content=other)
        hprint_message(title='target', content=target)
    shutil.copyfile(mainrc, target)
    with open(target, 'a') as f:
        f.write('\n')
        for other in iter__(other):
            other = path.join(rcdir, other)
            f.write(f'source "{other}"\n')
            f.write(f'alias vim_{get_main_name(other)}="vim {other}"\n')
        f.write(f'alias vimsrc="vim {target}"\n')
        f.write(f'alias srcsrc="source {target}"\n')
