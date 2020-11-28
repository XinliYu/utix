#! /bin/bash

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
INIT_FILE="$SCRIPTDIR/_install_bashrc_init.sh"
if test -f $INIT_FILE; then
	echo "Sources the initial file $INIT_FILE"
    source $INIT_FILE
fi

default_utixdir='~/workspace/utix'
read -p "Enter the path to the utix directory [$default_utixdir]: " utixdir
utixdir=$(bash -c "echo ${utixdir:-$default_utixdir}")
echo "The utix directory path is '$utixdir'"
default_rcdir='~/workspace/scripts/bashrcs'
read -p "Enter the path to the rc-file directory [$default_rcdir]": rcdir
rcdir=$(bash -c "echo ${rcdir:-$default_rcdir}")
echo "The rc-file directory path is {$rcdir}"
while ["$mainrc" == ""] 
do
	read -p "Enter the main rc file name (required): " mainrc
done
echo "The main rc file is $mainrc"

default_otherrc='"commonrc.sh,hbrc.sh,s3rc.sh,tmuxrc.sh"'
read -p "Enter other rc files: [$default_otherrc]: " otherrc
otherrc=$(bash -c "echo ${otherrc:-$default_otherrc}")
echo "Other rc files: $otherrc"

default_rctarget='~/.bashrc'
read -p "Enter the destination path [$default_rctarget]: " rctarget
rctarget=$(bash -c "echo ${rctarget:-$default_rctarget}")
echo "The destination path is $rctarget"

PYTHONPATH=$utixdir:$PYTHONPATH python -c "from sys import argv; from utix.tools.rctool import install_bashrc; install_bashrc(*argv[1:])" $rcdir $mainrc $otherrc $rctarget
source $rctarget