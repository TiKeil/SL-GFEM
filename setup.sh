#!/bin/bash
#

skipvenv=0
while :; do
    case $1 in
        -s|--skipvenv)
            skipvenv=1
            ;;
        --)              # End of all options.
            shift
            break
            ;;
        -?*)
            printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
            ;;
        *)               # Default case: No more options, so break out of the loop.
            break
    esac

    shift
done

set -eu

# accounts for setup.sh not called from basedir
export BASEDIR="$(cd "$(dirname ${BASH_SOURCE[0]})" ; pwd -P )"

cd ${BASEDIR}
git submodule update --init --recursive

if [[ ${skipvenv} == 0 ]] ; then
  # initialize the virtualenv
  python3 -m venv venv
  source venv/bin/activate
fi

# install python dependencies into the virtualenv
pip3 install --upgrade pip
pip3 install -r requirements.txt

# install local pymor and slod version
cd "${BASEDIR}"
cd gridlod && pip install -e .
cd "${BASEDIR}"
cd slgfem && pip install -e .
cd "${BASEDIR}"
echo $PWD/ > $(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')/scripts.pth

pip install scikit-sparse

cd "${BASEDIR}"
echo
if [[ ${skipvenv} == 0 ]] ; then
  echo "All done! From now on run"
  echo "  source venv/bin/activate"
  echo "to activate the virtualenv!"
else
  echo "All done!"
fi
