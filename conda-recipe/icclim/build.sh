#!/bin/bash

MACHINE="$(uname 2>/dev/null)"

export CFLAGS="-fPIC -g -c -Wall"
export CXXFLAGS="${CFLAGS}"
export CPPFLAGS="-I${PREFIX}/include"
export LDFLAGS="-L${PREFIX}/lib"

LinuxInstallation() {
    # Build dependencies:

    gcc ${CXXFLAGS} icclim/libC.c -o icclim/libC.o || return 1;
    gcc -shared -o icclim/libC.so icclim/libC.o || return 1;

    ${PYTHON} setup.py install || exit 1;

    return 0;
}

case ${MACHINE} in
    'Linux')
        LinuxInstallation || exit 1;
        ;;
    'Darwin')
        LinuxInstallation || exit 1;
        ;;
    *)
        echo -e "Unsupported machine type: ${MACHINE}";
        exit 1;
        ;;
esac
