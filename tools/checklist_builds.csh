#!/bin/tcsh
#
# Compiles MAGMA with given build configuration,
# saving output to builds/<build>/*.txt, and
# saving objects and executables to builds/<build>/obj.tar.gz
# Takes one or more build configurations, which are suffix on make.inc.* files.
# Usage:
#     ./tools/checklist_builds.csh [ macos acml atlas mkl-gcc mkl-icc mkl-gcc-ilp64 mkl-icc-ilp64 openblas ]
#
# @author Mark Gates

mkdir builds

foreach build ( $* )
    echo "----------------------------------------"
    touch make.inc
    make clean >& /dev/null
    rm make.inc

    echo "========================================"
    echo "build $build"
    echo
    ln -s  make.inc.$build  make.inc
    mkdir builds/$build

    echo "make lib " `date`
    (make lib  >! builds/$build/lib-out.txt) >&! builds/$build/lib-err.txt
    if ( $? ) then
       echo "FAILED"
    endif

    echo "make test" `date`
    (make test >! builds/$build/test-out.txt) >&! builds/$build/test-err.txt
    if ( $? ) then
        echo "FAILED"
    endif

    cd sparse-iter
    echo "sparse:"
    echo "make lib " `date`
    (make lib  >! ../builds/$build/sparse-lib-out.txt) >&! ../builds/$build/sparse-lib-err.txt
    if ( $? ) then
        echo "FAILED"
    endif

    echo "make test" `date`
    (make test >! ../builds/$build/sparse-test-out.txt) >&! ../builds/$build/sparse-test-err.txt
    if ( $? ) then
        echo "FAILED"
    endif
    cd ..

    echo "tar objs " `date`
    ./tools/find_obj_files.csh
    tar -zcf builds/$build/obj.tar.gz `cat obj-files.txt`
    echo "done     " `date`
    echo

    echo "[return to continue]"
    set string=$<
end
