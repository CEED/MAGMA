#!/bin/tcsh
#
# Compiles MAGMA with given build configuration,
# saving output to builds/<build>/*.txt, and
# saving objects and executables to builds/<build>/obj.tar.gz
# Takes one or more build configurations, which are suffix on make.inc.* files.
# Usage:
#     ./tools/checklist_builds.csh [suffices from make.inc.*]
#
# @author Mark Gates

setenv usage "Usage: $0 [macos acml atlas mkl-gcc mkl-icc mkl-gcc-ilp64 mkl-icc-ilp64 openblas]"

setenv builds builds/`date +%Y-%m-%d`
echo "builds directory $builds"

setenv make 'make -j8'

if ( ! -d $builds ) then
    mkdir $builds
endif

which make

foreach build ( $* )
    echo "========================================"
    echo "build $build"
    echo
    if ( ! -e make.inc.$build ) then
        echo "Error: make.inc.$build does not exist"
        echo "$usage"
        echo "[return to continue]"
        set string=$<
        continue
    endif
    rm make.inc
    ln -s  make.inc.$build  make.inc
    #if ( -d $builds/$build ) then
    #    setenv build2 `uniqname $builds/$build`
    #    echo "mv $builds/$build $build2"
    #    mv $builds/$build $build2
    #endif
    mkdir $builds/$build

    echo "$make clean"
    touch make.inc
    $make clean >& /dev/null

    echo "$make lib " `date`
    ($make lib  >! $builds/$build/out-lib.txt) >&! $builds/$build/err-lib.txt
    if ( $? ) then
       echo "FAILED"
    endif

    echo "$make test -k" `date`
    ($make test >! $builds/$build/out-test.txt) >&! $builds/$build/err-test.txt
    if ( $? ) then
        echo "FAILED"
    endif

    cd sparse-iter
    echo "sparse:"
    echo "$make lib " `date`
    ($make lib  >! ../$builds/$build/out-sparse-lib.txt) >&! ../$builds/$build/err-sparse-lib.txt
    if ( $? ) then
        echo "FAILED"
    endif

    echo "$make test -k" `date`
    ($make test >! ../$builds/$build/out-sparse-test.txt) >&! ../$builds/$build/err-sparse-test.txt
    if ( $? ) then
        echo "FAILED"
    endif
    cd ..

    echo "tar objs " `date`
    ./tools/find_obj_files.csh
    tar -zcf $builds/$build/obj.tar.gz `cat obj-files.txt`
    echo "done     " `date`
    echo

    echo "[return to continue]"
    set string=$<
end
