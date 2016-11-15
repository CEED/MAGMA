#!/bin/sh
#
# Compile MAGMA with given build configurations,
# save output to builds/<config>/*.txt, and
# save libs and executables to builds/<config>/{lib, testing, sparse-testing},
# save objects to builds/<config>/obj.tar.gz.
# Takes one or more build configurations, which are suffices on make.inc.* files
# in the make.inc-examples directory.
# Usage:
#     ./tools/build.sh [-h|--help] [suffices from make.inc.*]
#
# @author Mark Gates

set -u

j=8
clean=1
sparse=1
tar=""
save=1
pause=""

usage="Usage: $0 [options] [acml macos mkl-gcc openblas ...]
    -h  --help      help
    -j #            parallel make threads, default $j
        --no-clean  skip 'make clean' before each build
        --no-sparse skip making sparse
        --no-save   skip saving libs and executables in builds/<config>/{lib, testing, sparse-testing}
    -t  --tar       tar object files in builds/<config>/obj.tar.gz
    -p  --pause     pause after each build"


# ----------------------------------------
# parse command line arguments
while [ $# -gt 0 ]; do
    case $1 in
        -h|--help)
            echo "$usage"
            exit
            ;;
        --no-clean)
            clean=""
            ;;
        --no-sparse)
            sparse=""
            ;;
        -t|--tar)
            tar=1
            ;;
        --no-save)
            save=""
            ;;
        -p|--pause)
            pause=1
            ;;
        -j)
            j=$2
            shift
            ;;
        --)
            shift
            break
            ;;
        -?*)
            echo "Error: unknown option: $1" >& 2
            exit
            ;;
        *)
            break
            ;;
    esac
    shift
done

if [ $# -gt 1 -a -z "$clean" ]; then
    echo "building multiple configurations requires cleaning; remove --no-clean"
    exit
fi


# ----------------------------------------
# usage: sep filename
# appends separator to existing file
function sep {
    if [ -e $1 ]; then
        echo "####################" `date` >> $1
    fi
}


# ----------------------------------------
# usage: run command output-filename error-filename
# runs command, saving stdout and stderr, and print error if it fails
function run {
    sep $2
    sep $3
    printf "%-32s %s\n"  "$1"  "`date`"
    #echo "$1 " `date`
    $1 >> $2 2>> $3
    if (($? > 0)); then
        echo "FAILED"
    fi
}


# ----------------------------------------
builds=builds/`date +%Y-%m-%d`
echo "builds directory $builds"

make="make -j$j"

if [ ! -d builds ]; then
    mkdir builds
fi

if [ ! -d $builds ]; then
    mkdir $builds
fi

for config in $@; do
    echo "========================================"
    echo "config $config"
    echo
    if [ ! -e make.inc-examples/make.inc.$config ]; then
        echo "Error: make.inc-examples/make.inc.$config does not exist"
        continue
    fi
    rm make.inc
    ln -s  make.inc-examples/make.inc.$config  make.inc

    if [ -d $builds/$config ]; then
        echo "$builds/$config already exists; append or make new directory? [aN]"
        read value
        if [ "$value" != "a" ]; then
            count=2
            while [ -d $builds/$config-$count ]; do
                count=$((count+1))
            done
            config=$config-$count
        fi
    fi
    mkdir $builds/$config
    
    if [ -n "$clean" ]; then
        echo "$make clean"
        touch make.inc
        $make clean > /dev/null
    else
        echo "SKIPPING CLEAN"
    fi
    
    run "$make lib"             $builds/$config/out-lib.txt          $builds/$config/err-lib.txt
    run "$make test -k"         $builds/$config/out-test.txt         $builds/$config/err-test.txt
    if [ -n "$sparse" ]; then
        run "$make sparse-lib"      $builds/$config/out-sparse-lib.txt   $builds/$config/err-sparse-lib.txt
        run "$make sparse-test -k"  $builds/$config/out-sparse-test.txt  $builds/$config/err-sparse-test.txt
    else
        echo "SKIPPING SPARSE"
    fi
    
    if [ -n "$save" ]; then
        echo "saving libs and executables to $builds/$config/{lib, testing, sparse-testing}"
        mkdir $builds/$config/lib
        mkdir $builds/$config/testing
        mkdir $builds/$config/sparse-testing
        mv lib/lib* $builds/$config/lib
        mv `find testing        -maxdepth 1 -perm -u+x -type f -not -name '*.py' -not -name '*.pl' -not -name '*.sh' -not -name '*.csh'` $builds/$config/testing
        if [ -n "$sparse" ]; then
            mv `find sparse/testing -maxdepth 1 -perm -u+x -type f -not -name '*.py' -not -name '*.pl' -not -name '*.sh' -not -name '*.csh'` $builds/$config/sparse-testing
        fi
    else
        echo "SKIPPING SAVE"
    fi
    
    if [ -n "$tar" ]; then
        echo "tar objs " `date`
        ./tools/find_obj_files.sh > obj-files.txt
        tar -zcf $builds/$config/obj.tar.gz -T obj-files.txt
        echo "done     " `date`
    else
        echo "SKIPPING TAR"
    fi
    
    if [ -n "$pause" ]; then
        echo "[return to continue]"
        read
    fi
done
