#!/bin/sh
#
# Finds doxygen groups that are in use,  sorts & puts in file "ingroup"
# Finds doxygen groups that are defined, sorts & puts in file "defgroup"
# Doing
#     diff ingroup defgroup
# provides an easy way to see what groups are used vs. defined.
#
# @author Mark Gates

cd .. && make generate && cd docs

grep -h '@ingroup' ../*/*.{h,c,cu,cpp} ../sparse-iter/*/*.{h,cu,cpp} | \
	perl -pe 's/^ *\*//;  s/^ +//;  s/\@ingroup/\@group/;' | \
	sort --unique > ingroup

egrep -h '^ *@defgroup' groups.dox | \
    egrep -v 'group_|core_blas' | \
    perl -pe 's/^ *\@defgroup +(\w+).*/\@group $1/;' | \
	sort > defgroup

opendiff ingroup defgroup
