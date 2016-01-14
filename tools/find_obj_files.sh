#!/bin/sh
#
# Finds all object files (.o, .a, .so, .mod, files with a+x, excluding scripts)
# Used in checklist_builds.sh
#
# @author Mark Gates

find . -name contrib -prune -o \
       -name .svn    -prune -o \
       -name build   -prune -o \
       -name builds  -prune -o \
       -name tools   -prune -o \
       -name CMakeFiles -prune -o \
       \(    -name '*.o' \
          -o -name '*.a' \
          -o -name '*.so' \
          -o -name '*.mod' \
          -o -perm +a+x \
          -not -type d -not -name '*.pl' -not -name '*.py' -not -name '*.csh' -not -name '*.sh' \) \
       -print
