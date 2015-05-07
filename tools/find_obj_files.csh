#!/bin/csh
#
# Finds all object files (.o, .a, .so, .mod, files with a+x, excluding scripts)
# and stores list in obj-files.txt
# Used in checklist_builds.csh
#
# @author Mark Gates

find . -name contrib -prune -o \
       -name .svn    -prune -o \
       \(    -name '*.o' \
          -o -name '*.a' \
          -o -name '*.so' \
          -o -name '*.mod' \
          -o -perm +a+x \
          -not -type d -not -name '*.pl' -not -name '*.py' -not -name '*.csh' -not -name '*.sh' \) \
       -print >! obj-files.txt
