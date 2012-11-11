###
#
#  @file FindHWLOC.cmake
#
#  @project MORSE
#  MORSE is a software package provided by:
#     Inria Bordeaux - Sud-Ouest,
#     Univ. of Tennessee,
#     Univ. of California Berkeley,
#     Univ. of Colorado Denver.
#
#  @version 0.1.0
#  @author Cedric Castagnede
#  @date 13-07-2012
#
###
#
# This module finds an installed HWLOC library.
# HWLOC is distributed at : 
#
# This module can be tuned for searching in a specific folder:
#  - HWLOC_DIR      : path of the presumed top directory of HWLOC
#               (the library will be searched in this in the first time).
#
#
# This module sets the following variables:
#  - HWLOC_FOUND        : set to true if a library is found.
#
#  - HWLOC_PATH         : path of the top directory of hwloc installation.
#  - HWLOC_VERSION      : version of hwloc.
#
#  - HWLOC_LDFLAGS      : string of all required linker flags (with -l and -L).
#  - HWLOC_LIBRARY      : list of all required library  (absolute path).
#  - HWLOC_LIBRARIES    : list of required linker flags (without -l and -L).
#  - HWLOC_LIBRARY_PATH : path of the library directory of hwloc installation.
#
#  - HWLOC_INLCUDE_PATH : path of the include directory of hwloc installation.
#
#  - HWLOC_BINARY_PATH  : path of the binary directory of hwloc installation.
#
###

# Early exit if already searched
IF(HWLOC_FOUND)
    MESSAGE(STATUS "Looking for HWLOC - already found")
    RETURN()
ENDIF(HWLOC_FOUND)

# Load required modules
INCLUDE(populatePACKAGE)
INCLUDE(findPACKAGE)
INCLUDE(infoHWLOC)

# Begin section - Looking for HWLOC
MESSAGE(STATUS "Looking for HWLOC")

# Define extra directory to look for
IF(MORSE_SEPARATE_PROJECTS)
    SET(HWLOC_DIR ${CMAKE_INSTALL_PREFIX}/hwloc)
ELSE(MORSE_SEPARATE_PROJECTS)
    SET(HWLOC_DIR ${CMAKE_INSTALL_PREFIX})
ENDIF(MORSE_SEPARATE_PROJECTS)

# Looking for dependencies
FIND_AND_POPULATE_LIBRARY("HWLOC")

# Define parameters for FIND_MY_PACKAGE
HWLOC_INFO_FIND()

# Search for the library
FIND_MY_PACKAGE("HWLOC")

# Begin section - Looking for HWLOC
IF(HWLOC_FOUND)
    MESSAGE(STATUS "Looking for HWLOC - found")
ELSE(HWLOC_FOUND)
    MESSAGE(STATUS "Looking for HWLOC - not found")
ENDIF(HWLOC_FOUND)

##
## @end file FindHWLOC.cmake
##
