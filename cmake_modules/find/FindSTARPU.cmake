###
#
#  @file FindSTARPU.cmake
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
# This module finds an installed STARPU library.
# STARPU is distributed at : 
#
# This module can be tuned for searching in a specific folder:
#  - STARPU_DIR      : path of the presumed top directory of STARPU
#               (the library will be searched in this in the first time).
#
#
# This module sets the following variables:
#  - STARPU_FOUND        : set to true if a library is found.
#
#  - STARPU_PATH         : path of the top directory of starpu installation.
#  - STARPU_VERSION      : version of starpu.
#
#  - STARPU_LDFLAGS      : string of all required linker flags (with -l and -L).
#  - STARPU_LIBRARY      : list of all required library  (absolute path).
#  - STARPU_LIBRARIES    : list of required linker flags (without -l and -L).
#  - STARPU_LIBRARY_PATH : path of the library directory of starpu installation.
#
#  - STARPU_INLCUDE_PATH : path of the include directory of starpu installation.
#
#  - STARPU_BINARY_PATH  : path of the binary directory of starpu installation.
#
###

# Early exit if already searched
IF(STARPU_FOUND)
    MESSAGE(STATUS "Looking for STARPU - already found")
    RETURN()
ENDIF(STARPU_FOUND)

# Load required modules
INCLUDE(populatePACKAGE)
INCLUDE(findPACKAGE)
INCLUDE(infoSTARPU)

# Begin section - Looking for STARPU
MESSAGE(STATUS "Looking for STARPU")

# Define extra directory to look for
IF(MORSE_SEPARATE_PROJECTS)
    SET(STARPU_DIR ${CMAKE_INSTALL_PREFIX}/starpu)
ELSE(MORSE_SEPARATE_PROJECTS)
    SET(STARPU_DIR ${CMAKE_INSTALL_PREFIX})
ENDIF(MORSE_SEPARATE_PROJECTS)

# Looking for dependencies
FIND_AND_POPULATE_LIBRARY("STARPU")

# Define parameters for FIND_MY_PACKAGE
STARPU_INFO_FIND()

# Search for the library
FIND_MY_PACKAGE("STARPU")

# Begin section - Looking for STARPU
IF(STARPU_FOUND)
    MESSAGE(STATUS "Looking for STARPU - found")
ELSE(STARPU_FOUND)
    MESSAGE(STATUS "Looking for STARPU - not found")
ENDIF(STARPU_FOUND)

##
## @end file FindSTARPU.cmake
##
