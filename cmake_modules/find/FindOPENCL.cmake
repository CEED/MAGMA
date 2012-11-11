###
#
# @file FindOPENCL.cmake
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
# This module finds an installed OPENCL library.
# OPENCL is distributed at : 
#
# This module can be tuned for searching in a specific folder:
#  - OPENCL_DIR      : path of the presumed top directory of OPENCL
#               (the library will be searched in this in the first time).
#
#
# This module sets the following variables:
#  - OPENCL_FOUND        : set to true if a library is found.
#
#  - OPENCL_PATH         : path of the top directory of fxt installation.
#  - OPENCL_VERSION      : version of fxt.
#
#  - OPENCL_LDFLAGS      : string of all required linker flags (with -l and -L).
#  - OPENCL_LIBRARY      : list of all required library  (absolute path).
#  - OPENCL_LIBRARIES    : list of required linker flags (without -l and -L).
#  - OPENCL_LIBRARY_PATH : path of the library directory of fxt installation.
#
#  - OPENCL_INLCUDE_PATH : path of the include directory of fxt installation.
#
###

# Early exit if already searched
IF(OPENCL_FOUND)
    MESSAGE(STATUS "Looking for OPENCL - already found")
    RETURN()
ENDIF(OPENCL_FOUND)

# Load required modules
INCLUDE(populatePACKAGE)
INCLUDE(findPACKAGE)
INCLUDE(infoOPENCL)

# Begin section - Looking for OPENCL
MESSAGE(STATUS "Looking for OPENCL")
    
# Define extra directory to look for
IF(MORSE_SEPARATE_PROJECTS)
    SET(OPENCL_DIR ${CMAKE_INSTALL_PREFIX}/opencl)
ELSE(MORSE_SEPARATE_PROJECTS)
    SET(OPENCL_DIR ${CMAKE_INSTALL_PREFIX})
ENDIF(MORSE_SEPARATE_PROJECTS)

# Looking for dependencies
FIND_AND_POPULATE_LIBRARY("OPENCL")

# Define parameters for FIND_MY_PACKAGE
OPENCL_INFO_FIND()

# Search for the library
FIND_MY_PACKAGE("OPENCL")

# Begin section - Looking for OPENCL
IF(OPENCL_FOUND)
    MESSAGE(STATUS "Looking for OPENCL - found")
ELSE(OPENCL_FOUND)
    MESSAGE(STATUS "Looking for OPENCL - not found")
ENDIF(OPENCL_FOUND)

##
## @end file FindOPENCL.cmake
##
