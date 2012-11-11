###
#
# @file FindAPPML.cmake
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
# This module finds an installed APPML library.
# APPML is distributed at : 
#
# This module can be tuned for searching in a specific folder:
#  - APPML_DIR      : path of the presumed top directory of APPML
#               (the library will be searched in this in the first time).
#
#
# This module sets the following variables:
#  - APPML_FOUND        : set to true if a library is found.
#
#  - APPML_PATH         : path of the top directory of fxt installation.
#  - APPML_VERSION      : version of fxt.
#
#  - APPML_LDFLAGS      : string of all required linker flags (with -l and -L).
#  - APPML_LIBRARY      : list of all required library  (absolute path).
#  - APPML_LIBRARIES    : list of required linker flags (without -l and -L).
#  - APPML_LIBRARY_PATH : path of the library directory of fxt installation.
#
#  - APPML_INLCUDE_PATH : path of the include directory of fxt installation.
#
###

# Early exit if already searched
IF(APPML_FOUND)
    MESSAGE(STATUS "Looking for APPML - already found")
    RETURN()
ENDIF(APPML_FOUND)

# Load required modules
INCLUDE(populatePACKAGE)
INCLUDE(findPACKAGE)
INCLUDE(infoAPPML)

# Begin section - Looking for APPML
MESSAGE(STATUS "Looking for APPML")
    
# Define extra directory to look for
IF(MORSE_SEPARATE_PROJECTS)
    SET(APPML_DIR ${CMAKE_INSTALL_PREFIX}/appml)
ELSE(MORSE_SEPARATE_PROJECTS)
    SET(APPML_DIR ${CMAKE_INSTALL_PREFIX})
ENDIF(MORSE_SEPARATE_PROJECTS)

# Looking for dependencies
FIND_AND_POPULATE_LIBRARY("APPML")

# Define parameters for FIND_MY_PACKAGE
APPML_INFO_FIND()

# Search for the library
FIND_MY_PACKAGE("APPML")

# Begin section - Looking for APPML
IF(APPML_FOUND)
    MESSAGE(STATUS "Looking for APPML - found")
ELSE(APPML_FOUND)
    MESSAGE(STATUS "Looking for APPML - not found")
ENDIF(APPML_FOUND)

##
## @end file FindAPPML.cmake
##
