###
#
#  @file FindFXT.cmake
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
####
# This module finds an installed FXT library.
# FXT is distributed at : 
#
# This module can be tuned for searching in a specific folder:
#  - FXT_DIR      : path of the presumed top directory of FXT
#               (the library will be searched in this in the first time).
#
#
# This module sets the following variables:
#  - FXT_FOUND        : set to true if a library is found.
#
#  - FXT_PATH         : path of the top directory of fxt installation.
#  - FXT_VERSION      : version of fxt.
#
#  - FXT_LDFLAGS      : string of all required linker flags (with -l and -L).
#  - FXT_LIBRARY      : list of all required library  (absolute path).
#  - FXT_LIBRARIES    : list of required linker flags (without -l and -L).
#  - FXT_LIBRARY_PATH : path of the library directory of fxt installation.
#
#  - FXT_INLCUDE_PATH : path of the include directory of fxt installation.
#
###

# Early exit if already searched
IF(FXT_FOUND)
    MESSAGE(STATUS "Looking for FXT - already found")
    RETURN()
ENDIF(FXT_FOUND)

# Load required modules
INCLUDE(populatePACKAGE)
INCLUDE(findPACKAGE)
INCLUDE(infoFXT)

# Begin section - Looking for FXT
MESSAGE(STATUS "Looking for FXT")
    
# Define extra directory to look for
IF(MORSE_SEPARATE_PROJECTS)
    SET(FXT_DIR ${CMAKE_INSTALL_PREFIX}/fxt)
ELSE(MORSE_SEPARATE_PROJECTS)
    SET(FXT_DIR ${CMAKE_INSTALL_PREFIX})
ENDIF(MORSE_SEPARATE_PROJECTS)

# Looking for dependencies
FIND_AND_POPULATE_LIBRARY("FXT")

# Define parameters for FIND_MY_PACKAGE
FXT_INFO_FIND()

# Search for the library
FIND_MY_PACKAGE("FXT")

# Begin section - Looking for FXT
IF(FXT_FOUND)
    MESSAGE(STATUS "Looking for FXT - found")
ELSE(FXT_FOUND)
    MESSAGE(STATUS "Looking for FXT - not found")
ENDIF(FXT_FOUND)

##
## @end file FindFXT.cmake
##
