###
#
#  @file FindLAPACKE.cmake
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
# This module finds an installed LAPACKE library.
# LAPACKE is distributed at : 
#
# This module can be tuned for searching in a specific folder:
#  - LAPACKE_DIR      : path of the presumed top directory of LAPACKE
#               (the library will be searched in this in the first time).
#
#
# This module sets the following variables:
#  - LAPACKE_FOUND        : set to true if a library is found.
#
#  - LAPACKE_PATH         : path of the top directory of lapacke installation.
#  - LAPACKE_VERSION      : version of lapacke.
#
#  - LAPACKE_LDFLAGS      : string of all required linker flags.
#  - LAPACKE_LIBRARY      : list of all required library  (absolute path).
#  - LAPACKE_LIBRARIES    : list of required linker flags (without -l and -L).
#  - LAPACKE_LIBRARY_PATH : path of the library directory of lapacke installation.
#
#  - LAPACKE_INLCUDE_PATH : path of the include directory of lapacke installation.
#
###

# Early exit if already searched
IF(LAPACKE_FOUND)
    MESSAGE(STATUS "Looking for LAPACKE - already found")
    RETURN()
ENDIF(LAPACKE_FOUND)

# Load required modules
INCLUDE(populatePACKAGE)
INCLUDE(findPACKAGE)
INCLUDE(infoLAPACKE)

# Begin section - Looking for LAPACKE
MESSAGE(STATUS "Looking for LAPACKE")

# Define extra directory to look for
IF(MORSE_SEPARATE_PROJECTS)
    SET(LAPACKE_DIR ${CMAKE_INSTALL_PREFIX}/lapacke)
ELSE(MORSE_SEPARATE_PROJECTS)
    SET(LAPACKE_DIR ${CMAKE_INSTALL_PREFIX})
ENDIF(MORSE_SEPARATE_PROJECTS)

# Looking for dependencies
FIND_AND_POPULATE_LIBRARY("LAPACKE")
IF(HAVE_M)
    LIST(APPEND LAPACKE_FIND_DEPS "M")
ENDIF(HAVE_M)

# Define parameters for FIND_MY_PACKAGE
LAPACKE_INFO_FIND()

# Search for the library
FIND_MY_PACKAGE("LAPACKE")

# Begin section - Looking for LAPACKE
IF(LAPACKE_FOUND)
    MESSAGE(STATUS "Looking for LAPACKE - found")
ELSE(LAPACKE_FOUND)
    MESSAGE(STATUS "Looking for LAPACKE - not found")
ENDIF(LAPACKE_FOUND)

##
## @end file FindLAPACKE.cmake
##
