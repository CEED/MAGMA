###
#
#  @file installHWLOC.cmake
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

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
INCLUDE(installExternalPACKAGE)

MACRO(INSTALL_HWLOC _MODE)

    # Define prefix paths
    # -------------------
    IF(MORSE_SEPARATE_PROJECTS)
        SET(HWLOC_PATH ${CMAKE_INSTALL_PREFIX}/hwloc)
    ELSE(MORSE_SEPARATE_PROJECTS)
        SET(HWLOC_PATH ${CMAKE_INSTALL_PREFIX})
    ENDIF(MORSE_SEPARATE_PROJECTS)

    # Define options
    # --------------
    UNSET(HWLOC_CONFIG_OPTS)
    LIST(APPEND HWLOC_CONFIG_OPTS --prefix=${HWLOC_PATH})
    LIST(APPEND HWLOC_CONFIG_OPTS --enable-shared)
    IF(NOT BUILD_SHARED_LIBS)
        LIST(APPEND HWLOC_CONFIG_OPTS --enable-static)
    ENDIF()

    # Define steps of installation
    # ----------------------------
    SET(HWLOC_SOURCE_PATH ${CMAKE_BINARY_DIR}/externals/hwloc)
    SET(HWLOC_BUILD_PATH  ${CMAKE_BINARY_DIR}/externals/hwloc)
    SET(HWLOC_CONFIG_CMD  ./configure)
    SET(HWLOC_BUILD_CMD   ${CMAKE_MAKE_PROGRAM})
    SET(HWLOC_INSTALL_CMD ${CMAKE_MAKE_PROGRAM} install)

    # Install the external package
    # ----------------------------
    INSTALL_EXTERNAL_PACKAGE("hwloc" "${HWLOC_USED_MODE}")

    # Set linker flags
    # ----------------
    SET(HWLOC_BINARY_PATH  "${HWLOC_BUILD_PATH}/utils/.libs")
    SET(HWLOC_LIBRARY_PATH "${HWLOC_BUILD_PATH}/src/.libs")
    SET(HWLOC_INCLUDE_PATH "${HWLOC_BUILD_PATH}/include")
    SET(HWLOC_LIBRARY      "${HWLOC_LIBRARY_PATH}/libhwloc.${MORSE_LIBRARY_EXTENSION}")
    SET(HWLOC_LDFLAGS      "-L${HWLOC_LIBRARY_PATH} -lhwloc")
    SET(HWLOC_LIBRARIES    "hwloc")

ENDMACRO(INSTALL_HWLOC)

##
## @end file installHWLOC.cmake
##

