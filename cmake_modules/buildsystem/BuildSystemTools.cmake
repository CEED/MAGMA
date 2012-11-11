###
#
#  @file BuildSystemTools.cmake 
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

###
#
# GET_VERSION: Get the version of the software by parsing a file
#
###

MACRO(GET_VERSION _PACKAGE _filepath)

    FILE(READ "${_filepath}" _file)
    STRING(REGEX REPLACE "(.*)define([ \t]*)${_PACKAGE}_VERSION_MAJOR([ \t]*)([0-9]+)(.*)" "\\4" ${_PACKAGE}_VERSION_MAJOR "${_file}")
    STRING(REGEX REPLACE "(.*)define([ \t]*)${_PACKAGE}_VERSION_MINOR([ \t]*)([0-9]+)(.*)" "\\4" ${_PACKAGE}_VERSION_MINOR "${_file}")
    STRING(REGEX REPLACE "(.*)define([ \t]*)${_PACKAGE}_VERSION_MICRO([ \t]*)([0-9]+)(.*)" "\\4" ${_PACKAGE}_VERSION_PATCH "${_file}")
    SET(${_PACKAGE}_VERSION_NUMBER "${${_PACKAGE}_VERSION_MAJOR}.${${_PACKAGE}_VERSION_MINOR}.${${_PACKAGE}_VERSION_PATCH}")
    #MESSAGE(STATUS "${_PACKAGE}_VERSION_MAJOR = -${${_PACKAGE}_VERSION_MAJOR}-")
    #MESSAGE(STATUS "${_PACKAGE}_VERSION_MINOR = -${${_PACKAGE}_VERSION_MINOR}-")
    #MESSAGE(STATUS "${_PACKAGE}_VERSION_MICRO = -${${_PACKAGE}_VERSION_MICRO}-")

ENDMACRO(GET_VERSION)


###
#
# ADD_FLAGS: Add _FLAGS in _VARIABLE with correct whitespace.
#
###
MACRO(ADD_FLAGS _VARIABLE _FLAGS)

    IF("${${_VARIABLE}}" STREQUAL "")
        SET(${_VARIABLE} "${_FLAGS}")
    ELSE()
        SET(${_VARIABLE} "${${_VARIABLE}} ${_FLAGS}")
    ENDIF()

ENDMACRO(ADD_FLAGS _VARIABLE _FLAGS)

###
#
# ADD_FLAGS: Add _FLAGS in _VARIABLE with correct whitespace.
#
###
MACRO(DEFINE_LIBRARY _NAME)

    MESSAGE(STATUS "Looking for lib${_NAME}")
    STRING(TOUPPER "${_NAME}" _UPPERNAME)
    STRING(REPLACE "+" "P" _UPPERNAME "${_UPPERNAME}")
    FIND_LIBRARY(${_UPPERNAME}_LIBRARY
                 NAME ${_NAME}
                 PATHS ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES})
    MARK_AS_ADVANCED(${_UPPERNAME}_LIBRARY)

    IF(${_UPPERNAME}_LIBRARY)
        MESSAGE(STATUS "Looking for lib${_NAME} - found")
        GET_FILENAME_COMPONENT(_libpath ${${_UPPERNAME}_LIBRARY} PATH)
        SET(${_UPPERNAME}_LDFLAGS      "-L${_libpath} -l${_NAME}")
        SET(${_UPPERNAME}_LIBRARY_PATH "${_libpath}")
        SET(${_UPPERNAME}_LIBRARIES    "${_NAME}")
        SET(${_UPPERNAME}_FOUND        TRUE)
        SET(HAVE_${_UPPERNAME}         ON)
        LIST(APPEND MORSE_EXTRA_LIBS "-l${_NAME}")

    ELSE(${_UPPERNAME}_LIBRARY)
        MESSAGE(STATUS "Looking for lib${_NAME} - not found")

    ENDIF(${_UPPERNAME}_LIBRARY)

ENDMACRO(DEFINE_LIBRARY)

###
#
# DEFINE_PRIORITY: Provides a tool to set priority according to MORSE_USE_<PACKAGE>
#
###
MACRO(DEFINE_PRIORITY _PACKAGE _DEPENDENCY _STATUS_ON _STATUS_OFF _STATUS_DEFAULT)

    IF("^${MORSE_USE_${_DEPENDENCY}}$" STREQUAL "^ON$")
        SET(${_PACKAGE}_${_DEPENDENCY}_PRIORITY "${_STATUS_ON}")

    ELSEIF("^${MORSE_USE_${_DEPENDENCY}}$" STREQUAL "^OFF$")
        SET(${_PACKAGE}_${_DEPENDENCY}_PRIORITY "${_STATUS_OFF}")

    ELSE()
        SET(${_PACKAGE}_${_DEPENDENCY}_PRIORITY "${_STATUS_DEFAULT}")

    ENDIF()

ENDMACRO(DEFINE_PRIORITY)

###
#
# UPDATE_ENV: Macro to update user's environnement
#
###
MACRO(UPDATE_ENV _VARNAME _ADD)

    STRING(REGEX MATCH "${_ADD}" _STATUS "$ENV{${_VARNAME}}")
    IF(DEFINED _STATUS)
        SET(ENV{${_VARNAME}} "${_ADD}:$ENV{${_VARNAME}}")
    ENDIF()

ENDMACRO(UPDATE_ENV)

###
#
# GENERATE_PKGCONFIG_FILE: generate a file .pc according to the options 
#
###
MACRO(GENERATE_PKGCONFIG_FILE _PACKAGE _file) 

    # Clear entry
    # -----------
    SET(${_PACKAGE}_REQUIRED "")

    # Define required package
    # -----------------------
    IF(MAGMA)
        IF(HAVE_PLASMA)
            SET(MAGMA_REQUIRED "${MAGMA_REQUIRED} plasma")
        ENDIF()

    ELSEIF(MAGMA_MORSE)
       IF(HAVE_PLASMA)
            SET(MORSE_REQUIRED "${MORSE_REQUIRED} plasma")
        ENDIF()
        IF(MORSE_SCHED_STARPU)
            IF(HAVE_MPI)
                SET(MORSE_REQUIRED "${MORSE_REQUIRED} libstarpumpi")
            ELSE()
                SET(MORSE_REQUIRED "${MORSE_REQUIRED} libstarpu")
            ENDIF()
        ENDIF()
        IF(HAVE_HWLOC)
            SET(MORSE_REQUIRED "${MORSE_REQUIRED} hwloc")
        ENDIF()
        IF(HAVE_FXT)
            SET(MORSE_REQUIRED "${MORSE_REQUIRED} fxt")
        ENDIF()

    ENDIF()

    # Create .pc file
    # ---------------
    GET_FILENAME_COMPONENT(_output_file ${_file} NAME)
    STRING(REPLACE ".in" "" _output_file "${_output_file}")
    SET(_output_file "${CMAKE_BINARY_DIR}/${_output_file}")
    CONFIGURE_FILE("${_file}" "${_output_file}" @ONLY)

    # installation
    # ------------
    INSTALL(FILES ${_output_file} DESTINATION lib/pkgconfig)

ENDMACRO(GENERATE_PKGCONFIG_FILE)

##
## @end file BuildSystemTools.cmake
##
