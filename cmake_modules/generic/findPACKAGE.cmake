###
#
# @file      : findPACKAGE.cmake
#
# @description   : Project MORSE (http://hiepacs.bordeaux.inria.fr/eagullo/morse)
#
# @version       :
# @created by    : Cedric Castagnede
# @creation date : 20-01-2012
# @last modified : Thu 05 Jul 2012 07:44:32 PM CEST
#
###
#
# This module finds an installed ${_NAMEVAR} library.
#
# This module sets the following variables:
#  - <PACKAGE>_FOUND        : set to true if a library is found.
#
#  - <PACKAGE>_PATH         : path of the top directory of <PACKAGE> installation.
#  - <PACKAGE>_VERSION      : version of the library.
#
#  - <PACKAGE>_LDFLAGS      : string of all required linker flags (with -l and -L). 
#  - <PACKAGE>_LIBRARY      : list of all required library  (absolute path).
#  - <PACKAGE>_LIBRARIES    : list of required linker flags (without -l and -L).
#  - <PACKAGE>_LIBRARY_PATH : path of the library directory of <PACKAGE> installation.
#
#  - <PACKAGE>_INLCUDE_PATH : path of the include directory of <PACKAGE> installation.
#
#  - <PACKAGE>_BINARY_PATH  : path of the binary directory of <PACKAGE> installation.
#
###
#
# You can set the variable DEBUG_MORSE_FINDPACKAGE to ON for printing all debug informations
#   eg: cmake <path> -DDEBUG_MORSE_FINDPACKAGE=ON <options>
#
###

INCLUDE(CheckFunctionExists)
INCLUDE(CheckFortranFunctionExists)
INCLUDE(FindPackageHandleStandardArgs)

# Find package
# ------------
MACRO(FIND_MY_PACKAGE _NAMEVAR
                      _check_c _check_fortran)

    # Found env var to check
    # ----------------------
    IF(WIN32)
        SET(_libdir LIB)
    ELSE(WIN32)
        IF(APPLE)
            SET(_libdir DYLD_LIBRARY_PATH)
        ELSE(APPLE)
            SET(_libdir LD_LIBRARY_PATH)
        ENDIF(APPLE)
    ENDIF(WIN32)

    # Set ${_NAMEVAR}_FOUND at FALSE by default
    # -----------------------------------------
    SET(${_NAMEVAR}_FOUND FALSE)
    SET(${_NAMEVAR}_ERROR_OCCURRED FALSE)

    # Optionally use pkg-config to detect include/library paths (if pkg-config is available)
    # --------------------------------------------------------------------------------------
    FIND_PACKAGE(PkgConfig QUIET)
    IF(PKG_CONFIG_EXECUTABLE AND ${_NAMEVAR}_name_pkgconfig)
        pkg_search_module(PC_${_NAMEVAR} QUIET ${${_NAMEVAR}_name_pkgconfig})
        DEBUG_PKG_CONFIG_OUTPUT(${_NAMEVAR})

    ELSE(PKG_CONFIG_EXECUTABLE AND ${_NAMEVAR}_name_pkgconfig)
        MESSAGE(STATUS "Looking for ${_NAMEVAR} - pkgconfig not used")

    ENDIF(PKG_CONFIG_EXECUTABLE AND ${_NAMEVAR}_name_pkgconfig)

    # Looking for version
    # -------------------
    IF(PC_${_NAMEVAR}_FOUND AND ${_NAMEVAR}_name_pkgconfig)
        SET(${_NAMEVAR}_VERSION ${PC_${_NAMEVAR}_VERSION})
    ENDIF(PC_${_NAMEVAR}_FOUND AND ${_NAMEVAR}_name_pkgconfig)

    # Looking for top directory
    # -------------------------
    IF(PC_${_NAMEVAR}_FOUND AND ${_NAMEVAR}_name_pkgconfig)
        SET(${_NAMEVAR}_PATH ${PC_${_NAMEVAR}_PREFIX})
    ENDIF(PC_${_NAMEVAR}_FOUND AND ${_NAMEVAR}_name_pkgconfig)

    # Looking for library
    # -------------------
    
    # Get and preporcess of collected informations
    SET(_all_libs "")
    LIST(APPEND _all_libs ${${_NAMEVAR}_name_library})
    LIST(APPEND _all_libs ${PC_${_NAMEVAR}_LIBRARIES})
    LIST(REMOVE_DUPLICATES _all_libs)
    LIST(FIND _all_libs "stdc++" is_extra_flags)
    IF("-1" MATCHES ${is_extra_flags})
        SET(is_extra_flags FALSE)
    ELSE()
        SET(is_extra_flags TRUE)
        LIST(REMOVE_ITEM _all_libs "stdc++")
    ENDIF()
    IF(${_NAMEVAR}_name_library)
        FOREACH(_lib_file ${_all_libs})
            FIND_LIBRARY(${_lib_file}_path
                         NAMES ${_lib_file}
                         PATHS ${PC_${_NAMEVAR}_LIBDIR} ${PC_${_NAMEVAR}_LIBRARY_DIRS} ${${_NAMEVAR}_DIR} ENV ${_libdir} 
                         PATH_SUFFIXES lib lib64 lib32
                        )
            MARK_AS_ADVANCED(${_lib_file}_path)
            IF(${_lib_file}_path)
                GET_FILENAME_COMPONENT(${_lib_file}_filename    ${${_lib_file}_path} NAME   )
                STRING(REGEX REPLACE "(.*)/${${_lib_file}_filename}" "\\1" ${_lib_file}_lib_path "${${_lib_file}_path}")
                LIST(APPEND ${_NAMEVAR}_LIBRARIES    "${_lib_file}")
                LIST(APPEND ${_NAMEVAR}_LIBRARY      "${${_lib_file}_path}")
                LIST(APPEND ${_NAMEVAR}_LIBRARY_PATH "${${_lib_file}_lib_path}")
                LIST(REMOVE_DUPLICATES ${_NAMEVAR}_LIBRARY_PATH)

            ELSE(${_lib_file}_path)
                SET(${_NAMEVAR}_ERROR_OCCURED TRUE)
                MESSAGE(STATUS "Looking for ${_lib_file} - not found")

            ENDIF(${_lib_file}_path)
        ENDFOREACH()

        SET(${_NAMEVAR}_LDFLAGS "")
        FOREACH(_lib_dir ${${_NAMEVAR}_LIBRARY_PATH})
            SET(${_NAMEVAR}_LDFLAGS "${${_NAMEVAR}_LDFLAGS}-L${_lib_dir} ")
        ENDFOREACH()
        FOREACH(_lib_flags ${${_NAMEVAR}_LIBRARIES})
            SET(${_NAMEVAR}_LDFLAGS "${${_NAMEVAR}_LDFLAGS}-l${_lib_flags} ")
        ENDFOREACH()

        # Fix case of stdc++
        IF(${is_extra_flags})
            LIST(APPEND ${_NAMEVAR}_LIBRARIES "stdc++")
            SET(${_NAMEVAR}_LDFLAGS "${${_NAMEVAR}_LDFLAGS}-lstdc++")

        ENDIF(${is_extra_flags})
    ENDIF(${_NAMEVAR}_name_library)

    # Looking for include
    # -------------------
    SET(_all_headers "")
    LIST(APPEND _all_headers ${${_NAMEVAR}_name_include})
    IF(${_NAMEVAR}_name_include)
        FOREACH(_header_file ${_all_headers})
            FIND_PATH(${_header_file}_path
                         NAMES ${_header_file}
                         PATHS ${PC_${_NAMEVAR}_INCLUDEDIR} ${PC_${_NAMEVAR}_INCLUDE_DIRS} ${${_NAMEVAR}_DIR}
                         PATH_SUFFIXES include ${${_NAMEVAR}_name_include_suffix} include/${${_NAMEVAR}_name_include_suffix}
                        )
            MARK_AS_ADVANCED(${_header_file}_path)
            IF(${_header_file}_path)
                STRING(REGEX REPLACE "(.*)/${_header_file}" "\\1" ${_header_file}_header_path "${${_header_file}_path}")
                LIST(APPEND ${_NAMEVAR}_INCLUDE_PATH "${${_header_file}_header_path}")
                LIST(REMOVE_DUPLICATES ${_NAMEVAR}_INCLUDE_PATH)

            #TODO: rajouter des tests sur les headers : CheckIncludeFile

            ELSE(${_header_file}_path)
                SET(${_NAMEVAR}_ERROR_OCCURED TRUE)
                MESSAGE(STATUS "Looking for ${_header_file} - not found")

            ENDIF(${_header_file}_path)
        ENDFOREACH()

    ENDIF(${_NAMEVAR}_name_include)

    # Looking for binary
    # ------------------
    SET(_all_binary "")
    LIST(APPEND _all_binary ${${_NAMEVAR}_name_binary})
    IF(${_NAMEVAR}_name_binary)

        FOREACH(_bin_file ${_all_binary})
            FIND_PATH(${_bin_file}_path
                         NAMES ${_bin_file}
                         PATHS ${${_NAMEVAR}_PATH} ${${_NAMEVAR}_DIR}
                         PATH_SUFFIXES bin bin64 bin32
                        )
            MARK_AS_ADVANCED(${_bin_file}_path)
            IF(${_bin_file}_path)
                STRING(REGEX REPLACE "(.*)/${_bin_file}" "\\1" ${_bin_file}_bin_path "${${_bin_file}_path}")
                LIST(APPEND ${_NAMEVAR}_BINARY_PATH "${${_bin_file}_bin_path}")
                LIST(REMOVE_DUPLICATES ${_NAMEVAR}_BINARY_PATH)

            ELSE(${_bin_file}_path)
                SET(${_NAMEVAR}_ERROR_OCCURED TRUE)
                MESSAGE(STATUS "Looking for ${_bin_file} - not found")

            ENDIF(${_bin_file}_path)
        ENDFOREACH()

    ENDIF(${_NAMEVAR}_name_binary)

    # Print the results obtained
    # --------------------------
    DEBUG_FIND_OUTPUT(${_NAMEVAR})

    # Check if the library works
    # --------------------------
    INCLUDE(checkPACKAGE)
    CHECK_PACKAGE(${_NAMEVAR})

    # Warning users on screen
    # -----------------------
    PACKAGE_HANDLE(${_NAMEVAR})

ENDMACRO(FIND_MY_PACKAGE)

###
#
#
#
###
MACRO(PACKAGE_HANDLE _NAMEVAR)

    # If one error occurred, we clean all variables
    # ---------------------------------------------
    IF(${${_NAMEVAR}_ERROR_OCCURRED})
        PACKAGE_CLEAN_VALUE(${_NAMEVAR})
        SET(${_NAMEVAR}_FOUND FALSE)
        #
        # TODO: Put test on REQUIRED - need to crash here
        #
        #IF()
        #    MESSAGE(FATAL_ERROR "Looking for ${_NAMEVAR} - not found")
        #ENDIF()

    ELSE(${${_NAMEVAR}_ERROR_OCCURRED})
        SET(${_NAMEVAR}_FOUND TRUE)

    ENDIF(${${_NAMEVAR}_ERROR_OCCURRED})

ENDMACRO(PACKAGE_HANDLE _NAMEVAR)

###
#
#
#
###
MACRO(PACKAGE_CLEAN_VALUE _NAMEVAR)

    SET(${_NAMEVAR}_VERSION          "${_NAMEVAR}_VERSION-NOTFOUND"     )
    SET(${_NAMEVAR}_PATH             "${_NAMEVAR}_PATH-NOTFOUND"        )
    IF(${_NAMEVAR}_name_library)
        SET(${_NAMEVAR}_LDFLAGS      "${_NAMEVAR}_LDFLAGS-NOTFOUND"     )
        SET(${_NAMEVAR}_LIBRARY      "${_NAMEVAR}_LIBRARY-NOTFOUND"     )
        SET(${_NAMEVAR}_LIBRARIES    "${_NAMEVAR}_LIBRARIES-NOTFOUND"   )
        SET(${_NAMEVAR}_LIBRARY_PATH "${_NAMEVAR}_LIBRARY_PATH-NOTFOUND")
    ENDIF(${_NAMEVAR}_name_library)
    IF(${_NAMEVAR}_name_binary)
        SET(${_NAMEVAR}_BINARY_PATH  "${_NAMEVAR}_BINARY_PATH-NOTFOUND" )
    ENDIF(${_NAMEVAR}_name_binary)
    IF(${_NAMEVAR}_name_include)
        SET(${_NAMEVAR}_INCLUDE_PATH "${_NAMEVAR}_INCLUDE_PATH-NOTFOUND")
    ENDIF(${_NAMEVAR}_name_include)

ENDMACRO(PACKAGE_CLEAN_VALUE)

###
#
# FIND_MY_PACAKGE_DEBUG: print all the results
#
###
MACRO(DEBUG_FIND_OUTPUT _NAMEVAR)

    IF(MORSE_DEBUG_CMAKE)

        # Load infoPACKAGE
        # ----------------
        INCLUDE(infoPACKAGE)
        INFO_FIND_PACKAGE()

        # Debug post-treatment
        # --------------------
        MESSAGE(STATUS "Find${_NAMEVAR}: Value for ${_NAMEVAR}")
        MESSAGE(STATUS "  * debug: ${_NAMEVAR}_ERROR_OCCURRED = ${${_NAMEVAR}_ERROR_OCCURRED}")
        MESSAGE(STATUS "  * debug: ${_NAMEVAR}_VERSION        = ${${_NAMEVAR}_VERSION}"       )
        MESSAGE(STATUS "  * debug: ${_NAMEVAR}_PATH           = ${${_NAMEVAR}_PATH}"          )
        IF(${_NAMEVAR}_name_library)
            MESSAGE(STATUS "  * debug: ${_NAMEVAR}_LIBRARY_PATH   = ${${_NAMEVAR}_LIBRARY_PATH}")
            MESSAGE(STATUS "  * debug: ${_NAMEVAR}_LDFLAGS        = ${${_NAMEVAR}_LDFLAGS}"     )
            MESSAGE(STATUS "  * debug: ${_NAMEVAR}_LIBRARY        = ${${_NAMEVAR}_LIBRARY}"     )
            MESSAGE(STATUS "  * debug: ${_NAMEVAR}_LIBRARIES      = ${${_NAMEVAR}_LIBRARIES}"   )
        ENDIF(${_NAMEVAR}_name_library)
        IF(${_NAMEVAR}_name_include)
            MESSAGE(STATUS "  * debug: ${_NAMEVAR}_INCLUDE_PATH   = ${${_NAMEVAR}_INCLUDE_PATH}")
        ENDIF(${_NAMEVAR}_name_include)
        IF(${_NAMEVAR}_name_binary)
            MESSAGE(STATUS "  * debug: ${_NAMEVAR}_BINARY_PATH    = ${${_NAMEVAR}_BINARY_PATH}" )
        ENDIF(${_NAMEVAR}_name_binary)

    ENDIF(MORSE_DEBUG_CMAKE)

ENDMACRO(DEBUG_FIND_OUTPUT)

###
#
# DEBUG_PKG_CONFIG_OUTPUT: print all the informations collect by `pkg_search_module'
#
###
MACRO(DEBUG_PKG_CONFIG_OUTPUT _NAMEVAR)

    IF(MORSE_DEBUG_CMAKE)

        # Load infoPACKAGE
        # ----------------
        INCLUDE(infoPACKAGE)
        INFO_FIND_PACKAGE()

        # Debug PKG-CONFIG
        # ----------------
        MESSAGE(STATUS "Find${_NAMEVAR}: Output form PKG-CONFIG - ${${_NAMEVAR}_name_pkgconfig}")
        MESSAGE(STATUS "  * debug: PC_${_NAMEVAR}_FOUND        = ${PC_${_NAMEVAR}_FOUND}"       )
        MESSAGE(STATUS "  * debug: PC_${_NAMEVAR}_VERSION      = ${PC_${_NAMEVAR}_VERSION}"     )
        MESSAGE(STATUS "  * debug: PC_${_NAMEVAR}_PREFIX       = ${PC_${_NAMEVAR}_PREFIX}"      )
        IF(${_NAMEVAR}_name_library)
            MESSAGE(STATUS "  * debug: PC_${_NAMEVAR}_LIBRARY_DIRS = ${PC_${_NAMEVAR}_LIBRARY_DIRS}")
            MESSAGE(STATUS "  * debug: PC_${_NAMEVAR}_LIBDIR       = ${PC_${_NAMEVAR}_LIBDIR}"      )
        ENDIF(${_NAMEVAR}_name_library)
        IF(${_NAMEVAR}_name_include)
            MESSAGE(STATUS "  * debug: PC_${_NAMEVAR}_INCLUDEDIR   = ${PC_${_NAMEVAR}_INCLUDEDIR}"  )
            MESSAGE(STATUS "  * debug: PC_${_NAMEVAR}_INCLUDE_DIRS = ${PC_${_NAMEVAR}_INCLUDE_DIRS}")
        ENDIF(${_NAMEVAR}_name_include)

    ENDIF(MORSE_DEBUG_CMAKE)

ENDMACRO(DEBUG_PKG_CONFIG_OUTPUT)

###
### END findPACKAGE.cmake
###
