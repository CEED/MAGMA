###
#
# @file      : checkPACKAGE.cmake
#
# @description   : Project MORSE (http://hiepacs.bordeaux.inria.fr/eagullo/morse)
#
# @version       :
# @created by    : Cedric Castagnede
# @creation date : 20-01-2012
# @last modified : Thu 05 Jul 2012 08:46:13 PM CEST
#
###

MACRO(CHECK_PACKAGE _NAME)

    # Import cmake modules
    # --------------------
    INCLUDE(CheckIncludeFiles)
    INCLUDE(CheckFunctionExists)
    INCLUDE(CheckFortranFunctionExists)

    # Load infoPACKAGE
    # ----------------
    INCLUDE(infoPACKAGE)
    INFO_FIND_PACKAGE()

    # Set internal variable
    # ---------------------
    STRING(TOUPPER "${_NAME}" _NAMEVAR)

    # Set ${_NAMEVAR}_ERROR_OCCURRED to TRUE if an error is encountered
    # -----------------------------------------------------------------
    IF(NOT DEFINED ${_NAMEVAR}_ERROR_OCCURRED)
        SET(${_NAMEVAR}_ERROR_OCCURRED FALSE)
    ENDIF(NOT DEFINED ${_NAMEVAR}_ERROR_OCCURRED)

    # Check if headers are provided
    # -----------------------------
#    IF(${_NAMEVAR}_name_include)
#        FOREACH(_header ${${_NAMEVAR}_name_include})
#            IF(${_NAMEVAR}_name_include_suffix)
#                SET(CMAKE_REQUIRED_INCLUDES "${${_NAMEVAR}_INCLUDE_PATH};${${_NAMEVAR}_INCLUDE_PATH}/${${_NAMEVAR}_name_include_suffix}")
#            ELSE(${_NAMEVAR}_name_include_suffix)
#                SET(CMAKE_REQUIRED_INCLUDES "${${_NAMEVAR}_INCLUDE_PATH}")
#            ENDIF(${_NAMEVAR}_name_include_suffix)
#            MESSAGE(STATUS "CMAKE_FLAGS=${CMAKE_FLAGS}")
#            SET(CMAKE_REQUIRED_FLAGS ${${_NAMEVAR}_LDFLAGS})
#            STRING(TOUPPER "${_header}" _FILE)
#            STRING(REPLACE "." "_" _FILE "${_FILE}")
#            CHECK_INCLUDE_FILES(${_header} HAVE_${_FILE})
#            IF(NOT HAVE_${_FILE})
#                SET(${_NAMEVAR}_ERROR_OCCURRED TRUE)
#            ENDIF(NOT HAVE_${_FILE})
#        ENDFOREACH()
#    ENDIF(${_NAMEVAR}_name_include)

    # Check if library worked
    # -----------------------
    IF(${_NAMEVAR}_name_library)
        # Try to find the function symbol
        FOREACH(_type ${${_NAMEVAR}_type_library})
            # Check C funtion
            IF(${_type} MATCHES "C")
                SET(CMAKE_REQUIRED_INCLUDES ${${_NAMEVAR}_INCLUDE_PATH})
                SET(CMAKE_REQUIRED_LIBRARIES ${${_NAMEVAR}_LIBRARY})
                SET(CMAKE_REQUIRED_FLAGS ${${_NAMEVAR}_LDFLAGS})
                CHECK_FUNCTION_EXISTS(
                        ${${_NAMEVAR}_name_fct_test}
                        ${_NAMEVAR}_C_FOUND
                                     )
            ENDIF()

            # Check Fortran funtion
            IF(${_type} MATCHES "Fortran")
                SET(CMAKE_REQUIRED_LIBRARIES ${${_NAMEVAR}_LIBRARY})
                CHECK_FORTRAN_FUNCTION_EXISTS(
                        ${${_NAMEVAR}_name_fct_test}
                        ${_NAMEVAR}_F_FOUND
                                             )
            ENDIF()
        ENDFOREACH()

        # Provide an error if necessary
        IF(${_NAMEVAR}_C_FOUND OR ${_NAMEVAR}_F_FOUND)
            MESSAGE(STATUS "Looking for ${_NAMEVAR} - working")
        ELSE()
            SET(${_NAMEVAR}_ERROR_OCCURRED TRUE)
            MESSAGE(STATUS "Looking for ${_NAMEVAR} - not working")
        ENDIF()
    ENDIF(${_NAMEVAR}_name_library)

ENDMACRO(CHECK_PACKAGE)

###
### END checkPACKAGE.cmake
###
