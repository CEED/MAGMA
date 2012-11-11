###
#
#  @file DefineDependencies.cmake
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
# * Users variables to help MORSE dependencies system
#   - ${_UPPERWANTED}_LIB     :
#   - ${_UPPERWANTED}_INC     :
#   - ${_UPPERWANTED}_DIR     :
#   - ${_UPPERWANTED}_URL     :
#   - ${_UPPERWANTED}_TARBALL :
#
# * Variables of control the policy about MORSE dependencies
#   - ${_UPPERWANTED}_USE_LIB     :
#   - ${_UPPERWANTED}_USE_SYSTEM  : 
#   - ${_UPPERWANTED}_USE_TARBALL : 
#   - ${_UPPERWANTED}_USE_WEB     :
#   - ${_UPPERWANTED}_USE_AUTO    :
#
###

INCLUDE(populatePACKAGE)
INCLUDE(SetDependencies)

# Macro to define if you have to download or use a package
# --------------------------------------------------------
MACRO(DEFINE_PACKAGE _APPLICANT _WANTED _TYPE_DEP)

    # Define input variable to uppercase
    # ----------------------------------
    STRING(TOUPPER "${_WANTED}" _UPPERWANTED)
    STRING(TOLOWER "${_WANTED}" _LOWERWANTED)

    # Default options
    # ---------------
    IF(${_TYPE_DEP} MATCHES "suggests")
        SET(${_UPPERWANTED}_TRYUSE OFF)
        SET(${_UPPERWANTED}_REQUIRED OFF)

    ELSEIF(${_TYPE_DEP} MATCHES "recommends")
        SET(${_UPPERWANTED}_TRYUSE ON)
        SET(${_UPPERWANTED}_REQUIRED OFF)

    ELSEIF(${_TYPE_DEP} MATCHES "depends")
        SET(${_UPPERWANTED}_TRYUSE ON)
        SET(${_UPPERWANTED}_REQUIRED ON)

    ELSE()
        MESSAGE(FATAL_ERROR "DEV_ERROR - Macro DEFINE_PACKAGE - ${_UPPERWANTED} as ${_TYPE_DEP}")

    ENDIF()
    MARK_AS_ADVANCED(${_UPPERWANTED}_TRYUSE ${_UPPERWANTED}_REQUIRED)


    # Process to define if you have to download or use a package
    # ----------------------------------------------------------
    UNSET(HAVE_${_UPPERWANTED})
    IF(${${_UPPERWANTED}_TRYUSE})

        # Define manually processing
        IF(${_UPPERWANTED}_USE_LIB)
            SET_PACKAGE(${_UPPERWANTED})

        # Find in the system processing
        ELSEIF(${_UPPERWANTED}_USE_SYSTEM)
            IF("${_TYPE_DEP}" STREQUAL "depends")
                MESSAGE(STATUS "Looking for ${_UPPERWANTED} - dependency required by ${_APPLICANT}")
                FIND_PACKAGE(${_UPPERWANTED} REQUIRED)
                IF(${_UPPERWANTED}_FOUND)
                    POPULATE_COMPILE_SYSTEM(${_UPPERWANTED})
                ENDIF()

            ELSE()
                MESSAGE(STATUS "Looking for ${_UPPERWANTED} - dependency optionally asked by ${_APPLICANT}")
                FIND_PACKAGE(${_UPPERWANTED} QUIET)
                IF(${_UPPERWANTED}_FOUND)
                    POPULATE_COMPILE_SYSTEM(${_UPPERWANTED})
                ENDIF()

            ENDIF()
            IF(NOT ${_UPPERWANTED}_FOUND)
                MESSAGE(FATAL_ERROR "Looking for ${_UPPERWANTED} - not found")
            ENDIF()

        # Install form tarball processing
        ELSEIF(${_UPPERWANTED}_USE_TARBALL)
            INCLUDE(installPACKAGE)
            INSTALL_PACKAGE(${_UPPERWANTED} "TARBALL")

        # Install form web processing
        ELSEIF(${_UPPERWANTED}_USE_WEB)
            INCLUDE(installPACKAGE)
            INSTALL_PACKAGE(${_UPPERWANTED} "WEB")

        # Install form repository processing
        ELSEIF(${_UPPERWANTED}_USE_SVN)
            INCLUDE(installPACKAGE)
            INSTALL_PACKAGE(${_UPPERWANTED} "REPO")

        # Auot-install processing
        ELSEIF(${_UPPERWANTED}_USE_AUTO)
            IF("${_TYPE_DEP}" STREQUAL "depends")
                MESSAGE(STATUS "Looking for ${_UPPERWANTED} - dependency required by ${_APPLICANT}")
                FIND_PACKAGE(${_UPPERWANTED} REQUIRED)
                IF(${_UPPERWANTED}_FOUND)
                    POPULATE_COMPILE_SYSTEM(${_UPPERWANTED})
                ENDIF()

            ELSE()
                MESSAGE(STATUS "Looking for ${_UPPERWANTED} - dependency optionally asked by ${_APPLICANT}")
                FIND_PACKAGE(${_UPPERWANTED} QUIET)
                IF(${_UPPERWANTED}_FOUND)
                    POPULATE_COMPILE_SYSTEM(${_UPPERWANTED})
                ENDIF()

            ENDIF()
            IF(NOT ${_UPPERWANTED}_FOUND)
                IF(${${_UPPERWANTED}_REQUIRED})
                    INCLUDE(installPACKAGE)
                    INSTALL_PACKAGE(${_UPPERWANTED} "AUTO")

                ELSE(${${_UPPERWANTED}_REQUIRED})
                    MESSAGE(STATUS "Looking for ${_UPPERWANTED} - not found")
                    MESSAGE(STATUS "Installing ${_UPPERWANTED} - not necessary")

                ENDIF(${${_UPPERWANTED}_REQUIRED})
            ENDIF()

        ELSE()
            MESSAGE(STATUS "Looking for ${_UPPERWANTED} - not found")
            MESSAGE(STATUS "Installing ${_UPPERWANTED} - no")

        ENDIF()

    ELSE(${${_UPPERWANTED}_TRYUSE})
        MESSAGE(STATUS "Looking for ${_UPPERWANTED} - no")
        MESSAGE(STATUS "Installing ${_UPPERWANTED} - no")

    ENDIF(${${_UPPERWANTED}_TRYUSE})

ENDMACRO(DEFINE_PACKAGE)

##
## @end file DefineDependencies.cmake
##
