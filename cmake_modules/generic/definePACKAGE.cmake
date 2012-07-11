###
#
# @file          : definePACKAGE.cmake
#
# @description   :
#
# @version       :
# @created by    : Cedric Castagnede
# @creation date : 09-03-2012
# @last modified : ven. 06 juil. 2012 21:03:53 CEST
#
###
#
# * Users variables to help MORSE dependencies system
#   - ${_NAMEVAR}_LIB     :
#   - ${_NAMEVAR}_INC     :
#   - ${_NAMEVAR}_DIR     :
#   - ${_NAMEVAR}_URL     :
#   - ${_NAMEVAR}_TARBALL :
#
# * Variables of control the policy about MORSE dependencies
#   - ${_NAMEVAR}_USE_LIB     :
#   - ${_NAMEVAR}_USE_SYSTEM  : 
#   - ${_NAMEVAR}_USE_TARBALL : 
#   - ${_NAMEVAR}_USE_WEB     :
#   - ${_NAMEVAR}_USE_AUTO    :
#
#
#
#
###

# Macro to define if you have to download or use a package
# --------------------------------------------------------
MACRO(DEFINE_PACKAGE _NAME _TYPE_DEP)

    STRING(TOUPPER "${_NAME}" _NAMEVAR)

    # Default options
    # ---------------
    IF(${_TYPE_DEP} MATCHES "suggests")
        SET(${_NAMEVAR}_TRYUSE OFF)
        SET(${_NAMEVAR}_REQUIRED OFF)

    ELSEIF(${_TYPE_DEP} MATCHES "recommands")
        SET(${_NAMEVAR}_TRYUSE ON)
        SET(${_NAMEVAR}_REQUIRED OFF)

    ELSEIF(${_TYPE_DEP} MATCHES "depends")
        SET(${_NAMEVAR}_TRYUSE ON)
        SET(${_NAMEVAR}_REQUIRED ON)

    ELSE()
        MESSAGE(FATAL_ERROR "DEV_ERROR - Macro DEFINE_PACKAGE")

    ENDIF()
    MARK_AS_ADVANCED(${_NAMEVAR}_TRYUSE ${_NAMEVAR}_REQUIRED)

    # Switch options according to user's desires
    # ------------------------------------------
    IF(NOT DEFINED MORSE_USE_${_NAMEVAR})
        SET(MORSE_USE_${_NAMEVAR} ""
            CACHE STRING "Enable/Disable ${_NAMEVAR} dependency (ON/OFF/<not-defined>)")
    ENDIF(NOT DEFINED MORSE_USE_${_NAMEVAR})

    IF("${MORSE_USE_${_NAMEVAR}}" MATCHES "OFF")
        SET(${_NAMEVAR}_TRYUSE OFF)
        SET(${_NAMEVAR}_REQUIRED OFF)
    ELSEIF(NOT "" MATCHES "${MORSE_USE_${_NAMEVAR}}")
        SET(${_NAMEVAR}_TRYUSE ON)
        SET(${_NAMEVAR}_REQUIRED ON)
    ENDIF()

    # Set policy about ${_NAMEVAR} (install/find/set/...)
    # ---------------------------------------------------

    # Set the policy if ***_LIB + ***_INC is defined
    IF(DEFINED ${_NAMEVAR}_LIB)
        OPTION(${_NAMEVAR}_USE_LIB "Enable/Disable to link with ${_NAMEVAR}" ON)
        OPTION(${_NAMEVAR}_USE_AUTO "Enable/Disable to install ${_NAMEVAR} if it was not found" OFF)
        SET(${_NAMEVAR}_TRYUSE ON)
        SET(${_NAMEVAR}_REQUIRED ON)
    ELSE()
        OPTION(${_NAMEVAR}_USE_LIB "Enable/Disable to link with ${_NAMEVAR}" OFF)
        OPTION(${_NAMEVAR}_USE_AUTO "Enable/Disable to install ${_NAMEVAR} if it was not found" ON)
    ENDIF()

    # Set the policy if ***_DIR is defined
    IF(DEFINED ${_NAMEVAR}_DIR)
        OPTION(${_NAMEVAR}_USE_SYSTEM "Enable/Disable to look for ${_NAMEVAR} installation in environment" ON)
        OPTION(${_NAMEVAR}_USE_AUTO "Enable/Disable to install ${_NAMEVAR} if it was not found" OFF)
        SET(${_NAMEVAR}_TRYUSE ON)
        SET(${_NAMEVAR}_REQUIRED ON)
    ELSE()
        OPTION(${_NAMEVAR}_USE_SYSTEM "Enable/Disable to look for ${_NAMEVAR} installation in environment" OFF)
        OPTION(${_NAMEVAR}_USE_AUTO "Enable/Disable to install ${_NAMEVAR} if it was not found" ON)
    ENDIF()

    # Set the policy if ***_URL is defined
    IF(DEFINED ${_NAMEVAR}_URL)
        OPTION(${_NAMEVAR}_USE_WEB "Enable/Disable to install ${_NAMEVAR} form ${${_NAMEVAR}_URL}" ON)
        OPTION(${_NAMEVAR}_USE_AUTO "Enable/Disable to install ${_NAMEVAR} if it was not found" OFF)
        SET(${_NAMEVAR}_TRYUSE ON)
        SET(${_NAMEVAR}_REQUIRED ON)
    ELSE()
        OPTION(${_NAMEVAR}_USE_WEB "Enable/Disable to install ${_NAMEVAR} form ${${_NAMEVAR}_URL}" OFF)
        OPTION(${_NAMEVAR}_USE_AUTO "Enable/Disable to install ${_NAMEVAR} if it was not found" ON)
    ENDIF()

    # Set the policy if ***_TARBALL is defined
    IF(DEFINED ${_NAMEVAR}_TARBALL)
        OPTION(${_NAMEVAR}_USE_TARBALL "Enable/Disable to install ${_NAMEVAR} form ${CMAKE_SOURCE_DIR}/externals/${${_NAMEVAR}_TARBALL}" ON)
        OPTION(${_NAMEVAR}_USE_AUTO "Enable/Disable to install ${_NAMEVAR} if it was not found" OFF)
        SET(${_NAMEVAR}_TRYUSE ON)
        SET(${_NAMEVAR}_REQUIRED ON)
    ELSE()
        OPTION(${_NAMEVAR}_USE_TARBALL "Enable/Disable to install ${_NAMEVAR} form ${CMAKE_SOURCE_DIR}/externals/${${_NAMEVAR}_TARBALL}" OFF)
        OPTION(${_NAMEVAR}_USE_AUTO "Enable/Disable to install ${_NAMEVAR} if it was not found" ON)
    ENDIF()

    # Others policy - experimental tricks
    OPTION(${_NAMEVAR}_USE_SVN "Enable/Disable to install ${_NAMEVAR} form the repository" OFF)

    # Force installation for eigen
    # ----------------------------
    IF("${_NAMEVAR}" MATCHES "BLAS")
        STRING(TOUPPER "${MORSE_USE_BLAS}" VALUE_MORSE_USE_BLAS)
        IF("${VALUE_MORSE_USE_BLAS}" MATCHES "EIGEN" OR
           "${VALUE_MORSE_USE_BLAS}" MATCHES "REFBLAS")
            SET(BLAS_TRYUSE      ON )
            SET(BLAS_REQUIRED    ON )
            SET(BLAS_USE_LIB     OFF)
            SET(BLAS_USE_SYSTEM  OFF)
            SET(BLAS_USE_TARBALL ON )
            SET(BLAS_USE_WEB     ON )
            SET(BLAS_USE_AUTO    OFF)
        ENDIF()
    ENDIF()

    # Print debug message
    # -------------------
    DEBUG_DEFINE_OUTPUT(${_NAMEVAR})

    # Process to define if you have to download or use a package
    # ----------------------------------------------------------
    UNSET(HAVE_${_NAMEVAR})
    UNSET(${_NAMEVAR}_BUILD_MODE)
    IF(${${_NAMEVAR}_TRYUSE})

        # Define manually processing
        IF(${_NAMEVAR}_USE_LIB)
            INCLUDE(setPACKAGE)
            SET_PACKAGE(${_NAMEVAR})

        # Find in the system processing
        ELSEIF(${_NAMEVAR}_USE_SYSTEM)
            #
            # TODO: SET REQUIRED and search only in system path only
            #
            #IF(${${_NAMEVAR}_REQUIRED})
            #    FIND_PACKAGE(${_NAMEVAR} REQUIRED)
            #ELSE(${${_NAMEVAR}_REQUIRED})
            #    FIND_PACKAGE(${_NAMEVAR} QUIET)
            #ENDIF(${${_NAMEVAR}_REQUIRED})
            FIND_PACKAGE(${_NAMEVAR})
            IF(${_NAMEVAR}_FOUND)
                SET(HAVE_${_NAMEVAR} ON)
                LINK_DIRECTORIES(${${_NAMEVAR}_LIBRARY_PATH})
                INCLUDE_DIRECTORIES(${${_NAMEVAR}_INCLUDE_PATH})
                LIST(INSERT EXTRA_LIBS 0 ${${_NAMEVAR}_LIBRARIES})
            ELSE(${_NAMEVAR}_FOUND)
                MESSAGE(FATAL_ERROR "Looking for ${_NAMEVAR} - not found")
            ENDIF()

        # Install form tarball processing
        ELSEIF(${_NAMEVAR}_USE_TARBALL)
            INCLUDE(installPACKAGE)
            INSTALL_PACKAGE(${_NAMEVAR} "TARBALL")
            SET(HAVE_${_NAMEVAR} ON)
            LINK_DIRECTORIES(${${_NAMEVAR}_LIBRARY_PATH})
            INCLUDE_DIRECTORIES(${${_NAMEVAR}_INCLUDE_PATH})
            LIST(INSERT EXTRA_LIBS 0 ${${_NAMEVAR}_LIBRARIES})

        # Install form web processing
        ELSEIF(${_NAMEVAR}_USE_WEB)
            INCLUDE(installPACKAGE)
            INSTALL_PACKAGE(${_NAMEVAR} "WEB")
            SET(HAVE_${_NAMEVAR} ON)
            LINK_DIRECTORIES(${${_NAMEVAR}_LIBRARY_PATH})
            INCLUDE_DIRECTORIES(${${_NAMEVAR}_INCLUDE_PATH})
            LIST(INSERT EXTRA_LIBS 0 ${${_NAMEVAR}_LIBRARIES})

        # Install form repository processing
        ELSEIF(${_NAMEVAR}_USE_SVN)
            INCLUDE(installPACKAGE)
            INSTALL_PACKAGE(${_NAMEVAR} "REPO")
            SET(HAVE_${_NAMEVAR} ON)
            LINK_DIRECTORIES(${${_NAMEVAR}_LIBRARY_PATH})
            INCLUDE_DIRECTORIES(${${_NAMEVAR}_INCLUDE_PATH})
            LIST(INSERT EXTRA_LIBS 0 ${${_NAMEVAR}_LIBRARIES})

        # Auot-install processing
        ELSEIF(${_NAMEVAR}_USE_AUTO)
            #
            # TODO: SET REQUIRED and search only in system path only
            #
            #IF(${${_NAMEVAR}_REQUIRED})
            #    FIND_PACKAGE(${_NAMEVAR} REQUIRED)
            #ELSE(${${_NAMEVAR}_REQUIRED})
            #    FIND_PACKAGE(${_NAMEVAR} QUIET)
            #ENDIF(${${_NAMEVAR}_REQUIRED})
            FIND_PACKAGE(${_NAMEVAR})
            IF(${_NAMEVAR}_FOUND)
                SET(HAVE_${_NAMEVAR} ON)
                LINK_DIRECTORIES(${${_NAMEVAR}_LIBRARY_PATH})
                INCLUDE_DIRECTORIES(${${_NAMEVAR}_INCLUDE_PATH})
                LIST(INSERT EXTRA_LIBS 0 ${${_NAMEVAR}_LIBRARIES})

            ELSE(${_NAMEVAR}_FOUND)
                IF(${${_NAMEVAR}_REQUIRED})
                    INCLUDE(installPACKAGE)
                    INSTALL_PACKAGE(${_NAMEVAR} "AUTO")
                    SET(HAVE_${_NAMEVAR} ON)
                    LINK_DIRECTORIES(${${_NAMEVAR}_LIBRARY_PATH})
                    INCLUDE_DIRECTORIES(${${_NAMEVAR}_INCLUDE_PATH})
                    LIST(INSERT EXTRA_LIBS 0 ${${_NAMEVAR}_LIBRARIES})

                ELSE(${${_NAMEVAR}_REQUIRED})
                    MESSAGE(STATUS "Looking for ${_NAMEVAR} - not found")
                    MESSAGE(STATUS "Installing ${_NAMEVAR} - not necessary")

                ENDIF(${${_NAMEVAR}_REQUIRED})
            ENDIF()

        ELSE()
            MESSAGE(STATUS "Looking for ${_NAMEVAR} - not found")
            MESSAGE(STATUS "Installing ${_NAMEVAR} - no")

        ENDIF()

    ELSE(${${_NAMEVAR}_TRYUSE})
        MESSAGE(STATUS "Looking for ${_NAMEVAR} - no")
        MESSAGE(STATUS "Installing ${_NAMEVAR} - no")

    ENDIF(${${_NAMEVAR}_TRYUSE})

ENDMACRO(DEFINE_PACKAGE)

###
#
#
#
###
MACRO(DEBUG_DEFINE_OUTPUT _NAMEVAR)
    # Status of internal variables
    # ----------------------------
    IF(MORSE_DEBUG_CMAKE)
        MESSAGE(STATUS "Status of ${_NAMEVAR}:")
        MESSAGE(STATUS "  --> MORSE_USE_${_NAMEVAR}       : ${MORSE_USE_${_NAMEVAR}}")
        MESSAGE(STATUS "  --> Internal dependency management behavior:")
        MESSAGE(STATUS "        - ${_NAMEVAR}_TRYUSE      : ${${_NAMEVAR}_TRYUSE}")
        MESSAGE(STATUS "        - ${_NAMEVAR}_REQUIRED    : ${${_NAMEVAR}_REQUIRED}")
        MESSAGE(STATUS "        - ${_NAMEVAR}_USE_LIB     : ${${_NAMEVAR}_USE_LIB}")
        MESSAGE(STATUS "        - ${_NAMEVAR}_USE_SYSTEM  : ${${_NAMEVAR}_USE_SYSTEM}")
        MESSAGE(STATUS "        - ${_NAMEVAR}_USE_TARBALL : ${${_NAMEVAR}_USE_TARBALL}")
        MESSAGE(STATUS "        - ${_NAMEVAR}_USE_WEB     : ${${_NAMEVAR}_USE_WEB}")
        MESSAGE(STATUS "        - ${_NAMEVAR}_USE_AUTO    : ${${_NAMEVAR}_USE_AUTO}")
        MESSAGE(STATUS "  --> Given user's values:")
        MESSAGE(STATUS "        - ${_NAMEVAR}_LIB         : ${${_NAMEVAR}_LIB}")
        MESSAGE(STATUS "        - ${_NAMEVAR}_INC         : ${${_NAMEVAR}_INC}")
        MESSAGE(STATUS "        - ${_NAMEVAR}_DIR         : ${${_NAMEVAR}_DIR}")
        MESSAGE(STATUS "        - ${_NAMEVAR}_URL         : ${${_NAMEVAR}_URL}")
        MESSAGE(STATUS "        - ${_NAMEVAR}_TARBALL     : ${${_NAMEVAR}_TARBALL}")

    ENDIF(MORSE_DEBUG_CMAKE)

ENDMACRO(DEBUG_DEFINE_OUTPUT)

###
### END definePACKAGE.cmake
###
