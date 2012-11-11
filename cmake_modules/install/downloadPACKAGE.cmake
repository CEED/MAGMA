###
#
#  @file downloadPACKAGE.cmake
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

# Macro to download tarball
# -------------------------
MACRO(DOWNLOAD_PACKAGE _NAME)

    STRING(TOUPPER "${_NAME}" _NAMEVAR)
    MESSAGE(STATUS "Download: ${${_NAMEVAR}_URL}")
    IF(DEFINED ${_NAMEVAR}_MD5SUM)
        FILE(DOWNLOAD ${${_NAMEVAR}_URL}
             ${CMAKE_SOURCE_DIR}/externals/${${_NAMEVAR}_USED_FILE}
             EXPECTED_MD5 ${${_NAMEVAR}_MD5SUM}
             STATUS IS_GOT
             SHOW_PROGRESS
             )
    ELSE()
        MESSAGE(STATUS "Warning: ${_NAMEVAR} checksum is not performed")
        FILE(DOWNLOAD ${${_NAMEVAR}_URL}
             ${CMAKE_SOURCE_DIR}/externals/${${_NAMEVAR}_USED_FILE}
             STATUS IS_GOT
             SHOW_PROGRESS
             )
    ENDIF()
    LIST(GET IS_GOT 0 IS_GOT_CODE)
    LIST(GET IS_GOT 1 IS_GOT_MSG)
    IF(IS_GOT_CODE)
        MESSAGE(FATAL_ERROR "ERROR: ${_NAMEVAR} tarball -- not downloaded")
    ENDIF(IS_GOT_CODE)
    MESSAGE(STATUS "Download: ${${_NAMEVAR}_URL} - Done")

ENDMACRO(DOWNLOAD_PACKAGE)

# Macro to download tarball (alternative methode)
# -----------------------------------------------
MACRO(DOWNLOAD_PACKAGE_ALTERNATIVE _NAME)

    STRING(TOUPPER "${_NAME}" _NAMEVAR)

    IF(WIN32)
        DOWNLOAD_PACKAGE("${_NAMEVAR}")
    ENDIF(WIN32)

    IF(APPLE)
        MESSAGE(STATUS "Download: ${${_NAMEVAR}_URL}")
        EXECUTE_PROCESS(
            COMMAND curl ${${_NAMEVAR}_URL} -o ${${_NAMEVAR}_USED_FILE}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/externals/
            )
        MESSAGE(STATUS "Download: ${${_NAMEVAR}_URL} - Done")

    ELSE(APPLE)
        MESSAGE(STATUS "Download: ${${_NAMEVAR}_URL}")
        EXECUTE_PROCESS(
            COMMAND wget ${${_NAMEVAR}_URL} -O ${${_NAMEVAR}_USED_FILE}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/externals/
            )
        MESSAGE(STATUS "Download: ${${_NAMEVAR}_URL} - Done")

    ENDIF(APPLE)

ENDMACRO(DOWNLOAD_PACKAGE_ALTERNATIVE)

# Macro to know if you have to download tarball
# ---------------------------------------------
MACRO(DEFINE_DOWNLOAD_PACKAGE _NAME _MODE)

    # Define name
    # -----------
    STRING(TOUPPER "${_NAME}" _NAMEVAR)

    # Installation with a tarball in ${CMAKE_SOURCE_DIR}/externals
    # ------------------------------------------------------------
    IF(${_MODE} MATCHES "TARBALL")
        # Message
        MESSAGE(STATUS "Installing ${_NAMEVAR} - mode TARBALL started")

        # Check if tarball exists 
        IF(NOT EXISTS ${CMAKE_SOURCE_DIR}/externals/${${_NAMEVAR}_TARBALL})
            MESSAGE(FATAL_ERROR "Installing ${_NAMEVAR} - ${CMAKE_SOURCE_DIR}/externals/${${_NAMEVAR}_TARBALL} not found")
        ENDIF()

        # Check md5sum
        IF(DEFINED ${_NAMEVAR}_MD5SUM)
            EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E md5sum ${CMAKE_SOURCE_DIR}/externals/${${_NAMEVAR}_TARBALL}
                            OUTPUT_VARIABLE ${_NAMEVAR}_TARBALL_MD5_FOUND
                            OUTPUT_STRIP_TRAILING_WHITESPACE
                           )
            IF(NOT "${${_NAMEVAR}_TARBALL_MD5_FOUND}" MATCHES "${${_NAMEVAR}_MD5SUM}")
                MESSAGE(FATAL_ERROR "Installing ${_NAMEVAR} - ${${_NAMEVAR}_TARBALL} checksum failed")
            ELSE()
                MESSAGE(STATUS "Installing ${_NAMEVAR} - ${${_NAMEVAR}_TARBALL} checksum succeed")
            ENDIF()
        ELSE()
            MESSAGE(STATUS "Warning: ${${_NAMEVAR}_TARBALL} checksum is not performed")
        ENDIF()

        # Define installation mode in cmake env
        SET(${_NAMEVAR}_USED_MODE "TARBALL"               )
        SET(${_NAMEVAR}_USED_FILE "${${_NAMEVAR}_TARBALL}")

    # Installation with a tarball form the web
    # ----------------------------------------
    ELSEIF(${_MODE} MATCHES "WEB")

        # Message
        MESSAGE(STATUS "Installing ${_NAMEVAR} - mode WEB started")

        # Remove tarball if it already exists
        IF(EXISTS ${CMAKE_SOURCE_DIR}/externals/${${_NAMEVAR}_TARBALL})
            FILE(REMOVE ${CMAKE_SOURCE_DIR}/externals/${${_NAMEVAR}_TARBALL})
        ENDIF()

        # Download tarball and check md5sum
        STRING(REGEX REPLACE ".*/(.*)$" "\\1" ${_NAMEVAR}_USED_FILE "${${_NAMEVAR}_URL}")
        DOWNLOAD_PACKAGE(${_NAMEVAR})

        # Define installation mode in cmake env
        SET(${_NAMEVAR}_USED_MODE "WEB")

    # Installation auto
    # -----------------
    ELSEIF(${_MODE} MATCHES "AUTO")

        # Check if tarball exists
        IF(NOT EXISTS ${CMAKE_SOURCE_DIR}/externals/${${_NAMEVAR}_TARBALL})
            # Message
            MESSAGE(STATUS "Installing ${_NAMEVAR} - mode WEB started")

            # Download tarball and check md5sum
            STRING(REGEX REPLACE ".*/(.*)$" "\\1" ${_NAMEVAR}_USED_FILE "${${_NAMEVAR}_URL}")
            DOWNLOAD_PACKAGE(${_NAMEVAR})

            # Define installation mode in cmake env
            SET(${_NAMEVAR}_USED_MODE "WEB")

        ELSE()
            # Check md5sum
            IF(DEFINED ${_NAMEVAR}_MD5SUM)
                EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E md5sum ${CMAKE_SOURCE_DIR}/externals/${${_NAMEVAR}_TARBALL}
                                OUTPUT_VARIABLE ${_NAMEVAR}_TARBALL_MD5_FOUND
                                OUTPUT_STRIP_TRAILING_WHITESPACE
                               )
                IF(NOT "${${_NAMEVAR}_TARBALL_MD5_FOUND}" MATCHES "${${_NAMEVAR}_MD5SUM}")
                    # Message
                    MESSAGE(STATUS "Installing ${_NAMEVAR} - mode WEB started")

                    # Remove old tarball (md5sum failed)
                    EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_SOURCE_DIR}/externals/${${_NAMEVAR}_TARBALL})

                    # Download tarball and check md5sum
                    STRING(REGEX REPLACE ".*/(.*)$" "\\1" ${_NAMEVAR}_USED_FILE "${${_NAMEVAR}_URL}")
                    DOWNLOAD_PACKAGE(${_NAMEVAR})

                    # Define installation mode in cmake env
                    SET(${_NAMEVAR}_USED_MODE "WEB")

                ELSE()
                    MESSAGE(STATUS "Installing ${_NAMEVAR} - mode TARBALL started")
                    MESSAGE(STATUS "Installing ${_NAMEVAR} - ${${_NAMEVAR}_TARBALL} checksum succeed")

                ENDIF()
            ELSE()
                MESSAGE(STATUS "Installing ${_NAMEVAR} - mode TARBALL started")
                MESSAGE(STATUS "Warning: ${${_NAMEVAR}_TARBALL} checksum is not performed")

            ENDIF()

            # Define installation mode in cmake env
            SET(${_NAMEVAR}_USED_MODE "TARBALL")
            SET(${_NAMEVAR}_USED_FILE "${${_NAMEVAR}_TARBALL}")
        ENDIF()

    # Installation form a repository
    # ------------------------------
    ELSEIF(${_MODE} MATCHES "REPO")
        # Define installation mode in cmake env
        SET(${_NAMEVAR}_USED_MODE "${${_NAMEVAR}_REPO_MODE}")

    ELSE()
        MESSAGE(FATAL_ERROR "DEV_ERROR - Macro DEFINE_DOWNLOAD_PACKAGE")

    ENDIF()

ENDMACRO(DEFINE_DOWNLOAD_PACKAGE)

##
## @end file downloadPACKAGE.cmake
##
