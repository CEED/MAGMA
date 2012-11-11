###
#
#  @file installExternalPACKAGE.cmake
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
INCLUDE(ExternalProject)

MACRO(INSTALL_EXTERNAL_PACKAGE _NAME _MODE)

    # Define input variable to uppercase
    # ----------------------------------
    STRING(TOUPPER "${_NAME}" _NAMEVAR)
    STRING(TOLOWER "${_NAME}" _LOWNAME)
    SET(${_NAMEVAR}_DOWNLOAD_PATH ${CMAKE_SOURCE_DIR}/externals)

    # Create a target distclean to contains all clean target create from externalProject
    # ----------------------------------------------------------------------------------
    IF(NOT TARGET clean_all)
        ADD_CUSTOM_TARGET(clean_all
                          COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                                                   --target clean)
    ENDIF(NOT TARGET clean_all)
    IF(NOT TARGET clean_dependencies)
        ADD_CUSTOM_TARGET(clean_dependencies)
    ENDIF(NOT TARGET clean_dependencies)

    # Define dummy target for test if test target was not defined by info
    # -------------------------------------------------------------------
    SET(${_NAMEVAR}_DUMMY_CMD ${CMAKE_COMMAND} -E echo)

    # Create the main step of this externalProject
    # --------------------------------------------
    IF(NOT DEFINED ${_NAMEVAR}_EP)

        # Create a directory to store all stamp for externalProject
        # ---------------------------------------------------------
        SET(${_NAMEVAR}_STAMP_DIR "${CMAKE_BINARY_DIR}/externals_stamp/${_LOWNAME}")
        IF(NOT EXISTS ${${_NAMEVAR}_STAMP_DIR})
            EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E make_directory ${${_NAMEVAR}_STAMP_DIR})
        ENDIF()
        SET_DIRECTORY_PROPERTIES(PROPERTIES CLEAN_NO_CUSTOM ${${_NAMEVAR}_STAMP_DIR})
        ADD_CUSTOM_TARGET(${_LOWNAME}_clean
                          COMMAND ${CMAKE_COMMAND} -E copy ${_LOWNAME}/${_LOWNAME}_build-urlinfo.txt .
                          COMMAND ${CMAKE_COMMAND} -E copy ${_LOWNAME}/extract-${_LOWNAME}_build.cmake .
                          COMMAND ${CMAKE_COMMAND} -E copy ${_LOWNAME}/verify-${_LOWNAME}_build.cmake .
                          COMMAND ${CMAKE_COMMAND} -E remove_directory ${_LOWNAME}
                          COMMAND ${CMAKE_COMMAND} -E remove_directory ${${_NAMEVAR}_SOURCE_PATH}
                          COMMAND ${CMAKE_COMMAND} -E make_directory ${_LOWNAME}
                          COMMAND ${CMAKE_COMMAND} -E copy ${_LOWNAME}_build-urlinfo.txt ${_LOWNAME}
                          COMMAND ${CMAKE_COMMAND} -E copy extract-${_LOWNAME}_build.cmake ${_LOWNAME}
                          COMMAND ${CMAKE_COMMAND} -E copy verify-${_LOWNAME}_build.cmake ${_LOWNAME}
                          COMMAND ${CMAKE_COMMAND} -E remove ${_LOWNAME}_build-urlinfo.txt
                          COMMAND ${CMAKE_COMMAND} -E remove extract-${_LOWNAME}_build.cmake
                          COMMAND ${CMAKE_COMMAND} -E remove verify-${_LOWNAME}_build.cmake
                          WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/externals_stamp)
        ADD_DEPENDENCIES(clean_all ${_LOWNAME}_clean)
        ADD_DEPENDENCIES(clean_dependencies ${_LOWNAME}_clean)

        # Looking for installation mode
        # -----------------------------
        IF("${_MODE}" MATCHES "TARBALL" OR "${_MODE}" MATCHES "WEB")
            SET(${_NAMEVAR}_GETSOURCE ${CMAKE_SOURCE_DIR}/externals/${${_NAMEVAR}_USED_FILE})
            IF(DEFINED ${_NAMEVAR}_MD5SUM)
                ExternalProject_Add(${_LOWNAME}_build
                        DEPENDS           ${${_NAMEVAR}_MAKEDEP}
                        STAMP_DIR         ${${_NAMEVAR}_STAMP_DIR}
                        DOWNLOAD_DIR      ${${_NAMEVAR}_DOWNLOAD_PATH}
                        PATCH_COMMAND     ${${_NAMEVAR}_PATCH_CMD}
                        SOURCE_DIR        ${${_NAMEVAR}_SOURCE_PATH}
                        BINARY_DIR        ${${_NAMEVAR}_BUILD_PATH}
                        INSTALL_DIR       ${${_NAMEVAR}_PATH}
                        URL               ${${_NAMEVAR}_GETSOURCE}
                        URL_MD5           ${${_NAMEVAR}_MD5SUM}
                        CONFIGURE_COMMAND ${${_NAMEVAR}_CONFIG_CMD} ${${_NAMEVAR}_CONFIG_OPTS} 
                        BUILD_COMMAND     ${${_NAMEVAR}_BUILD_CMD} 
                        INSTALL_COMMAND   ${${_NAMEVAR}_DUMMY_CMD}
                        TEST_COMMAND      ${${_NAMEVAR}_DUMMY_CMD}
                        TEST_AFTER_INSTALL 1
                        )
            ELSE()
                ExternalProject_Add(${_LOWNAME}_build
                        DEPENDS           ${${_NAMEVAR}_MAKEDEP}
                        DOWNLOAD_DIR      ${${_NAMEVAR}_DOWNLOAD_PATH}
                        PATCH_COMMAND     ${${_NAMEVAR}_PATCH_CMD}
                        SOURCE_DIR        ${${_NAMEVAR}_SOURCE_PATH}
                        BINARY_DIR        ${${_NAMEVAR}_BUILD_PATH}
                        INSTALL_DIR       ${${_NAMEVAR}_PATH}
                        URL               ${${_NAMEVAR}_GETSOURCE}
                        CONFIGURE_COMMAND ${${_NAMEVAR}_CONFIG_CMD} ${${_NAMEVAR}_CONFIG_OPTS} 
                        BUILD_COMMAND     ${${_NAMEVAR}_BUILD_CMD} 
                        INSTALL_COMMAND   ${${_NAMEVAR}_DUMMY_CMD}
                        TEST_COMMAND      ${${_NAMEVAR}_DUMMY_CMD}
                        TEST_AFTER_INSTALL 1
                        )
            ENDIF()
            MESSAGE(STATUS "Installing ${_NAMEVAR} - mode ${_MODE} done")
    
        ELSEIF("${_MODE}" MATCHES "SVN")
            SET(${_NAMEVAR}_GETSOURCE ${${_NAMEVAR}_REPO_URL})
            ExternalProject_Add(${_LOWNAME}_build
                    DEPENDS           ${${_NAMEVAR}_MAKEDEP}
                    DOWNLOAD_DIR      ${${_NAMEVAR}_DOWNLOAD_PATH}
                    PATCH_COMMAND     ${${_NAMEVAR}_PATCH_CMD}
                    SOURCE_DIR        ${${_NAMEVAR}_SOURCE_PATH}
                    BINARY_DIR        ${${_NAMEVAR}_BUILD_PATH}
                    INSTALL_DIR       ${${_NAMEVAR}_PATH}
                    REPO_URLOSITORY   ${${_NAMEVAR}_GETSOURCE}
                    SVN_USERNAME      ${${_NAMEVAR}_REPO_ID}
                    SVN_PASSWORD      ${${_NAMEVAR}_REPO_PWD}
                    CONFIGURE_COMMAND ${${_NAMEVAR}_CONFIG_CMD} ${${_NAMEVAR}_CONFIG_OPTS} 
                    BUILD_COMMAND     ${${_NAMEVAR}_BUILD_CMD}
                    INSTALL_COMMAND   ${${_NAMEVAR}_DUMMY_CMD}
                    TEST_COMMAND      ${${_NAMEVAR}_DUMMY_CMD}
                    TEST_AFTER_INSTALL 1
                    )
            MESSAGE(STATUS "Installing ${_NAMEVAR} - mode ${_MODE} done")
    
        ELSEIF("${_MODE}" MATCHES "CVS")
            SET(${_NAMEVAR}_GETSOURCE ${${_NAMEVAR}_CVS_REP})
            ExternalProject_Add(${_LOWNAME}_build
                    DEPENDS           ${${_NAMEVAR}_MAKEDEP}
                    DOWNLOAD_DIR      ${${_NAMEVAR}_DOWNLOAD_PATH}
                    PATCH_COMMAND     ${${_NAMEVAR}_PATCH_CMD}
                    SOURCE_DIR        ${${_NAMEVAR}_SOURCE_PATH}
                    BINARY_DIR        ${${_NAMEVAR}_BUILD_PATH}
                    INSTALL_DIR       ${${_NAMEVAR}_PATH}
                    CVS_REPOSITORY    ${${_NAMEVAR}_GETSOURCE}
                    CVS_MODULE        ${${_NAMEVAR}_CVS_MOD}
                    CONFIGURE_COMMAND ${${_NAMEVAR}_CONFIG_CMD} ${${_NAMEVAR}_CONFIG_OPTS} 
                    BUILD_COMMAND     ${${_NAMEVAR}_BUILD_CMD}
                    INSTALL_COMMAND   ${${_NAMEVAR}_DUMMY_CMD}
                    TEST_COMMAND      ${${_NAMEVAR}_DUMMY_CMD}
                    TEST_AFTER_INSTALL 1
                    )
            MESSAGE(STATUS "Installing ${_NAMEVAR} - mode ${_MODE} done")
    
        ELSE()
            MESSAGE(FATAL_ERROR "DEV_ERROR - Macro INSTALL_EXTERNAL_PACKAGE - <${_NAMEVAR}|${_MODE}>")
    
        ENDIF()

        # Define additional step of the build phase
        # ------------------------------------------
        IF(DEFINED ${_NAMEVAR}_ADD_BUILD_STEP)
            FOREACH(_STEP ${${_NAMEVAR}_ADD_BUILD_STEP})
                ExternalProject_Add_Step(${_NAME}_build ${_STEP}
                         COMMAND ${${_STEP}_CMD}
                         WORKING_DIRECTORY ${${_STEP}_DIR}
                         DEPENDEES ${${_STEP}_DEP_BEFORE}
                         DEPENDERS ${${_STEP}_DEP_AFTER})
            ENDFOREACH()
        ENDIF()

        # Add install step of external porject during install phase 
        # ---------------------------------------------------------
        INSTALL(CODE "MESSAGE(STATUS \"Installation of ${_LOWNAME}\")")
        IF(MORSE_DEBUG_CMAKE)
            INSTALL(CODE "EXECUTE_PROCESS(COMMAND ${${_NAMEVAR}_INSTALL_CMD} WORKING_DIRECTORY ${${_NAMEVAR}_BUILD_PATH})")
        ELSE(MORSE_DEBUG_CMAKE)
            INSTALL(CODE "EXECUTE_PROCESS(COMMAND ${${_NAMEVAR}_INSTALL_CMD} WORKING_DIRECTORY ${${_NAMEVAR}_BUILD_PATH} OUTPUT_QUIET ERROR_QUIET)")
        ENDIF(MORSE_DEBUG_CMAKE)

        # Define additional step of the install phase
        # -------------------------------------------
        IF(DEFINED ${_NAMEVAR}_ADD_INSTALL_STEP)
            FOREACH(_STEP ${${_NAMEVAR}_ADD_INSTALL_STEP})
                IF(MORSE_DEBUG_CMAKE)
                    INSTALL(CODE "EXECUTE_PROCESS(COMMAND ${${_STEP}_CMD} WORKING_DIRECTORY ${${_STEP}_DIR} OUTPUT_QUIET ERROR_QUIET)")
                ELSE(MORSE_DEBUG_CMAKE)
                    INSTALL(CODE "EXECUTE_PROCESS(COMMAND ${${_STEP}_CMD} WORKING_DIRECTORY ${${_STEP}_DIR})")
                ENDIF(MORSE_DEBUG_CMAKE)
            ENDFOREACH()
        ENDIF()

        # Define status of PACKAGE
        # ------------------------
        SET(HAVE_${_NAMEVAR} ON)
        SET(${_NAMEVAR}_EP   ON)

        # Message to advise installation mode
        # -----------------------------------
        IF(MORSE_DEBUG_CMAKE)
            MESSAGE(STATUS "  * debug:")
            MESSAGE(STATUS "  * debug: Installing ${_NAMEVAR}")
            MESSAGE(STATUS "  * debug:")
            MESSAGE(STATUS "  * debug:  - source  : ${${_NAMEVAR}_GETSOURCE}")
            MESSAGE(STATUS "  * debug:  - prefix  : ${${_NAMEVAR}_PATH}")
            MESSAGE(STATUS "  * debug:  - options : ${${_NAMEVAR}_CONFIG_OPTS}")
            MESSAGE(STATUS "  * debug:")
        ENDIF(MORSE_DEBUG_CMAKE)

    ENDIF()

ENDMACRO(INSTALL_EXTERNAL_PACKAGE)

##
## @end file installExternalPACKAGE.cmake
##

