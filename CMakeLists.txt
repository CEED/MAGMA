###
#
#  @file CMakeLists.txt
# 
#  @project MAGMA
#  MAGMA is a software package provided by:
#     Inria Bordeaux - Sud-Ouest,
#     Univ. of Tennessee,
#     Univ. of California Berkeley,
#     Univ. of Colorado Denver.
# 
#  @version 0.2.0
#  @author Cedric Castagnede
#  @date 13-07-2012
#   
###

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#############################################
#                                           #
#      Force compilation out-of-source      #
#                                           #
#############################################
STRING(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" MORSE_COMPILE_INPLACE)
IF(MORSE_COMPILE_INPLACE)
    MESSAGE(FATAL_ERROR "MORSE requires an out of source build.
  1/ remove CMakeCache.txt and CMakeFiles in ${CMAKE_SOURCE_DIR}
  2/ do mkdir build && cd build and tape cmake ../")
ENDIF(MORSE_COMPILE_INPLACE)


#############################################
#                                           #
#        Definition of the project          #
#                                           #
#############################################
PROJECT(MORSE Fortran C CXX)


#############################################
#                                           #
#      Systems checks and parameters        #
#                                           #
#############################################

# CMake can find FindFoo.cmake in this directory
# ----------------------------------------------
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules)
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules/cmake-2.8-8)
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules/buildsystem)
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules/find)
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules/info)
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules/install)
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules/options)
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules/testing)
LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules/source_generator)

# Active cmake options
# --------------------
INCLUDE(BuildSystemTools)
INCLUDE(CMakeDependentOption)
INCLUDE(CMakeDetermineSystem)
INCLUDE(CheckCCompilerFlag)
INCLUDE(CheckFunctionExists)
INCLUDE(CheckSymbolExists)
INCLUDE(CheckIncludeFiles)
INCLUDE(Summary)

# Active CTEST
# ------------
INCLUDE(CTest)
ENABLE_TESTING()


#############################################
#                                           #
#     Definition of the GENERIC options     #
#                                           #
#############################################

# Define the default install directory
# ------------------------------------
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    SET(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Install directory used by install" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# Organize output files in a tree
# -------------------------------
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${MORSE_BINARY_DIR}/lib)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${MORSE_BINARY_DIR}/lib)

# Set the build type
# ------------------
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
        "Choose the type of build, options are None, Debug, Release, RelWithDebInfo and MinSizeRel." FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# Misc options
# ------------
OPTION(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# Set computed precisions
# -----------------------
OPTION(BUILD_SINGLE       "Build MORSE Single Precision"         ON)
OPTION(BUILD_DOUBLE       "Build MORSE Double Precision"         ON)
OPTION(BUILD_COMPLEX      "Build MORSE Complex Precision"        ON)
OPTION(BUILD_COMPLEX16    "Build MORSE Double Complex Precision" ON)
CMAKE_DEPENDENT_OPTION(BUILD_MIXEDREAL    "Build MORSE Mixed Real Precision"    ON
                                          "BUILD_SINGLE;BUILD_DOUBLE"           OFF)
CMAKE_DEPENDENT_OPTION(BUILD_MIXEDCOMPLEX "Build MORSE Mixed Complex Precision" ON
                                          "BUILD_COMPLEX;BUILD_COMPLEX16"       OFF)

# Define precision supported by MORSE
# -----------------------------------
IF(BUILD_SINGLE)
    LIST(APPEND MORSE_PRECISION "s")
ENDIF()

IF(BUILD_DOUBLE)
    LIST(APPEND MORSE_PRECISION "d")
ENDIF()

IF(BUILD_MIXEDREAL)
    LIST(APPEND MORSE_PRECISION "ds")
ENDIF()

IF(BUILD_COMPLEX)
    LIST(APPEND MORSE_PRECISION "c")
ENDIF()

IF(BUILD_COMPLEX16)
    LIST(APPEND MORSE_PRECISION "z")
ENDIF()

IF(BUILD_MIXEDCOMPLEX)
    LIST(APPEND MORSE_PRECISION "zc")
ENDIF()


#############################################
#                                           #
#      Definition of the MORSE options      #
#                                           #
#############################################

# MORSE options
# -------------
#OPTION(MORSE_USE_MULTICORE
#  "Enable/Disable multi-threaded kernels in MORSE" OFF)
IF(NOT MORSE_SCHED_STARPU AND NOT MORSE_SCHED_QUARK)
    OPTION(MORSE_SCHED_STARPU
          "Enable/Disable StarPU scheduler in MORSE" ON)
ELSE()
    OPTION(MORSE_SCHED_STARPU
          "Enable/Disable StarPU scheduler in MORSE" OFF)
ENDIF()
OPTION(MORSE_SCHED_QUARK
      "Enable/Disable Quark scheduler in MORSE" OFF)

OPTION(MORSE_SEPARATE_PROJECTS
       "Enable/disable the separation of different projects in the installation tree" ON)
OPTION(MORSE_DEBUG_CMAKE
       "Enable/disable the verbosity of cmake step" OFF)
OPTION(MORSE_ENABLE_TESTING
       "Enable/disable all the tests in MORSE" ON)
OPTION(MORSE_ENABLE_TIMING
       "Enable/disable all the tests in MORSE" ON)

#############################################
#                                           #
#      Definition of the Poject library     #
#                                           #
#############################################

# MAGMA libraries
# ---------------
OPTION(MAGMA "Enable/Disable compilation of MAGMA mono-GPU" ON)

# MAGMA_MORSE libraries
# ---------------------
OPTION(MAGMA_MORSE "Enable/Disable MORSE version of MAGMA" OFF)

# Define extra options for specific package
# -----------------------------------------
INCLUDE(ExtraOptions)
EXTRA_OPTIONS()


#############################################
#                                           #
#             Check the system              #
#                                           #
#############################################

# Detect the system and set specific options
# ------------------------------------------
INCLUDE(SystemDetection)

# Define Fortran mangling
# -----------------------
INCLUDE(FortranMangling)
FortranMangling()

# Verify if ISO C Binding is available
# ------------------------------------
INCLUDE(ISOCBinding)
ISOCBinding()

# PKG_CONFIG
# ----------
FIND_PACKAGE(PkgConfig QUIET)
IF(PKG_CONFIG_EXECUTABLE)
    MESSAGE(STATUS "Found pkg-config: ${PKG_CONFIG_EXECUTABLE}")
ELSE(PKG_CONFIG_EXECUTABLE)
    MESSAGE(STATUS "Looking for pkg-config - not found")
ENDIF(PKG_CONFIG_EXECUTABLE)

# PYTHON
# ------
FIND_PACKAGE(PythonInterp REQUIRED)


#############################################
#                                           #
#       Check options compatibility         #
#                                           #
#############################################

# Create all options and handle user's desires
# --------------------------------------------
INCLUDE(HandleOptions)
HANDLE_OPTIONS()

# Create distribiuted dependencies graph for MORSE
# ------------------------------------------------
INCLUDE(HandleDistrib)
FOREACH(_library ${MORSE_ALL_LIBRARIES};MORSE)
    CREATE_GRAPH_DISTRIB("${CMAKE_BINARY_DIR}" "${_library}")
ENDFOREACH()

# Create requested dependencies graph for MORSE_ALL_LIBRARIES
# -----------------------------------------------------------
INCLUDE(HandleDependencies)
CREATE_GRAPH_DEPENDENCIES("MORSE")


#############################################
#                                           #
#    Look for the required dependencies     #
#                                           #
#############################################

# Looking for dependencies
# ------------------------
INCLUDE(FindDependencies)
FIND_DEPENDENCIES("MORSE")

# Force installation of magma in a particular case
# ------------------------------------------------
IF(MAGMA_MORSE AND CUDA_FOUND)
    MESSAGE(STATUS "Installing MAGMA - installed as an internal project")
    SET(MAGMA ON CACHE BOOL "Enable/Disable compilation of MAGMA mono-GPU" FORCE)
ENDIF(MAGMA_MORSE AND CUDA_FOUND)


#############################################
#                                           #
#        Compilation of the MAGMA           #
#                                           #
#############################################
IF(MAGMA)
    # Create ${CMAKE_BINARY_DIR}/include
    # ----------------------------------
    IF(NOT EXISTS ${CMAKE_BINARY_DIR}/include)
        FILE(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/include)
    ENDIF()

    # Set the version of this project
    # -------------------------------
    INCLUDE(BuildSystemTools)
    GET_VERSION("MAGMA" "${CMAKE_CURRENT_SOURCE_DIR}/include/magma.h")

    # Define pre-processing variables
    # -------------------------------
    ADD_DEFINITIONS(\"-Dmagma_devptr_t=integer\(kind=${SIZEOF_VOID_PTR}\)\") #"
    IF(MAGMA_USE_PLASMA OR HAVE_PLASMA)
        ADD_DEFINITIONS(-DMAGMA_WITH_PLASMA)
    ENDIF(MAGMA_USE_PLASMA OR HAVE_PLASMA)

    # Define precision supported by MAGMA
    # -----------------------------------
    SET(MAGMA_PRECISION ${MORSE_PRECISION})

    # Define the type of the GPU card
    # -------------------------------
    UNSET(MAGMA_GPU_OPTS)
    LIST(APPEND MAGMA_GPU_OPTS "-ccbin=${CMAKE_C_COMPILER}")

    IF(MAGMA_USE_FERMI)
        ADD_DEFINITIONS(-DGPUSHMEM=200)
        LIST(APPEND MAGMA_GPU_OPTS "-arch=sm_20")

    ELSE(MAGMA_USE_FERMI)
        ADD_DEFINITIONS(-DGPUSHMEM=130)
        LIST(APPEND MAGMA_GPU_OPTS "-arch=sm_13")
        LIST(APPEND MAGMA_GPU_OPTS "--generate-code")
        LIST(APPEND MAGMA_GPU_OPTS "arch=compute_13,code=compute_13")
        LIST(APPEND MAGMA_GPU_OPTS "--generate-code")
        LIST(APPEND MAGMA_GPU_OPTS "arch=compute_10,code=compute_10")

    ENDIF(MAGMA_USE_FERMI)

    # Define compilation flags according to the compiler
    # --------------------------------------------------
    IF(CUDA_FOUND)
        LIST(APPEND CUDA_NVCC_FLAGS ${MAGMA_GPU_OPTS})
        LIST(APPEND CUDA_NVCC_FLAGS "--compiler-options")
        LIST(APPEND CUDA_NVCC_FLAGS "-fno-strict-aliasing")
        LIST(APPEND CUDA_NVCC_FLAGS "-DUNIX")
    ENDIF()

    # Add MAGMA includes to everything
    # --------------------------------
    INCLUDE_DIRECTORIES(${CMAKE_BINARY_DIR}/include)
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/include)
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/control)
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/control)
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/interface_cuda)
    INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/interface_cuda)

    # Check for the subdirectories 
    # ----------------------------
    ADD_SUBDIRECTORY(include)
    ADD_SUBDIRECTORY(src)
    ADD_SUBDIRECTORY(magmablas)
    IF(MORSE_ENABLE_TESTING)
        ADD_SUBDIRECTORY(testing)
    ENDIF()

    # Create file magma.pc
    # --------------------
    GET_VERSION("MAGMA" "${CMAKE_CURRENT_SOURCE_DIR}/include/magma.h")
    GENERATE_PKGCONFIG_FILE("MAGMA" "${CMAKE_CURRENT_SOURCE_DIR}/lib/pkgconfig/magma.pc.in")

    # Installation
    # ------------
    INSTALL(DIRECTORY ${CMAKE_BINARY_DIR}/include
            DESTINATION ${CMAKE_INSTALL_PREFIX})

ENDIF()

#############################################
#                                           #
#     Compilation of the MAGMA_MORSE        #
#                                           #
#############################################
IF(MAGMA_MORSE)
    ADD_SUBDIRECTORY(multi-gpu-dynamic)
ENDIF()

#############################################
#                                           #
#    Print the summary of configuration     #
#                                           #
#############################################
SUMMARY()

##
## @end file CMakeLists.txt
##
