###
#
#  @file HandleOptions.cmake
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
#  @date 16-07-2012
#
###

INCLUDE(infoPACKAGE)

###
#
# DEFINE: all dependencies
#
###
UNSET(MORSE_ALL_PACKAGES)

# CPU - Implementation
LIST(APPEND MORSE_ALL_PACKAGES "MPI")

# GPU - Implementation
LIST(APPEND MORSE_ALL_PACKAGES "CUDA")
LIST(APPEND MORSE_ALL_PACKAGES "OPENCL")

# CPU - Detection
LIST(APPEND MORSE_ALL_PACKAGES "FXT")
LIST(APPEND MORSE_ALL_PACKAGES "HWLOC")

# CPU - Linear Algebra
LIST(APPEND MORSE_ALL_PACKAGES "TMG")
LIST(APPEND MORSE_ALL_PACKAGES "BLAS")
LIST(APPEND MORSE_ALL_PACKAGES "CBLAS")
LIST(APPEND MORSE_ALL_PACKAGES "LAPACK")
LIST(APPEND MORSE_ALL_PACKAGES "LAPACKE")
LIST(APPEND MORSE_ALL_PACKAGES "PLASMA")

# GPU - Linear Algebra
LIST(APPEND MORSE_ALL_PACKAGES "CLMAGMA")

# Runtime
LIST(APPEND MORSE_ALL_PACKAGES "STARPU")
LIST(APPEND MORSE_ALL_PACKAGES "QUARK")

###
#
# DEFINE: all libraries
#
###
UNSET(MORSE_ALL_LIBRARIES)

# MORSE library
SET(MORSE ON)
LIST(APPEND MORSE_ALL_LIBRARIES "MAGMA")
LIST(APPEND MORSE_ALL_LIBRARIES "MAGMA_MORSE")

###
#
#
#
###
MACRO(HANDLE_OPTIONS)

    FOREACH(_package ${MORSE_ALL_PACKAGES})

        # Define input variable to uppercase
        # ----------------------------------
        STRING(TOUPPER "${_package}" _UPPERWANTED)
        STRING(TOLOWER "${_package}" _LOWERWANTED)

        # Unset MORSE_REQUIRE_${_UPPERWANTED} to remove memory between two 'cmake`
        # ------------------------------------------------------------------------
        UNSET(MORSE_REQUIRE_${_UPPERWANTED})

        # Switch options according to user's desires
        # ------------------------------------------
        IF("^${_UPPERWANTED}$" STREQUAL "^STARPU$" OR "^${_UPPERWANTED}$" STREQUAL "^QUARK$")
            IF(MORSE_SCHED_STARPU)
                SET(MORSE_USE_STARPU "ON"
                    CACHE STRING "Enable/Disable STARPU dependency (ON/OFF/<not-defined>)")
                SET(MORSE_USE_QUARK  "OFF"
                    CACHE STRING "Enable/Disable QUARK dependency (ON/OFF/<not-defined>)")
                SET(STARPU_TRYUSE   ON)
                SET(STARPU_REQUIRED ON)
                SET(QUARK_TRYUSE    OFF)
                SET(QUARK_REQUIRED  OFF)

            ELSEIF(MORSE_SCHED_QUARK)
                SET(MORSE_USE_STARPU "OFF"
                    CACHE STRING "Enable/Disable STARPU dependency (ON/OFF/<not-defined>)")
                SET(MORSE_USE_QUARK  "ON"
                    CACHE STRING "Enable/Disable QUARK dependency (ON/OFF/<not-defined>)")
                SET(STARPU_TRYUSE   OFF)
                SET(STARPU_REQUIRED OFF)
                SET(QUARK_TRYUSE    ON)
                SET(QUARK_REQUIRED  ON)

            ENDIF()

        ELSE()
            IF(NOT DEFINED MORSE_USE_${_UPPERWANTED})
                SET(MORSE_USE_${_UPPERWANTED} ""
                    CACHE STRING "Enable/Disable ${_UPPERWANTED} dependency (ON/OFF/<not-defined>)")
            ENDIF(NOT DEFINED MORSE_USE_${_UPPERWANTED})

            IF("^${MORSE_USE_${_UPPERWANTED}}$" STREQUAL "^OFF$")
                SET(MORSE_REQUIRE_${_UPPERWANTED} "suggests")
                SET(${_UPPERWANTED}_TRYUSE   OFF)
                SET(${_UPPERWANTED}_REQUIRED OFF)

            ELSEIF("^${MORSE_USE_${_UPPERWANTED}}$" STREQUAL "^ON$" OR
                   NOT "^${MORSE_USE_${_UPPERWANTED}}$" STREQUAL "^$")
                SET(MORSE_REQUIRE_${_UPPERWANTED} "depends")
                SET(${_UPPERWANTED}_TRYUSE   ON)
                SET(${_UPPERWANTED}_REQUIRED ON)

            ENDIF()
        ENDIF()

        # Set policy about ${_NAMEVAR} (install/find/set/...)
        # ---------------------------------------------------

        # Define default values for options
        OPTION(${_UPPERWANTED}_USE_LIB
               "Enable/Disable to link with ${_UPPERWANTED}" OFF)
        OPTION(${_UPPERWANTED}_USE_SYSTEM
               "Enable/Disable to look for ${_UPPERWANTED} installation in environment" OFF)
        OPTION(${_UPPERWANTED}_USE_WEB
               "Enable/Disable to install ${_UPPERWANTED} form web" OFF)
        OPTION(${_UPPERWANTED}_USE_TARBALL
               "Enable/Disable to install ${_UPPERWANTED} form a tarball" OFF)
        OPTION(${_UPPERWANTED}_USE_SVN
               "Enable/Disable to install ${_UPPERWANTED} form the repository" OFF)
        OPTION(${_UPPERWANTED}_USE_AUTO
               "Enable/Disable to install ${_UPPERWANTED} if it was not found" ON)
        MARK_AS_ADVANCED(${_UPPERWANTED}_USE_LIB)
        MARK_AS_ADVANCED(${_UPPERWANTED}_USE_SYSTEM)
        MARK_AS_ADVANCED(${_UPPERWANTED}_USE_WEB)
        MARK_AS_ADVANCED(${_UPPERWANTED}_USE_TARBALL)
        MARK_AS_ADVANCED(${_UPPERWANTED}_USE_SVN)
        MARK_AS_ADVANCED(${_UPPERWANTED}_USE_AUTO)

        # Set the policy if ***_LIB + ***_INC is defined
        IF(DEFINED ${_UPPERWANTED}_LIB OR ${_UPPERWANTED}_USE_LIB)
            SET(${_UPPERWANTED}_USE_LIB ON CACHE BOOL
                "Enable/Disable to link with ${_UPPERWANTED}")
            SET(${_UPPERWANTED}_USE_AUTO OFF CACHE BOOL
                "Enable/Disable to install ${_UPPERWANTED} if it was not found")
            IF(NOT DEFINED MORSE_USE_${_UPPERWANTED})
                SET(MORSE_USE_${_UPPERWANTED} "ON" CACHE STRING
                    "Enable/Disable ${_UPPERWANTED} dependency (ON/OFF/<not-defined>)" FORCE)
            ENDIF(NOT DEFINED MORSE_USE_${_UPPERWANTED})
            SET(MORSE_REQUIRE_${_UPPERWANTED} "suggests")
            SET(${_UPPERWANTED}_TRYUSE   ON)
            SET(${_UPPERWANTED}_REQUIRED ON)
        ELSE()
            SET(${_UPPERWANTED}_USE_LIB OFF CACHE BOOL
                "Enable/Disable to link with ${_UPPERWANTED}")
            SET(${_UPPERWANTED}_USE_AUTO ON CACHE BOOL
                "Enable/Disable to install ${_UPPERWANTED} if it was not found")
        ENDIF()

        # Set the policy if ***_DIR is defined
        IF(DEFINED ${_UPPERWANTED}_DIR OR ${_UPPERWANTED}_USE_SYSTEM)
            SET(${_UPPERWANTED}_USE_SYSTEM ON CACHE BOOL
                "Enable/Disable to look for ${_UPPERWANTED} installation in environment")
            SET(${_UPPERWANTED}_USE_AUTO OFF CACHE BOOL
                "Enable/Disable to install ${_UPPERWANTED} if it was not found")
            IF(NOT DEFINED MORSE_USE_${_UPPERWANTED})
                SET(MORSE_USE_${_UPPERWANTED} "ON" CACHE STRING
                    "Enable/Disable ${_UPPERWANTED} dependency (ON/OFF/<not-defined>)" FORCE)
            ENDIF(NOT DEFINED MORSE_USE_${_UPPERWANTED})
            SET(MORSE_REQUIRE_${_UPPERWANTED} "suggests")
            SET(${_UPPERWANTED}_TRYUSE   ON)
            SET(${_UPPERWANTED}_REQUIRED ON)
        ELSE()
            SET(${_UPPERWANTED}_USE_SYSTEM OFF CACHE BOOL
                "Enable/Disable to look for ${_UPPERWANTED} installation in environment")
            SET(${_UPPERWANTED}_USE_AUTO ON CACHE BOOL
                "Enable/Disable to install ${_UPPERWANTED} if it was not found")
        ENDIF()

        # Set the policy if ***_URL is defined
        IF(DEFINED ${_UPPERWANTED}_URL OR ${_UPPERWANTED}_USE_WEB)
            SET(${_UPPERWANTED}_USE_WEB ON CACHE BOOL
                "Enable/Disable to install ${_UPPERWANTED} form web")
            SET(${_UPPERWANTED}_USE_AUTO OFF CACHE BOOL
                "Enable/Disable to install ${_UPPERWANTED} if it was not found")
            IF(NOT DEFINED MORSE_USE_${_UPPERWANTED})
                SET(MORSE_USE_${_UPPERWANTED} "ON" CACHE STRING
                    "Enable/Disable ${_UPPERWANTED} dependency (ON/OFF/<not-defined>)" FORCE)
            ENDIF(NOT DEFINED MORSE_USE_${_UPPERWANTED})
            SET(MORSE_REQUIRE_${_UPPERWANTED} "suggests")
            SET(${_UPPERWANTED}_TRYUSE   ON)
            SET(${_UPPERWANTED}_REQUIRED ON)
        ELSE()
            SET(${_UPPERWANTED}_USE_WEB OFF CACHE BOOL
                "Enable/Disable to install ${_UPPERWANTED} form web")
            SET(${_UPPERWANTED}_USE_AUTO ON CACHE BOOL
                "Enable/Disable to install ${_UPPERWANTED} if it was not found")
        ENDIF()

        # Set the policy if ***_TARBALL is defined
        IF(DEFINED ${_UPPERWANTED}_TARBALL OR ${_UPPERWANTED}_USE_TARBALL)
            SET(${_UPPERWANTED}_USE_TARBALL ON CACHE BOOL
                "Enable/Disable to install ${_UPPERWANTED} form a tarball")
            SET(${_UPPERWANTED}_USE_AUTO OFF CACHE BOOL
                "Enable/Disable to install ${_UPPERWANTED} if it was not found")
            IF(NOT DEFINED MORSE_USE_${_UPPERWANTED})
                SET(MORSE_USE_${_UPPERWANTED} "ON" CACHE STRING
                    "Enable/Disable ${_UPPERWANTED} dependency (ON/OFF/<not-defined>)" FORCE)
            ENDIF(NOT DEFINED MORSE_USE_${_UPPERWANTED})
            SET(MORSE_REQUIRE_${_UPPERWANTED} "suggests")
            SET(${_UPPERWANTED}_TRYUSE   ON)
            SET(${_UPPERWANTED}_REQUIRED ON)
        ELSE()
            SET(${_UPPERWANTED}_USE_TARBALL OFF CACHE BOOL
                "Enable/Disable to install ${_UPPERWANTED} form a tarball")
            SET(${_UPPERWANTED}_USE_AUTO ON CACHE BOOL
                "Enable/Disable to install ${_UPPERWANTED} if it was not found")
        ENDIF()

        # Set the policy if ***_SVN
        #
        # TODO: put a list like: ***_SVN="repository;type_repository;username;password"
        #
    
        # Desactive some stuff (eg installation of CUDA)
        # ----------------------------------------------
        #
        # TODO:
        #

        # Print debug message
        # -------------------
        DEBUG_HANDLE_OPTIONS("${_UPPERWANTED}")

    ENDFOREACH()

ENDMACRO(HANDLE_OPTIONS)

###
#
#
#
###
MACRO(DEBUG_HANDLE_OPTIONS _PACKAGE)
    # Status of internal variables
    # ----------------------------
    IF(MORSE_DEBUG_CMAKE)
        MESSAGE(STATUS "  * debug:")
        MESSAGE(STATUS "  * debug: Status of ${_PACKAGE}:")
        MESSAGE(STATUS "  * debug:")
        MESSAGE(STATUS "  * debug: --> MORSE_USE_${_PACKAGE}     : ${MORSE_USE_${_PACKAGE}}")
        MESSAGE(STATUS "  * debug: --> Internal dependency management behavior:")
        MESSAGE(STATUS "  * debug:     - ${_PACKAGE}_TRYUSE      : ${${_PACKAGE}_TRYUSE}")
        MESSAGE(STATUS "  * debug:     - ${_PACKAGE}_REQUIRED    : ${${_PACKAGE}_REQUIRED}")
        MESSAGE(STATUS "  * debug:     - ${_PACKAGE}_USE_LIB     : ${${_PACKAGE}_USE_LIB}")
        MESSAGE(STATUS "  * debug:     - ${_PACKAGE}_USE_SYSTEM  : ${${_PACKAGE}_USE_SYSTEM}")
        MESSAGE(STATUS "  * debug:     - ${_PACKAGE}_USE_TARBALL : ${${_PACKAGE}_USE_TARBALL}")
        MESSAGE(STATUS "  * debug:     - ${_PACKAGE}_USE_WEB     : ${${_PACKAGE}_USE_WEB}")
        MESSAGE(STATUS "  * debug:     - ${_PACKAGE}_USE_AUTO    : ${${_PACKAGE}_USE_AUTO}")
        MESSAGE(STATUS "  * debug: --> Given user's values:")
        MESSAGE(STATUS "  * debug:     - ${_PACKAGE}_LIB         : ${${_PACKAGE}_LIB}")
        MESSAGE(STATUS "  * debug:     - ${_PACKAGE}_INC         : ${${_PACKAGE}_INC}")
        MESSAGE(STATUS "  * debug:     - ${_PACKAGE}_DIR         : ${${_PACKAGE}_DIR}")
        MESSAGE(STATUS "  * debug:     - ${_PACKAGE}_URL         : ${${_PACKAGE}_URL}")
        MESSAGE(STATUS "  * debug:     - ${_PACKAGE}_TARBALL     : ${${_PACKAGE}_TARBALL}")
        MESSAGE(STATUS "  * debug:")

    ENDIF(MORSE_DEBUG_CMAKE)

ENDMACRO(DEBUG_HANDLE_OPTIONS)


##
## @end file HandleOptions.cmake
##
