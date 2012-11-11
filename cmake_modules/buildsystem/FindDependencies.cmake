###
#
#  @file FindDependencies.cmake
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
#  @date 15-07-2012
#
###

INCLUDE(DefineDependencies)

###
#
# FIND_DEPENDENCIES: Macro to call installation of dependencies
#
###
MACRO(FIND_DEPENDENCIES _NAME)

    # Define input variable to uppercase
    # ----------------------------------
    STRING(TOUPPER "${_NAME}" _UPFDVAR)
    STRING(TOLOWER "${_NAME}" _LOWFDVAR)

    # Determine the end of the loop
    # -----------------------------
    # if it is MORSE, we need to use ${_UPFDVAR}_DIRECT_REQUESTED
    # if not, we need to use ${_UPFDVAR}_DIRECT_DEPS
    IF("^${_UPFDVAR}$" STREQUAL "^MORSE$")
        SET(${_UPFDVAR}_END_LOOP ${${_UPFDVAR}_DIRECT_REQUESTED})
    ELSE()
        SET(${_UPFDVAR}_END_LOOP ${${_UPFDVAR}_DIRECT_DEPS}) 
    ENDIF()

    # Define dependencies
    # -------------------
    FOREACH(_dep ${${_UPFDVAR}_END_LOOP})
        IF(NEED_${_dep})
            # Define input variable to uppercase
            STRING(TOUPPER "${_dep}" _DEPFDVAR)
            STRING(TOLOWER "${_dep}" _LOWDEPVAR)

            # Search for dependencies
            IF("^$" STREQUAL "^${${_DEPFDVAR}_USED_MODE}$")
                IF(MORSE_DEBUG_CMAKE)
                    MESSAGE(STATUS "Define ${_UPFDVAR} - ${_UPFDVAR} ${MORSE_REQUIRE_${_DEPFDVAR}} ${_DEPFDVAR}")
                ENDIF(MORSE_DEBUG_CMAKE)

                # Put _UPFDVAR in BFD_MEMORY for recursivity
                LIST(APPEND BFD_MEMORY_1 ${_UPFDVAR})
                LIST(APPEND BFD_MEMORY_2 ${_DEPFDVAR})

                # Looking for _DEPFDVAR
                DEFINE_PACKAGE("${_UPFDVAR}" "${_DEPFDVAR}" "${MORSE_REQUIRE_${_DEPFDVAR}}")

                # Get back _UPFDVAR from BFD_MEMORY_1
                LIST(LENGTH BFD_MEMORY_1 BFD_SIZE)
                MATH(EXPR IEP_ID "${BFD_SIZE}-1")
                LIST(GET BFD_MEMORY_1 ${IEP_ID} _UPFDVAR)
                LIST(REMOVE_AT BFD_MEMORY_1 ${IEP_ID})

                # Get back _UPFDVAR from BFD_MEMORY_2
                LIST(LENGTH BFD_MEMORY_2 BFD_SIZE)
                MATH(EXPR IEP_ID "${BFD_SIZE}-1")
                LIST(GET BFD_MEMORY_2 ${IEP_ID} _DEPFDVAR)
                LIST(REMOVE_AT BFD_MEMORY_2 ${IEP_ID})

            ELSE(NOT DEFINED ${_DEPFDVAR}_USED_MODE)
                IF(MORSE_DEBUG_CMAKE)
                    MESSAGE(STATUS "Define ${_UPFDVAR} - ${_UPFDVAR} ${MORSE_REQUIRE_${_DEPFDVAR}} ${_DEPFDVAR} but mode ${${_DEPFDVAR}_USED_MODE} already defined")
                ENDIF(MORSE_DEBUG_CMAKE)
            ENDIF()

            # Define dependencies if there are build steps
            IF(${${_DEPFDVAR}_EP})
                STRING(TOLOWER "${_DEPFDVAR}" _LOWDEPVAR)
                LIST(APPEND ${_UPFDVAR}_MAKEDEP ${_LOWDEPVAR}_build)
                LIST(REMOVE_DUPLICATES ${_UPFDVAR}_MAKEDEP)
            ENDIF()
        ENDIF(NEED_${_dep})
    ENDFOREACH()

ENDMACRO(FIND_DEPENDENCIES)

##
## @end file FindDependencies.cmake
##
