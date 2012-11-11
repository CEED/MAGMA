###
#
#  @file HandleDependencies.cmake
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
INCLUDE(WriteGraph)

###
#
#
#
###
MACRO(MODIFY_REQUIRE_DEPENDENCIES _PREFIX _PACKAGE _STATUS)

    # Define input variable to uppercase
    # ----------------------------------
    STRING(TOUPPER "${_STATUS}" _newSTATUS_UP)
    STRING(TOUPPER "${${_PREFIX}_REQUIRE_${_PACKAGE}}" _oldSTATUS_UP)

    # Redefined the dependency
    # ------------------------
    SET(${_PREFIX}_REQUIRE_${_PACKAGE} "${_STATUS}")
    LIST(APPEND ${_PREFIX}_${_newSTATUS_UP}_DEPS "${_PACKAGE}")
    LIST(REMOVE_DUPLICATES ${_PREFIX}_${_newSTATUS_UP}_DEPS)

    # Remove old status
    # -----------------
    IF(NOT "^${_oldSTATUS_UP}$" STREQUAL "^${_newSTATUS_UP}$")
        LIST(FIND ${_PREFIX}_${_oldSTATUS_UP}_DEPS "${_PACKAGE}" IS_${_PACKAGE})
        IF(NOT "^${IS_${_PACKAGE}}$" STREQUAL "^-1$")
            LIST(REMOVE_ITEM ${_PREFIX}_${_oldSTATUS_UP}_DEPS "${_PACKAGE}")
        ENDIF()
    ENDIF()

    # Update list of ***_ALL_REQUESTED
    # --------------------------------
    UNSET(${_PREFIX}_ALL_REQUESTED)
    LIST(APPEND ${_PREFIX}_ALL_REQUESTED ${${_PREFIX}_RECOMMENDS_DEPS})
    LIST(APPEND ${_PREFIX}_ALL_REQUESTED ${${_PREFIX}_DEPENDS_DEPS})

ENDMACRO(MODIFY_REQUIRE_DEPENDENCIES)


###
#
# RECORD_MORSE_DEPENDENCIES: record the graph dependencies according to user's request
#
###
MACRO(RECORD_MORSE_DEPENDENCIES _PREFIX _NODE_UP _NODE_DOWN)

    # Analyse dependencies
    IF("^${${_PREFIX}_REQUIRE_${_NODE_DOWN}}$" STREQUAL "^$")
        MODIFY_REQUIRE_DEPENDENCIES("${_PREFIX}"
                                    "${_NODE_DOWN}"
                                    "${${_NODE_UP}_${_NODE_DOWN}_PRIORITY}")

    ELSE()
        IF("${${_NODE_UP}_${_NODE_DOWN}_PRIORITY}" STREQUAL "depends")
            MODIFY_REQUIRE_DEPENDENCIES("${_PREFIX}"
                                        "${_NODE_DOWN}"
                                        "depends")

        ELSEIF("${${_NODE_UP}_${_NODE_DOWN}_PRIORITY}" STREQUAL "recommends")
            IF(NOT "${${_PREFIX}_REQUIRE_${_NODE_DOWN}}" STREQUAL "depends")
                MODIFY_REQUIRE_DEPENDENCIES("${_PREFIX}"
                                            "${_NODE_DOWN}"
                                            "recommends")
            ENDIF()

        ELSEIF("${${_NODE_UP}_${_NODE_DOWN}_PRIORITY}" STREQUAL "suggests")
            IF(NOT "${${_PREFIX}_REQUIRE_${_NODE_DOWN}}" STREQUAL "depends" AND
               NOT "${${_PREFIX}_REQUIRE_${_NODE_DOWN}}" STREQUAL "recommends")
                MODIFY_REQUIRE_DEPENDENCIES("${_PREFIX}"
                                            "${_NODE_DOWN}"
                                            "suggests")
            ENDIF()

        ENDIF()
    ENDIF()



ENDMACRO(RECORD_MORSE_DEPENDENCIES)


###
#
# FINALIZE_RECORD_MORSE: Write graph dependencies
#                        WARNING: recursive macro 
#
###
MACRO(FINALIZE_RECORD_MORSE _PREFIX _NODEUP)

    # Define input variable to uppercase
    # ----------------------------------
    STRING(TOUPPER "${_PREFIX}" _DGD_up_prefix)
    STRING(TOUPPER "${_NODEUP}" _DGD_up_nodeup)

    # Load info about package
    # -----------------------
    INFO_DEPS_PACKAGE("${_DGD_up_nodeup}")

    # Write dependencies
    # ------------------
    FOREACH(_dep ${${_DGD_up_nodeup}_DIRECT_DEPS})
        # Define input variable to uppercase
        STRING(TOUPPER "${_dep}" _DGD_up_nodedown)

        # Write in file
        IF(NOT "^$" STREQUAL "^${_DGD_up_nodedown}$" AND
           NOT "${${_DGD_up_nodeup}_${_DGD_up_nodedown}_PRIORITY}" STREQUAL "suggests")
            # Write in the memory
            WRITE_GRAPH_MEMORY("${_DGD_up_nodeup}"
                               "${_DGD_up_nodedown}"
                               "${MORSE_REQUIRE_${_DGD_up_nodedown}}")

            # Put _DGD_up_nodeup in IEP_MEMORY for recursivity
            LIST(APPEND IEP_MEMORY ${_DGD_up_nodeup})

            # Write the dependencies for _DGD_up_nodedown
            FINALIZE_RECORD_MORSE("${_DGD_up_prefix}" "${_DGD_up_nodedown}")

            # Get back _DGD_up_nodeup from IEP_MEMORY
            LIST(LENGTH IEP_MEMORY IEP_SIZE)
            MATH(EXPR IEP_ID "${IEP_SIZE}-1")
            LIST(GET IEP_MEMORY ${IEP_ID} _DGD_up_nodeup)
            LIST(REMOVE_AT IEP_MEMORY ${IEP_ID})
        ENDIF()
    ENDFOREACH()

ENDMACRO(FINALIZE_RECORD_MORSE)

###
#
# DISCOVER_GRAPH_DEPENDENCIES: Discover graph dependencies
#                              WARNING: recursive macro 
#
###
MACRO(DISCOVER_GRAPH_DEPENDENCIES _PREFIX _NODEUP)

    # Define input variable to uppercase
    # ----------------------------------
    STRING(TOUPPER "${_PREFIX}" _DGD_up_prefix)
    STRING(TOUPPER "${_NODEUP}" _DGD_up_nodeup)

    # Load info about package
    # -----------------------
    INFO_DEPS_PACKAGE("${_DGD_up_nodeup}")

    # Discover dependencies
    # ---------------------
    FOREACH(_dep ${${_DGD_up_nodeup}_DIRECT_DEPS})
        # Define input variable to uppercase
        STRING(TOUPPER "${_dep}" _DGD_up_nodedown)

        IF(NOT "^$" STREQUAL "^${_DGD_up_nodedown}$" AND
           NOT "${${_DGD_up_nodeup}_${_DGD_up_nodedown}_PRIORITY}" STREQUAL "suggests")
            # Update a list
            LIST(APPEND FILE_ALL_DEPS ${_DGD_up_nodedown})
            LIST(REMOVE_DUPLICATES FILE_ALL_DEPS)

            # Add new entry form the database
            RECORD_MORSE_DEPENDENCIES("${_DGD_up_prefix}"
                                      "${_DGD_up_nodeup}"
                                      "${_DGD_up_nodedown}")

            # Put _DGD_up_nodeup in IEP_MEMORY for recursivity
            LIST(APPEND IEP_MEMORY ${_DGD_up_nodeup})

            # Write the dependencies for _DGD_up_nodedown
            IF("${${_DGD_up_nodeup}_${_DGD_up_nodedown}_PRIORITY}" STREQUAL "depends")
                DISCOVER_GRAPH_DEPENDENCIES("${_DGD_up_prefix}" "${_DGD_up_nodedown}")
            ENDIF()

            # Get back _DGD_up_nodeup from IEP_MEMORY
            LIST(LENGTH IEP_MEMORY IEP_SIZE)
            MATH(EXPR IEP_ID "${IEP_SIZE}-1")
            LIST(GET IEP_MEMORY ${IEP_ID} _DGD_up_nodeup)
            LIST(REMOVE_AT IEP_MEMORY ${IEP_ID})
        ENDIF()
    ENDFOREACH()

ENDMACRO(DISCOVER_GRAPH_DEPENDENCIES)

### 
#
# CREATE_GRAPH_DISTRIB: call to create the dependencies graph of the distribution of MORSE
#
###
MACRO(CREATE_GRAPH_DEPENDENCIES _NAME)

    # Define input variable to uppercase
    # ----------------------------------
    STRING(TOUPPER "${_NAME}" _UPPERNAME)
    STRING(TOLOWER "${_NAME}" _LOWERNAME)

    # Give informations
    # -----------------
    INFO_DEPS_PACKAGE("${_UPPERNAME}")
    MESSAGE(STATUS "Calling request graph dependencies")
    MESSAGE(STATUS "Calling request graph dependencies - ${${_UPPERNAME}_DIRECT_DEPS} are explored")

    # Create the graph
    # ----------------
    INIT_GRAPH_FILE("${_LOWERNAME}" "${CMAKE_BINARY_DIR}" "request")
    DISCOVER_GRAPH_DEPENDENCIES("${_UPPERNAME}" "${_UPPERNAME}")
    FINALIZE_RECORD_MORSE("${_UPPERNAME}" "${_UPPERNAME}")
    WRITE_GRAPH_FILE("${_LOWERNAME}" "${CMAKE_BINARY_DIR}" "request")

    # Status of internal variables
    # ----------------------------
    MESSAGE(STATUS "Parsing dependencies - ${_UPPERNAME} direct dependencies      : ${${_UPPERNAME}_DIRECT_DEPS}")
    MESSAGE(STATUS "Parsing dependencies - ${_UPPERNAME} required dependencies    : ${${_UPPERNAME}_DEPENDS_DEPS}")
    MESSAGE(STATUS "Parsing dependencies - ${_UPPERNAME} recommended dependencies : ${${_UPPERNAME}_RECOMMENDS_DEPS}")

    # Add a variable to know if the package is required to be tested
    # --------------------------------------------------------------
    #SET(${_UPPERNAME}_DIRECT_REQUESTED CACHE STRING "")
    SET(${_UPPERNAME}_DIRECT_REQUESTED "")
    FOREACH(_package ${${_UPPERNAME}_ALL_REQUESTED})
        IF("^${_package}$" STREQUAL "^MAGMA$" OR
           "^${_package}$" STREQUAL "^MAGMA_MORSE$")
            FOREACH(_dep ${${_package}_DIRECT_DEPS})
                LIST(FIND ${_UPPERNAME}_ALL_REQUESTED "${_dep}" IS_${_dep})
                IF(NOT "^${IS_${_dep}}$" STREQUAL "^-1$")
                    LIST(APPEND ${_UPPERNAME}_DIRECT_REQUESTED ${_dep})
                ENDIF()
            ENDFOREACH()

        ELSE()
            LIST(APPEND ${_UPPERNAME}_DIRECT_REQUESTED ${_package})

        ENDIF()
        SET(NEED_${_package} ON)
    ENDFOREACH()
    LIST(REMOVE_DUPLICATES ${_UPPERNAME}_DIRECT_REQUESTED)

ENDMACRO(CREATE_GRAPH_DEPENDENCIES)

##
## @end file HandleDependencies.cmake
##
