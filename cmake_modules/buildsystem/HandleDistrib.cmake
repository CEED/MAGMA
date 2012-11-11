###
#
#  @file HandleDistrib.cmake
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
# DISCOVER_GRAPH_DISTRIB: Discover graph dependencies
#                         WARNING: recursive macro 
#
###
MACRO(DISCOVER_GRAPH_DISTRIB _PREFIX _NODEUP)

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

        IF(NOT "^$" STREQUAL "^${_DGD_up_nodedown}$")
            # Update a list
            LIST(APPEND FILE_ALL_DEPS ${_DGD_up_nodedown})
            LIST(REMOVE_DUPLICATES FILE_ALL_DEPS)

            # Write the dependency
            WRITE_GRAPH_MEMORY("${_DGD_up_nodeup}"
                               "${_DGD_up_nodedown}"
                               "${${_DGD_up_nodeup}_${_DGD_up_nodedown}_PRIORITY}")

            # Put _DGD_up_nodeup in IEP_MEMORY for recursivity
            LIST(APPEND IEP_MEMORY ${_DGD_up_nodeup})

            # Write the dependencies for _DGD_up_nodedown
            DISCOVER_GRAPH_DISTRIB("${_PREFIX}" "${_DGD_up_nodedown}")

            # Get back _DGD_up_nodeup from IEP_MEMORY
            LIST(LENGTH IEP_MEMORY IEP_SIZE)
            MATH(EXPR IEP_ID "${IEP_SIZE}-1")
            LIST(GET IEP_MEMORY ${IEP_ID} _DGD_up_nodeup)
            LIST(REMOVE_AT IEP_MEMORY ${IEP_ID})
        ENDIF()
    ENDFOREACH()

ENDMACRO(DISCOVER_GRAPH_DISTRIB)


###
#
# CREATE_GRAPH_DISTRIB: call to create the dependencies graph of the distribution of MORSE
#
###
MACRO(CREATE_GRAPH_DISTRIB _PATH _NAME)

    # Define input variable to uppercase
    # ----------------------------------
    STRING(TOUPPER "${_NAME}" _UPPERNAME)
    STRING(TOLOWER "${_NAME}" _LOWERNAME)

    # Create the graph
    # ----------------
    INIT_GRAPH_FILE("${_LOWERNAME}" "${_PATH}" "distrib")
    DISCOVER_GRAPH_DISTRIB("${_UPPERNAME}" "${_UPPERNAME}")
    SET(DISTRIB_ALL_PACKAGES "${FILE_ALL_DEPS}")
    WRITE_GRAPH_FILE("${_LOWERNAME}" "${_PATH}" "distrib")

ENDMACRO(CREATE_GRAPH_DISTRIB)


##
## @end file HandleDistrib.cmake 
##
