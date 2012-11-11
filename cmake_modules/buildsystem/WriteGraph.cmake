###
#
#  @file WriteGraph.cmake
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
# INIT_FILE_DISTRIB: create the dot file and initialize it
#
# _TYPE : <distrib|request>
#
###
MACRO(INIT_GRAPH_FILE _NAME _PREFIX _TYPE)

    # Define input variable to uppercase
    # ----------------------------------
    STRING(TOUPPER "${_NAME}" _UPPERNAME)
    STRING(TOLOWER "${_NAME}" _LOWERNAME)

    # Clean the contents memory
    # -------------------------
    UNSET(FILE_ALL_DEPS)
    UNSET(FILE_DISTRIB)

    # Generate the graph as dot file
    # ------------------------------
    IF("^${_TYPE}$" STREQUAL "^distrib$")
        IF(EXISTS ${_PREFIX}/dependencies_graph_distrib_${_LOWERNAME}.dot)
            FILE(REMOVE ${_PREFIX}/dependencies_graph_distrib_${_LOWERNAME}.dot)
        ENDIF()
        FILE(APPEND ${_PREFIX}/dependencies_graph_distrib_${_LOWERNAME}.dot
"
digraph G{
# depends:    blue
# recommends: black
# suggests:   black, dotted
node [shape=box]
0 [style=\"invis\"]
1 [style=\"invis\"]
2 [style=\"invis\"]
3 [style=\"invis\"]
0 -> 1 [color=blue, label=\"depends\"]
1 -> 2 [color=black, label=\"recommends\"]
2 -> 3 [color=black,style=dotted, label=\"suggests\"]
{rank=same;0;1;2;3}
#0 -> Binary          [style=\"invis\"]
0 -> Library         [style=\"invis\"]
0 -> ExternalPackage [style=\"invis\"]
#Binary            [color=green,style=bold]
Library           [color=red,style=bold]
ExternalPackage   [shape=rectangle]
{rank=same;Library;ExternalPackage}
"
        )

    ELSEIF("^${_TYPE}$" STREQUAL "^request$")
        IF(EXISTS ${_PREFIX}/dependencies_graph_request_${_LOWERNAME}.dot)
            FILE(REMOVE ${_PREFIX}/dependencies_graph_request_${_LOWERNAME}.dot)
        ENDIF()
        FILE(APPEND ${_PREFIX}/dependencies_graph_request_${_LOWERNAME}.dot
"
digraph G{
# requested:   blue
# recommended: black
node [shape=box]
0 [style=\"invis\"]
1 [style=\"invis\"]
2 [style=\"invis\"]
0 -> 1 [color=blue, label=\"depends\"]
1 -> 2 [color=black, label=\"recommends\"]
{rank=same;0;1;2}
#0 -> Binary          [style=\"invis\"]
0 -> Library         [style=\"invis\"]
0 -> ExternalPackage [style=\"invis\"]
#Binary            [color=green,style=bold]
Library           [color=red,style=bold]
ExternalPackage   [shape=rectangle]
{rank=same;Library;ExternalPackage}
"
        )
    ENDIF()

ENDMACRO(INIT_GRAPH_FILE)


###
#
# WRITE_FILE_DISTRIB: write the memory and the end of the dot file
#
###
MACRO(WRITE_GRAPH_FILE _NAME _PREFIX _TYPE)

    # Define input variable to uppercase
    # ----------------------------------
    STRING(TOUPPER "${_NAME}" _UPPERNAME)
    STRING(TOLOWER "${_NAME}" _LOWERNAME)

    # Looking for used packages and putting on the same rank
    # ------------------------------------------------------
    SET(LEVEL1 "")
    SET(LEVEL2 "")
    SET(LEVEL3 "")
    FOREACH(_package ${FILE_ALL_DEPS})
        IF("^${_package}$" STREQUAL "^BLAS$"     OR 
           "^${_package}$" STREQUAL "^MPI$"      OR 
           "^${_package}$" STREQUAL "^CUDA$"     OR 
           "^${_package}$" STREQUAL "^OPENCL$"   OR 
           "^${_package}$" STREQUAL "^FXT$"      OR 
           "^${_package}$" STREQUAL "^HWLOC$"      )
            SET(LEVEL1 "${LEVEL1} ${_package}")
        ENDIF()
        IF("^${_package}$" STREQUAL "^CBLAS$"    OR
           "^${_package}$" STREQUAL "^LAPACK$"     )
            SET(LEVEL2 "${LEVEL2} ${_package}")
        ENDIF()
        IF("^${_package}$" STREQUAL "^QUARK$"    OR
           "^${_package}$" STREQUAL "^STARPU$"     )
            SET(LEVEL3 "${LEVEL3} ${_package}")
        ENDIF()
    ENDFOREACH()

    # Write the file
    # --------------
    FOREACH(_ligne ${FILE_DISTRIB})
        FILE(APPEND ${_PREFIX}/dependencies_graph_${_TYPE}_${_LOWERNAME}.dot "${_ligne}")
    ENDFOREACH()
    FILE(APPEND ${_PREFIX}/dependencies_graph_${_TYPE}_${_LOWERNAME}.dot
"
{rank=same; ${LEVEL1}}
{rank=same; ${LEVEL2}}
{rank=same; ${LEVEL3}}
}
"
        )

    # Clean memory
    # ------------
    UNSET(FILE_DISTRIB)
    UNSET(FILE_ALL_DEPS)

    # Generate the graph as pdf
    # -------------------------
    FIND_PROGRAM(DOT_COMPILER dot)
    IF(DOT_COMPILER)
        EXECUTE_PROCESS(
            COMMAND ${DOT_COMPILER} -Tpdf dependencies_graph_${_TYPE}_${_LOWERNAME}.dot
                                    -o dependencies_graph_${_TYPE}_${_LOWERNAME}.pdf
            WORKING_DIRECTORY ${_PREFIX}
                       )
    ENDIF(DOT_COMPILER)

ENDMACRO(WRITE_GRAPH_FILE)


###
#
# WRITE_FILE_MEMORY_DISTRIB: add in a memory the future content of the dot file
#
###
MACRO(WRITE_GRAPH_MEMORY _package1 _package2 _status)

    # Generate the options that compile MORSE_ALL_LIBRAIRIES 
    # ------------------------------------------------------
    IF("^MORSE$" STREQUAL "^${_package1}$")
        FOREACH(_pak ${MORSE_ALL_LIBRARIES})
            IF("^${_package2}$" STREQUAL "^${_pak}$")
                IF("^${${_package2}}$" STREQUAL "^ON$")
                    LIST(APPEND FILE_DISTRIB
                         "${_package1} -> ${_package2} [color=blue]\n")
                ELSE()
                    LIST(APPEND FILE_DISTRIB
                         "${_package1} -> ${_package2} [color=black,label=\"${_package2}\"]\n")
                ENDIF()
            ENDIF()
        ENDFOREACH()

    ELSE()
        # Handle dependencies about MORSE_ALL_PACKAGES
        IF("${_status}" STREQUAL "depends")
            LIST(APPEND FILE_DISTRIB
                 "${_package1} -> ${_package2} [color=blue]\n")
    
        ELSEIF("${_status}" STREQUAL "recommends")
            IF("^${_package1}$" STREQUAL "^MAGMA_MORSE$" AND "^${_package2}$" STREQUAL "^MAGMA$")
                LIST(APPEND FILE_DISTRIB
                "${_package1} -> ${_package2} [color=black,label=\"MORSE_USE_CUDA\"]\n")

            ELSEIF("^${_package1}$" STREQUAL "^MAGMA$" AND "^${_package2}$" STREQUAL "^PLASMA$")
                LIST(APPEND FILE_DISTRIB
                "${_package1} -> ${_package2} [color=black,label=\"${_package1}_USE_${_package2}\"]\n")

            ELSEIF("^${_package2}$" STREQUAL "^STARPU$" OR "^${_package2}$" STREQUAL "^QUARK$")
                LIST(APPEND FILE_DISTRIB
                "${_package1} -> ${_package2} [color=black,label=\"MORSE_SCHED_${_package2}\"]\n")

            ELSE()
                LIST(APPEND FILE_DISTRIB
                "${_package1} -> ${_package2} [color=black,label=\"MORSE_USE_${_package2}\"]\n")

            ENDIF()

        ELSEIF("${_status}" STREQUAL "suggests")
            IF("^${_package1}$" STREQUAL "^MAGMA_MORSE$" AND "^${_package2}$" STREQUAL "^MAGMA$")
                LIST(APPEND FILE_DISTRIB
                "${_package1} -> ${_package2} [color=black,style=dotted,label=\"MORSE_USE_CUDA\"]\n")

            ELSEIF("^${_package1}$" STREQUAL "^MAGMA$" AND "^${_package2}$" STREQUAL "^PLASMA$")
                LIST(APPEND FILE_DISTRIB
                "${_package1} -> ${_package2} [color=black,style=dotted,label=\"${_package1}_USE_${_package2}\"]\n")

            ELSEIF("^${_package2}$" STREQUAL "^STARPU$" OR "^${_package2}$" STREQUAL "^QUARK$")
                LIST(APPEND FILE_DISTRIB
                "${_package1} -> ${_package2} [color=black,style=dotted,label=\"MORSE_SCHED_${_package2}\"]\n")

            ELSE()
                LIST(APPEND FILE_DISTRIB
                "${_package1} -> ${_package2} [color=black,style=dotted,label=\"MORSE_USE_${_package2}\"]\n")

            ENDIF()

        ENDIF()

    ENDIF()

    # Specific tag of ExternalPackage
    FOREACH(_pak1 ${_package1};${_package2})
        FOREACH(_pak2 ${MORSE_ALL_PACKAGES})
            IF("^${_pak1}$" STREQUAL "^${_pak2}$")
                 LIST(APPEND FILE_DISTRIB 
                      "${_pak1} [shape=rectangle]\n")
            ENDIF()
        ENDFOREACH()
    ENDFOREACH()

    # Specific tag of library
    FOREACH(_pak1 ${_package1};${_package2})
        FOREACH(_pak2 ${MORSE_ALL_LIBRARIES})
            IF("^${_pak1}$" STREQUAL "^${_pak2}$")
                 LIST(APPEND FILE_DISTRIB 
                      "${_pak1} [color=red,style=bold]\n")
            ENDIF()
        ENDFOREACH()
    ENDFOREACH()

    LIST(REMOVE_DUPLICATES FILE_DISTRIB)

ENDMACRO(WRITE_GRAPH_MEMORY) 


##
## @end file WriteGraph.cmake 
##
