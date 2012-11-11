###
#
#  @file infoMAGMA_MORSE.cmake 
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

###
#   
#   
#   
###     
MACRO(MAGMA_MORSE_INFO_DEPS)
    # Define the direct dependencies of the library
    # ---------------------------------------------
    UNSET(MAGMA_MORSE_DIRECT_DEPS)
    LIST(APPEND MAGMA_MORSE_DIRECT_DEPS "LAPACKE")
    LIST(APPEND MAGMA_MORSE_DIRECT_DEPS "CBLAS"  )
    LIST(APPEND MAGMA_MORSE_DIRECT_DEPS "STARPU" )
    LIST(APPEND MAGMA_MORSE_DIRECT_DEPS "QUARK"  )
    LIST(APPEND MAGMA_MORSE_DIRECT_DEPS "TMG"    )
    IF(NOT MORSE_SCHED_QUARK AND NOT "^${MORSE_USE_QUARK}$" STREQUAL "^ON$")
        LIST(APPEND MAGMA_MORSE_DIRECT_DEPS "MAGMA"  )
    ENDIF()

    # Define the priority of dependencies
    # -----------------------------------
    INCLUDE(BuildSystemTools)
    DEFINE_PRIORITY("MAGMA_MORSE" "LAPACKE"   "depends" "depends"  "depends")
    DEFINE_PRIORITY("MAGMA_MORSE" "CBLAS"     "depends" "depends"  "depends")

    IF(MORSE_ENABLE_TESTING)
        DEFINE_PRIORITY("MAGMA_MORSE" "TMG"   "depends" "depends"  "depends"   )
    ELSE(MORSE_ENABLE_TESTING)
        DEFINE_PRIORITY("MAGMA_MORSE" "TMG"   "depends" "depends"  "recommends")
    ENDIF(MORSE_ENABLE_TESTING)

    IF(MAGMA)
        DEFINE_PRIORITY("MAGMA_MORSE" "MAGMA" "depends" "depends"  "depends"   )
    ELSE(MAGMA)
        DEFINE_PRIORITY("MAGMA_MORSE" "MAGMA" "depends" "suggests" "recommends")
    ENDIF(MAGMA)

    IF(NOT MORSE_SCHED_STARPU AND NOT MORSE_SCHED_QUARK)
        # if nothing is declare (for example during cpack or default option),
        # we set priority accoording to main CMakeLists.txt
        SET(MAGMA_MORSE_STARPU_PRIORITY       "depends" )
        SET(MAGMA_MORSE_QUARK_PRIORITY        "suggests")
    ELSE()
        # by default, we set to "suggests" and we increase priority
        # according to user's requests
        SET(MAGMA_MORSE_QUARK_PRIORITY        "suggests")
        SET(MAGMA_MORSE_STARPU_PRIORITY       "suggests")

        # for StarPU
        IF(MORSE_SCHED_STARPU OR "^${MORSE_USE_STARPU}$" STREQUAL "^ON$")
            SET(MAGMA_MORSE_STARPU_PRIORITY   "depends" )
        ENDIF()

        # for QUARK
        IF(MORSE_SCHED_QUARK OR "^${MORSE_USE_QUARK}$" STREQUAL "^ON$")
            SET(MAGMA_MORSE_QUARK_PRIORITY    "depends" )
        ENDIF()
    ENDIF()

ENDMACRO(MAGMA_MORSE_INFO_DEPS)

### 
#   
#
#   
### 
MACRO(MAGMA_MORSE_INFO_INSTALL)
    
ENDMACRO(MAGMA_MORSE_INFO_INSTALL)

##
## @end file infoMAGMA_MORSE.cmake 
##
