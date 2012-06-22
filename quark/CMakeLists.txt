###
#
# @file          : CMakeLists.txt
#
# @description   : Project MAGMA
#
# @version       :
# @created by    : Cedric Castagnede
# @creation date : 25-01-2012
# @last modified : mer. 16 mai 2012 11:36:11 CEST
#
###

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# Define the list of sources
# --------------------------
SET(QUARK_SRCS icl_hash.c
               icl_list.c
               quarkos.c
               quark.c
               )

# Compile step
# ------------
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)
ADD_LIBRARY(quark ${QUARK_SRCS})
SET_PROPERTY(TARGET quark PROPERTY LINKER_LANGUAGE Fortran)

# Install step
# ------------
INSTALL(TARGETS quark
        DESTINATION lib)

###
### CMakeLists.txt
###
