###
#
# @file CMakeLists.txt
#
# CMake Information
# PLASMA is a software package provided by Univ. of Tennessee,
# Univ. of California Berkeley and Univ. of Colorado Denver
#
# @version 2.1.0
# @author Asim YarKhan
# @date 2010-09-20
#
###

project( QUARK )
cmake_minimum_required( VERSION 2.6 )
set( CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS TRUE)

set( QUARK_VERSION_MAJOR "1" )
set( QUARK_VERSION_MINOR "0" )
set( QUARK_VERSION_PATCH "0" )
set( QUARK_VERSION "${PLASMA_VERSION_MAJOR}.${PLASMA_VERSION_MINOR}.${PLASMA_VERSION_PATCH}" )

# Disallow in-source build
string( COMPARE EQUAL "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}" INSOURCE )
if( INSOURCE )
  message( FATAL_ERROR "\n"
  "This project requires an out-of-source-directory "
  "build. Please create a separate build directory and "
  "use CMake to configure and build there. \n"
  "e.g. mkdir build_dir; cd build_dir; cmake ..; make" )
endif()

# Release build is the default
if( NOT DEFINED CMAKE_BUILD_TYPE )
  set(CMAKE_BUILD_TYPE "Release")
endif()

# No shared libraries by default
option( BUILD_SHARED_LIBS "Build shared libraries" OFF )

# Enable language compilers used in this project
enable_language( Fortran C )


set( quark_SRCS icl_hash.c icl_list.c quarkos.c quark.c )
set( quark_HEADERS quark.h quark_unpack_args.h icl_hash.h icl_list.h quarkwinthread.h bsd_queue.h bsd_tree.h quark_trace.h )

## This needs to be enabled if QUARK is being used outside of PLASMA
## This simply duplicates the Windows pthreads implementation in plasmawinthread.c.
#if(WIN32)
#  set( quark_SRCS ${quark_SRCS} quarkwinthread.c )
#endif( WIN32 )

add_library( quark ${quark_SRCS} )

INSTALL( TARGETS quark ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin )
INSTALL( FILES ${quark_HEADERS} DESTINATION quark )
#INSTALL( FILES ${quark_SRCS} ${quark_HEADERS} quarkwinthread.c quarkos-hwloc.c CMakeLists.txt LICENSE DESTINATION quark )

