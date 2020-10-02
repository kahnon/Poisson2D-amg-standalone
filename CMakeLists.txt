cmake_minimum_required(VERSION 3.11...3.17)

#project description
project(amg_test
  VERSION 0.1
  DESCRIPTION "First Test of axisymmetric 2D AMG Poisson solver"
  LANGUAGES CXX
)

#prevent in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds not allowed. Use -B option for out-of-source build.")
endif()

#put all executables and libraries into a directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

set( CMAKE_BUILD_TYPE DEBUG )
option( USE_ADDRESS_SANITIZER on)

set( CMAKE_CXX_STANDARD 20 )
set( CMAKE_CXX_FLAGS "-fopenmp -Wall -pedantic-errors -march=native -pipe" )
set( CMAKE_CXX_FLAGS_RELEASE "-O3 -fno-exceptions -fno-rtti -DNDEBUG" )
set( CMAKE_CXX_FLAGS_DEBUG "-Og -D_GLIBCXX_ASSERTIONS -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC" )

if ( ${USE_ADDRESS_SANITIZER} )
  set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address" )
endif()

add_executable(bin_amg amg.cpp)
#target_include_directories( bin_amg_test )
#target_link_libraries( bin_amg_test )

