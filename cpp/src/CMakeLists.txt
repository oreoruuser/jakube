cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(Jakube
  DESCRIPTION "Approximate Nearest Neighbors Algorithm Library"
  VERSION 0.0.1
  LANGUAGES CXX)

add_library(Jakube INTERFACE)
add_library(Jakube::Jakube ALIAS Jakube)

foreach (HEADER jakubelib.h kissrandom.h)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/src/${HEADER}" "${CMAKE_CURRENT_BINARY_DIR}/include/jakube/${HEADER}" COPYONLY)
endforeach ()

target_include_directories(Jakube INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Install
include(GNUInstallDirs)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS Jakube
  EXPORT JakubeTargets)

install(EXPORT JakubeTargets
  FILE JakubeConfig.cmake
  NAMESPACE Jakube::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/jakube)

export(TARGETS Jakube NAMESPACE Jakube:: FILE JakubeConfig.cmake)

