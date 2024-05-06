cmake_minimum_required(VERSION 3.16)


project(FASTST)
set(NAME "FASTST")
project(FASTST LANGUAGES CXX)


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(INSTALL_DIR "${CMAKE_INSTALL_PREFIX}")
set(BUILD_DIR "${PROJECT_SOURCE_DIR}/build")



add_subdirectory(src)
option(FASTST-test "Build tests" ON)
if(FASTST-test)
  message(STATUS "Tests enabled")
  enable_testing()
  add_subdirectory(test)
endif()















