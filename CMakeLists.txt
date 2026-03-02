cmake_minimum_required(VERSION 4.0)
project(memlayout LANGUAGES CXX)

option(BUILD_TESTING "Build unit tests" OFF)
option(BUILD_DOCS "Build documentation" OFF)

add_subdirectory(memlayout)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

if(BUILD_DOCS)
  add_subdirectory(docs)
endif()
