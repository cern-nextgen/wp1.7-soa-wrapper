add_library(Wrapper INTERFACE)

target_include_directories(
    Wrapper INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

add_library(memlayout::Wrapper ALIAS Wrapper)

install(
    TARGETS Wrapper
    EXPORT memlayoutTargets
    INCLUDES DESTINATION include
)

install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/wrapper.h
    DESTINATION include/memlayout
)

file(
    WRITE
    "${CMAKE_CURRENT_BINARY_DIR}/memlayoutConfig.cmake"
    "include(\"\${CMAKE_CURRENT_LIST_DIR}/memlayoutTargets.cmake\")\n"
)

install(
    EXPORT memlayoutTargets
    FILE memlayoutTargets.cmake
    NAMESPACE memlayout::
    DESTINATION lib/cmake/memlayout
)

install(
    FILES
    "${CMAKE_CURRENT_BINARY_DIR}/memlayoutConfig.cmake"
    DESTINATION lib/cmake/memlayout
)