set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

include(FetchContent)
FetchContent_Declare(
    googletest
    DOWNLOAD_EXTRACT_TIMESTAMP NEW
    URL https://github.com/google/googletest/archive/52eb8108c5bdec04579160ae17225d66034bd723.zip
)
FetchContent_MakeAvailable(googletest)
enable_testing()
include(GoogleTest)

add_subdirectory(cpu)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND OR CMAKE_CUDA_COMPILER)
    add_subdirectory(gpu)
else()
    message(STATUS "CUDA not found - skipping GPU tests")
endif()
