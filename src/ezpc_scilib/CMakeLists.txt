include_directories(${CMAKE_CURRENT_LIST_DIR})

find_package(Eigen3 3.3 QUIET NO_MODULE PATHS "${CMAKE_BINARY_DIR}/eigen" NO_DEFAULT_PATH)
if (NOT Eigen3_FOUND)
    message(STATUS "Eigen 3.3 was not found: clone and install Eigen3 locally")
    if (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/eigen/CMakeLists.txt")
        find_package(Git REQUIRED)
        message(STATUS "initialize Git submodule: extern/eigen")
        execute_process(COMMAND git submodule update --init --recursive extern/eigen
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
    endif ()
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory eigen
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
    execute_process(COMMAND ${CMAKE_COMMAND} ${PROJECT_SOURCE_DIR}/extern/eigen
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/eigen")
    find_package(Eigen3 3.3 REQUIRED NO_MODULE PATHS "${CMAKE_BINARY_DIR}/eigen" NO_DEFAULT_PATH)
endif()

find_package(Threads REQUIRED)

find_package(OpenMP REQUIRED)

add_subdirectory(Utils)
add_subdirectory(OT)
add_subdirectory(GC)
add_subdirectory(Millionaire)
add_subdirectory(BuildingBlocks)
add_subdirectory(NonLinear)
add_subdirectory(Math)
add_subdirectory(FixedPoint)


add_library(FASTST-EZPCLIB
    INTERFACE
)



target_link_libraries(FASTST-EZPCLIB
    INTERFACE
    FASTST-EZPCLIB-BuildingBlocks
    FASTST-EZPCLIB-FixedPoint
    FASTST-EZPCLIB-Millionaire
    FASTST-EZPCLIB-OT
    FASTST-EZPCLIB-Utils
    FASTST-EZPCLIB-GC
    FASTST-EZPCLIB-NonLinear
    FASTST-EZPCLIB-Math
)

target_include_directories(FASTST-EZPCLIB
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:./>
)





