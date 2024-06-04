set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(cmake/source_of_randomness.cmake)



find_package(SEAL 4.1.1 EXACT QUIET PATHS "${PROJECT_SOURCE_DIR}/build/" NO_DEFAULT_PATH)
if (NOT SEAL_FOUND)
    message(STATUS "SEAL 4.1.1 was not found: clone and install SEAL locally")
    if (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/SEAL/CMakeLists.txt")
        find_package(Git REQUIRED)
        message(STATUS "initialize Git submodule: extern/SEAL")
        execute_process(COMMAND git submodule update --init --recursive extern/SEAL
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
        execute_process(COMMAND git apply "${PROJECT_SOURCE_DIR}/cmake/seal.patch"
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL")
    endif ()
    execute_process(COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/build .
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL/")
    execute_process(COMMAND ${CMAKE_COMMAND} --build . --target install
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/SEAL/")
    find_package(SEAL 4.1.1 EXACT REQUIRED PATHS "${PROJECT_SOURCE_DIR}/build/" NO_DEFAULT_PATH)
endif()

FIND_PACKAGE(GMP REQUIRED)


FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("-- OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


add_library(FASTST-UTILS
    he-tools.cpp
    he-bfv.cpp
    mat-tools.cpp
    io.cpp 
)

target_compile_options(FASTST-UTILS
PUBLIC
    "-pthread;-maes;-msse4.1;-mavx;-mavx2;-faligned-new;-std=c++17;-O3;-Wno-parentheses-equality;-Wno-deprecated-declarations;-Wno-macro-redefined")

if(USE_RANDOM_DEVICE)
    target_compile_definitions(FASTST-UTILS PUBLIC EMP_USE_RANDOM_DEVICE=1)
else(USE_RANDOM_DEVICE)
    target_compile_options(FASTST-UTILS  PUBLIC "-mrdseed")
endif(USE_RANDOM_DEVICE)

target_include_directories(FASTST-UTILS
     PUBLIC ${GMP_INCLUDE_DIR}
)

target_link_libraries(FASTST-UTILS
    PUBLIC
    SEAL::seal
    FASTST-SRC  
    OpenMP::OpenMP_CXX
    ${GMP_LIBRARIES}
)
