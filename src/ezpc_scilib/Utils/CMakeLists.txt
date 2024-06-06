set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(cmake/source_of_randomness.cmake)

find_package(OpenSSL REQUIRED)
FIND_PACKAGE(GMP REQUIRED)

# find_package(emp-tool QUIET PATHS "${PROJECT_SOURCE_DIR}/build/" NO_DEFAULT_PATH)
# if (NOT emp-tool_FOUND)
#     message(STATUS "emptool was not found: clone and install emptool locally")
#     if (NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/EMP-TOOL/CMakeLists.txt")
#         find_package(Git REQUIRED)
#         message(STATUS "initialize Git submodule: extern/EMP-TOOL")
#         execute_process(COMMAND git submodule update --init --recursive extern/EMP-TOOL
#                 WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
#         execute_process(COMMAND git apply "${PROJECT_SOURCE_DIR}/cmake/emp-tool.patch"
#             WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/EMP-TOOL")    
#     endif ()
#     execute_process(COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/build .
#             WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/EMP-TOOL/")
#     execute_process(COMMAND ${CMAKE_COMMAND} --build . --target install
#             WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extern/EMP-TOOL/")

#     find_package(emp-tool REQUIRED PATHS "${PROJECT_SOURCE_DIR}/build/" NO_DEFAULT_PATH)

# endif()



add_library(FASTST-EZPCLIB-Utils INTERFACE)

target_compile_options(FASTST-EZPCLIB-Utils
    INTERFACE
    "-pthread;-maes;-msse4.1;-mavx;-mavx2;-faligned-new;-std=c++17;-O3;-Wno-parentheses-equality;-Wno-deprecated-declarations;-Wno-macro-redefined")
#     "-pthread;-Wall;-maes;-msse4.1;-mavx;-mavx2;-faligned-new;-std=c++17;-w;-g;-ggdb;-O0")


if(USE_RANDOM_DEVICE)
    target_compile_definitions(FASTST-EZPCLIB-Utils INTERFACE EMP_USE_RANDOM_DEVICE=1)
else(USE_RANDOM_DEVICE)
    target_compile_options(FASTST-EZPCLIB-Utils  INTERFACE "-mrdseed")
endif(USE_RANDOM_DEVICE)

target_include_directories(FASTST-EZPCLIB-Utils
     INTERFACE ${GMP_INCLUDE_DIR}
)

target_link_libraries(FASTST-EZPCLIB-Utils
    INTERFACE ${GMP_LIBRARIES}
    # FASTST-UTILS
)