set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(cmake/source_of_randomness.cmake)

FIND_PACKAGE(GMP REQUIRED)


add_library(FASTST-EZPCLIB
    fixed-point.cpp
    bool-data.cpp
)


target_link_libraries(FASTST-EZPCLIB
    PUBLIC
    ${GMP_LIBRARIES}
    # FASTST-UTILS
)


target_compile_options(FASTST-EZPCLIB
PUBLIC
    "-pthread;-maes;-msse4.1;-mavx;-mavx2;-faligned-new;-std=c++17;-O3;-Wno-parentheses-equality;-Wno-deprecated-declarations;-Wno-macro-redefined")

if(USE_RANDOM_DEVICE)
    target_compile_definitions(FASTST-EZPCLIB PUBLIC EMP_USE_RANDOM_DEVICE=1)
else(USE_RANDOM_DEVICE)
    target_compile_options(FASTST-EZPCLIB  PUBLIC "-mrdseed")
endif(USE_RANDOM_DEVICE)

target_include_directories(FASTST-EZPCLIB
     PUBLIC ${GMP_INCLUDE_DIR}
)