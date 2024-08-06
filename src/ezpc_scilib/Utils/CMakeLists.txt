set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(cmake/source_of_randomness.cmake)

find_package(OpenSSL REQUIRED)
FIND_PACKAGE(GMP REQUIRED)

add_library(FASTST-EZPCLIB-Utils INTERFACE)

target_compile_options(FASTST-EZPCLIB-Utils
    INTERFACE
    "-pthread;-maes;-msse4.1;-mavx;-mavx2;-faligned-new;-std=c++17;-O3;;-Wno-deprecated-declarations;")
#    "-pthread;-maes;-msse4.1;-mavx;-mavx2;-faligned-new;-std=c++17;-O3;-Wno-parentheses-equality;-Wno-deprecated-declarations;-Wno-macro-redefined")
#     "-pthread;-Wall;-maes;-msse4.1;-mavx;-mavx2;-faligned-new;-std=c++17;-w;-g;-ggdb;-O0")


if(USE_RANDOM_DEVICE)
    target_compile_definitions(FASTST-EZPCLIB-Utils INTERFACE EMP_USE_RANDOM_DEVICE=1)
else(USE_RANDOM_DEVICE)
    target_compile_options(FASTST-EZPCLIB-Utils  INTERFACE "-mrdseed")
endif(USE_RANDOM_DEVICE)

target_include_directories(FASTST-EZPCLIB-Utils
     INTERFACE ${OPENSSL_INCLUDE_DIR} ${GMP_INCLUDE_DIR}
)

target_link_libraries(FASTST-EZPCLIB-Utils
    INTERFACE ${OPENSSL_LIBRARIES} ${GMP_LIBRARIES}

)