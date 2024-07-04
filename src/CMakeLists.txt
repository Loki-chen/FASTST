include_directories(${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(utils)
add_subdirectory(protocols)
add_subdirectory(ezpc_scilib)
add_library(FASTST-SRC
  transformer.cpp
  fixed-transformer.cpp
)


target_link_libraries(FASTST-SRC
    PUBLIC
    FASTST-UTILS
    FASTST-PROTOCOLS
    FASTST-EZPCLIB
)

target_include_directories(FASTST-SRC
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:./>
)

install(DIRECTORY
        FASTST-SRC
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp")





