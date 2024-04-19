include_directories(${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(utils)

add_library(FASTST-SRC
  INTERFACE
)


target_link_libraries(FASTST-SRC
    INTERFACE

    FASTST-UTILS
    
)

target_include_directories(FASTST-SRC
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:./>
)



# install(FILES
#         "utils.h"
#         DESTINATION include)

install(DIRECTORY
        FASTST-SRC
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp")





