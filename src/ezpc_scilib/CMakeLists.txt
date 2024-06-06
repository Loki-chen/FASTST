include_directories(${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(BuildingBlocks)
add_subdirectory(FixedPoint)
add_subdirectory(GC)
add_subdirectory(Math)
add_subdirectory(Millionaire)
add_subdirectory(OT)
add_subdirectory(Utils)



add_library(FASTST-EZPCLIB
    INTERFACE
)


target_link_libraries(FASTST-EZPCLIB
    INTERFACE
    FASTST-EZPCLIB-Utils
)

target_include_directories(FASTST-EZPCLIB
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:./>
)

