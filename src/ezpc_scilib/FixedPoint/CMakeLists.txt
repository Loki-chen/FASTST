

add_library(FASTST-EZPCLIB-FixedPoint
    fixed-point.cpp
    bool-data.cpp
)


target_link_libraries(FASTST-EZPCLIB-FixedPoint
    PUBLIC
    # FASTST-EZPCLIB-BuildingBlocks
    FASTST-EZPCLIB-Math

)


