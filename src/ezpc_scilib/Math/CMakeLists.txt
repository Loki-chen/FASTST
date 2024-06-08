

add_library(FASTST-EZPCLIB-Math
    math-functions.cpp
)


target_link_libraries(FASTST-EZPCLIB-Math
    PUBLIC 
    FASTST-EZPCLIB-NonLinear
    FASTST-EZPCLIB-BuildingBlocks
)


