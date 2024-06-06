

add_library(FASTST-FIXED
    fixed-point.cpp
    bool-data.cpp
)


target_link_libraries(FASTST-FIXED
    PUBLIC
    FASTST-UTILS
)
