

add_library(FASTST-FIXED
    fixed_lib.cpp

)


target_link_libraries(FASTST-FIXED
    PUBLIC
    FASTST-UTILS
)
