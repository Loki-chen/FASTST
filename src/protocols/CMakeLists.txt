


add_library(FASTST-PROTOCOLS
    attention.cpp
    ffn.cpp

    layer-norm1.cpp
    layer-norm2.cpp
)




target_link_libraries(FASTST-PROTOCOLS
    PUBLIC
    FASTST-SRC
)
