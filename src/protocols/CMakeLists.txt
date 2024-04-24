


add_library(FASTST-PROTOCOLS
    attention.cpp
    layer_norm1.cpp
    ffn.cpp
    layer_norm2.cpp
)




target_link_libraries(FASTST-PROTOCOLS
    PUBLIC
    FASTST-SRC
)
