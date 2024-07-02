


add_library(FASTST-PROTOCOLS
    attention.cpp
    ffn.cpp
    layer-norm.cpp
    fixed-attention.cpp
    fixed-layer-norm.cpp
)
 
 
target_link_libraries(FASTST-PROTOCOLS
    PUBLIC
    FASTST-UTILS
    FASTST-EZPCLIB
)
