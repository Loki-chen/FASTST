add_library(FASTST-EZPCLIB-GC
    emp-tool.cpp
    halfgate_gen.cpp
    halfgate_eva.cpp)
target_link_libraries(FASTST-EZPCLIB-GC
    PUBLIC 
    FASTST-EZPCLIB-Utils 
    FASTST-EZPCLIB-OT
)
