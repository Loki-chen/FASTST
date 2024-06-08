
add_library(FASTST-EZPCLIB-BuildingBlocks
    value-extension.cpp
    aux-protocols.cpp
    truncation.cpp
    linear-ot.cpp
    )



target_link_libraries(FASTST-EZPCLIB-BuildingBlocks
    PUBLIC
    FASTST-EZPCLIB-OT
    FASTST-EZPCLIB-GC
    FASTST-EZPCLIB-Millionaire
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
)


