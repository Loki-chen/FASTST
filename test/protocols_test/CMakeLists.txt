
include_directories(${PROJECT_SOURCE_DIR}/src)

# add_library(Pro_TEST
#     pro_attn_test.cpp
#     pro_layer-norm_test.cpp
#     pro_ffn_test.cpp
#     pro_transformer_test.cpp
#     )


# cost_tets
macro(add_pro_test _name)
    add_executable(pro_${_name} "pro_${_name}_test.cpp")
    target_link_libraries(pro_${_name} PUBLIC FASTST-SRC)
endmacro()
add_pro_test(attn)
add_pro_test(layer-norm)
add_pro_test(ffn)
add_pro_test(transformer)


