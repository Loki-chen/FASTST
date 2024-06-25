
include_directories(${PROJECT_SOURCE_DIR}/src)

# add_library(Pro_TEST
#     pro_attn_test.cpp
#     pro_layer-norm_test.cpp
#     pro_ffn_test.cpp
#     pro_transformer_test.cpp
#     )


# cost_tets
macro(add_fix_func_test _name)
    add_executable(fix_${_name} "fixed_${_name}_test.cpp")
    target_link_libraries(fix_${_name} PUBLIC FASTST-SRC)
endmacro()

 
add_fix_func_test(comm)
add_fix_func_test(add)
add_fix_func_test(mul)
add_fix_func_test(load-data)
add_fix_func_test(mean)
add_fix_func_test(conversion) 