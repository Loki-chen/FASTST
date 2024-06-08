
include_directories(${PROJECT_SOURCE_DIR}/src)

# add_library(Acc_TEST
#     acc_attn_test.cpp
#     acc_ffn_test.cpp
#     acc_layer-norm_test.cpp
#     )


# acc test
macro(add_acc_test _name)
    add_executable(acc_${_name} "acc_${_name}_test.cpp")
    target_link_libraries(acc_${_name} PUBLIC FASTST-SRC)
endmacro()

add_acc_test(attn)
add_acc_test(layer-norm)
add_acc_test(ffn)


# bfv_acc test
macro(add_bfv_acc_test _name)
    add_executable(bfv_acc_${_name} "bfv_acc_${_name}_test.cpp")
    target_link_libraries(bfv_acc_${_name} PUBLIC FASTST-SRC)
endmacro()


add_bfv_acc_test(layer-norm)



