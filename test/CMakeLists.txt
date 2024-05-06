
include_directories(${PROJECT_SOURCE_DIR}/src)

# toy test
macro(add_test_protocols _name)
    add_executable(test-${_name} "test_${_name}.cpp")
    target_link_libraries(test-${_name} PUBLIC FASTST-SRC)
endmacro()

add_test_protocols(attn)



# acc test
macro(add_acc_test _name)
    add_executable(acc_${_name} "acc_${_name}_test.cpp")
    target_link_libraries(acc_${_name} PUBLIC FaST)
endmacro()

add_acc_test(attn)
add_acc_test(layer-norm1)
add_acc_test(ffn)


# cost_tets
macro(add_pro_test _name)
    add_executable(pro_${_name} "pro_${_name}_test.cpp")
    target_link_libraries(pro_${_name} PUBLIC FaST)
endmacro()
add_pro_test(attn)
add_pro_test(layer-norm1)
add_pro_test(ffn)
add_pro_test(transformer)