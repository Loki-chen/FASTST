macro(add_test _name)
    add_executable(${_name}_test "${_name}_test.cpp")
    target_link_libraries(${_name}_test PUBLIC FASTST-SRC LIB_BERT)
endmacro()

add_test(RFCP)
add_test(FIT)
add_test(softmax)
add_test(layernorm)
add_test(gelu)
add_test(p-r) # Prime to ring
add_test(linear1)