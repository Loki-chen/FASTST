


macro(add_test_protocols _name)
    add_executable(test-${_name} "test_${_name}.cpp")
    target_link_libraries(test-${_name} PUBLIC FASTST-SRC)
endmacro()

add_test_protocols(attn)

