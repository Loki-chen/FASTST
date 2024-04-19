


macro(add_test_att _name)
    add_executable(${_name}-att "test_sec_${_name}.cpp")
    target_link_libraries(${_name}-att PUBLIC FASTST-SRC)
endmacro()

add_test_att(trans)


