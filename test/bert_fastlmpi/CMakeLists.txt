add_library(BERT bert.cpp)

target_link_libraries(BERT
    PUBLIC
    FASTST-SRC
)

add_executable(bert_test "bert_test.cpp")
target_link_libraries(bert_test PUBLIC BERT)