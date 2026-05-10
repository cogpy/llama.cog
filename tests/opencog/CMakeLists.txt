llama_add_compile_flags()

set(OPENCOG_TESTS
    test-atomspace
    test-cognitive-cycle
    test-embodied-reasoning
    test-prompt-engine
    test-cognitive-system
)

foreach(test_target IN LISTS OPENCOG_TESTS)
    add_executable(${test_target} ${test_target}.cpp)
    target_link_libraries(${test_target} PRIVATE opencog)
    install(TARGETS ${test_target} RUNTIME)

    add_test(
        NAME ${test_target}
        WORKING_DIRECTORY .
        COMMAND $<TARGET_FILE:${test_target}>)
    set_property(TEST ${test_target} PROPERTY LABELS opencog)
endforeach()
