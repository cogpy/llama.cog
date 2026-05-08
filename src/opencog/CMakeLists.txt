llama_add_compile_flags()

#
# OpenCog cognitive architecture
#
# Static library housing the OpenCog-inspired cognitive components
# (AtomSpace, cognitive cycle, LLM inference engine, embodied reasoning).
# Kept separate from the core llama library so consumers that do not need
# the cognitive architecture do not pay for it. Only targets that opt in
# (currently the llama-cognitive example) link against this target.

add_library(opencog STATIC
            atomspace.cpp
            cognitive_cycle.cpp
            llm_inference.cpp
            opencog.cpp
            )

target_include_directories(opencog PUBLIC ../../include)
target_include_directories(opencog PRIVATE ..)
target_compile_features   (opencog PRIVATE cxx_std_17)

target_link_libraries(opencog PUBLIC llama ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(opencog PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
