include("ggml/cmake/common.cmake")

function(llama_add_compile_flags)
    if (LLAMA_FATAL_WARNINGS)
        if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            list(APPEND C_FLAGS   -Werror)
            list(APPEND CXX_FLAGS -Werror)
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            add_compile_options(/WX)
            # MSVC under /WX promotes a long list of pervasive narrowing /
            # data-loss warnings on llama.cpp upstream code (size_t -> int,
            # int64_t -> int, double -> float, etc.) to errors. Match the
            # suppressions that ggml/CMakeLists.txt already applies to its
            # own targets so /WX still catches genuine errors without
            # requiring per-line casts in every upstream file.
            add_compile_options(
                /wd4005  # Macro redefinition
                /wd4065  # 'switch' contains 'default' but no 'case' labels
                /wd4101  # Unreferenced local variable (common in 'catch' clauses)
                /wd4127  # Conditional expression is constant
                /wd4189  # Local variable initialised but not referenced
                /wd4244  # Conversion, possible loss of data
                /wd4267  # 'size_t' to a smaller type, possible loss of data
                /wd4305  # 'type1' to 'type2', possible loss of data
                /wd4566  # 'char' to 'wchar_t', possible loss of data
                /wd4996  # POSIX deprecation warnings
            )
        endif()
    endif()

    if (LLAMA_ALL_WARNINGS)
        if (NOT MSVC)
            list(APPEND C_FLAGS -Wshadow -Wstrict-prototypes -Wpointer-arith -Wmissing-prototypes
                                -Werror=implicit-int -Werror=implicit-function-declaration)

            list(APPEND CXX_FLAGS -Wmissing-declarations -Wmissing-noreturn)

            list(APPEND WARNING_FLAGS -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function)

            list(APPEND C_FLAGS   ${WARNING_FLAGS})
            list(APPEND CXX_FLAGS ${WARNING_FLAGS})

            ggml_get_flags(${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})

            add_compile_options("$<$<COMPILE_LANGUAGE:C>:${C_FLAGS};${GF_C_FLAGS}>"
                                "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS};${GF_CXX_FLAGS}>")
        else()
            # todo : msvc
            set(C_FLAGS   "" PARENT_SCOPE)
            set(CXX_FLAGS "" PARENT_SCOPE)
        endif()
    endif()
endfunction()
