cmake_minimum_required(VERSION 3.28)
project(motor CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/O2 /DNDEBUG)
else()
    add_compile_options(-O3 -DNDEBUG -flto)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

# Copy nnue.bin to build directory
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/nnue.bin
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/nnue.bin ${CMAKE_CURRENT_BINARY_DIR}/nnue.bin
    DEPENDS ${CMAKE_SOURCE_DIR}/nnue.bin
    COMMENT "Copying nnue.bin to build directory"
)

add_custom_target(copy_nnue_bin ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/nnue.bin)

add_executable(motor main.cpp)
add_dependencies(motor copy_nnue_bin)
