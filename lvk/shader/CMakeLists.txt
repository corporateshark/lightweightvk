# LightweightVK
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.16)

project(LVKShaderCompile CXX C)

file(GLOB SRC_FILES LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.c)
file(GLOB HEADER_FILES LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)

# TODO: Hack to get CMake to put DLLs alongside our executable. This isn't really a solution.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_library(LVKShaderCompile SHARED ${SRC_FILES} ${HEADER_FILES})

lvk_setup_target(LVKShaderCompile)
lvk_set_folder(LVKShaderCompile "LVK")

lvk_setup_groups("${SRC_FILES}")
lvk_setup_groups("${HEADER_FILES}")

# glslang
# cmake-format: off
set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "")
set(ENABLE_HLSL OFF CACHE BOOL "")
set(ENABLE_CTEST OFF CACHE BOOL "")
set(ENABLE_OPT OFF CACHE BOOL "")
set(ENABLE_SPVREMAPPER OFF CACHE BOOL "")
set(SKIP_GLSLANG_INSTALL ON CACHE BOOL "")
add_subdirectory(${LVK_ROOT_DIR}/third-party/deps/src/glslang "glslang")
lvk_set_folder(GenericCodeGen "third-party/glslang")
lvk_set_folder(glslang "third-party/glslang")
lvk_set_folder(MachineIndependent "third-party/glslang")
lvk_set_folder(OGLCompiler "third-party/glslang")
lvk_set_folder(OSDependent "third-party/glslang")
lvk_set_folder(SPIRV "third-party/glslang")
lvk_set_folder(glslang-default-resource-limits "third-party/glslang")
# cmake-format: on

target_link_libraries(LVKShaderCompile PRIVATE glslang SPIRV glslang-default-resource-limits)
target_link_libraries(LVKShaderCompile PRIVATE Vulkan::Headers)

target_include_directories(LVKShaderCompile PUBLIC "${LVK_ROOT_DIR}/third-party/deps/src/")