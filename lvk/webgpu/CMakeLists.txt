# LightweightVK
#
# Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.27)

project(LVKWebGPU CXX)

file(GLOB SRC_FILES LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)
file(GLOB HEADER_FILES LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)

add_library(LVKWebGPU STATIC ${SRC_FILES} ${HEADER_FILES})

lvk_setup_target(LVKWebGPU)
lvk_set_folder(LVKWebGPU "LVK")

lvk_setup_groups("${SRC_FILES}")
lvk_setup_groups("${HEADER_FILES}")

# spirv-cross-wgsl with the changes for LVK (branch `lvk-webgpu` of https://github.com/corporateshark/spirv-cross-wgsl)
set(LVK_SPIRV_CROSS_WGSL_DIR "${LVK_ROOT_DIR}/third-party/deps/src/spirv-cross-wgsl")
if(NOT EXISTS "${LVK_SPIRV_CROSS_WGSL_DIR}/spirv_wgsl.cpp")
  message(FATAL_ERROR "${LVK_SPIRV_CROSS_WGSL_DIR} is missing. Run `deploy_deps.py` with LVK_WITH_WEBGPU=ON first.")
endif()

# cmake-format: off
set(SPIRV_CROSS_CLI            OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_TESTS   OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_SHARED         OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_STATIC         ON  CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_GLSL    ON  CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_WGSL    ON  CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_HLSL    OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_MSL     OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_CPP     OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_REFLECT OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_C_API   OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_ENABLE_UTIL    OFF CACHE BOOL "" FORCE)
set(SPIRV_CROSS_SKIP_INSTALL   ON  CACHE BOOL "" FORCE)
# cmake-format: on
add_subdirectory(${LVK_SPIRV_CROSS_WGSL_DIR} "spirv-cross-wgsl")
foreach(target spirv-cross-core spirv-cross-glsl spirv-cross-wgsl)
  lvk_set_folder(${target} "third-party/spirv-cross-wgsl")
endforeach()

# Dawn is a separate Release build of the shared library `webgpu_dawn`: its SPIRV-Tools, Vulkan headers and Abseil would
# clash with LVK's
set(LVK_DAWN_DIR "${LVK_ROOT_DIR}/third-party/deps/src/dawn")
if(NOT EXISTS "${LVK_DAWN_DIR}/third_party/abseil-cpp/CMakeLists.txt")
  message(FATAL_ERROR "${LVK_DAWN_DIR} is missing. Run `deploy_deps.py` with LVK_WITH_WEBGPU=ON first.")
endif()

include(ExternalProject)
set(LVK_DAWN_PREFIX "${CMAKE_BINARY_DIR}/dawn")
# cmake-format: off
set(LVK_DAWN_CMAKE_ARGS
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_INSTALL_PREFIX=${LVK_DAWN_PREFIX}"
    "-DCMAKE_INSTALL_LIBDIR=lib"
    "-DDAWN_BUILD_MONOLITHIC_LIBRARY=SHARED"
    "-DDAWN_ENABLE_INSTALL=ON"
    "-DDAWN_FETCH_DEPENDENCIES=OFF"
    "-DDAWN_BUILD_SAMPLES=OFF"
    "-DDAWN_BUILD_TESTS=OFF"
    "-DDAWN_BUILD_PROTOBUF=OFF"
    "-DDAWN_USE_GLFW=OFF"
    "-DDAWN_ENABLE_D3D11=OFF"
    "-DDAWN_ENABLE_DESKTOP_GL=OFF"
    "-DDAWN_ENABLE_OPENGLES=OFF"
    "-DTINT_BUILD_CMD_TOOLS=OFF"
    "-DTINT_BUILD_TESTS=OFF"
    "-DTINT_BUILD_IR_BINARY=OFF"
    "-DTINT_BUILD_GLSL_WRITER=OFF"
    "-DTINT_BUILD_GLSL_VALIDATOR=OFF")
# cmake-format: on
if(Python3_EXECUTABLE)
  list(APPEND LVK_DAWN_CMAKE_ARGS "-DPython3_EXECUTABLE=${Python3_EXECUTABLE}")
endif()

if(WIN32)
  # otherwise Dawn looks for vulkan-1.dll next to webgpu_dawn.dll only (and fails with ERROR_INVALID_PARAMETER)
  list(APPEND LVK_DAWN_CMAKE_ARGS "-DDAWN_FORCE_SYSTEM_COMPONENT_LOAD=ON")
  set(LVK_DAWN_LIBRARY "${LVK_DAWN_PREFIX}/bin/webgpu_dawn.dll")
  set(LVK_DAWN_IMPLIB "${LVK_DAWN_PREFIX}/lib/webgpu_dawn.lib")
elseif(APPLE)
  set(LVK_DAWN_LIBRARY "${LVK_DAWN_PREFIX}/lib/libwebgpu_dawn.dylib")
else()
  set(LVK_DAWN_LIBRARY "${LVK_DAWN_PREFIX}/lib/libwebgpu_dawn.so")
endif()

# a bare --parallel means unlimited jobs with Makefiles
cmake_host_system_information(RESULT LVK_NUM_CORES QUERY NUMBER_OF_LOGICAL_CORES)
ExternalProject_Add(Dawn
  SOURCE_DIR "${LVK_DAWN_DIR}"
  BINARY_DIR "${LVK_DAWN_PREFIX}/build"
  CMAKE_ARGS ${LVK_DAWN_CMAKE_ARGS}
  BUILD_ALWAYS ON
  BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release --parallel ${LVK_NUM_CORES}
  INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR> --config Release
  BUILD_BYPRODUCTS "${LVK_DAWN_LIBRARY}" ${LVK_DAWN_IMPLIB}
  # MSBuild takes Dawn's status line "-- Dawn enable error injection: OFF" for an error and fails the build
  LOG_CONFIGURE ON
  LOG_OUTPUT_ON_FAILURE ON
  USES_TERMINAL_BUILD ON)
lvk_set_folder(Dawn "third-party/dawn")

# the include directory of an imported target must exist at configure time
file(MAKE_DIRECTORY "${LVK_DAWN_PREFIX}/include")
add_library(LVKDawn SHARED IMPORTED GLOBAL)
set_target_properties(LVKDawn PROPERTIES IMPORTED_LOCATION "${LVK_DAWN_LIBRARY}"
                                         INTERFACE_INCLUDE_DIRECTORIES "${LVK_DAWN_PREFIX}/include"
                                         INTERFACE_COMPILE_DEFINITIONS "WGPU_SHARED_LIBRARY")
if(WIN32)
  set_target_properties(LVKDawn PROPERTIES IMPORTED_IMPLIB "${LVK_DAWN_IMPLIB}")
endif()
add_dependencies(LVKWebGPU Dawn)
target_link_libraries(LVKWebGPU PUBLIC LVKDawn)

target_include_directories(LVKWebGPU PRIVATE ${LVK_SPIRV_CROSS_WGSL_DIR})
target_compile_definitions(LVKWebGPU PUBLIC "LVK_WITH_WEBGPU=1")

# LVKLibrary: the backend-independent parts of LVK.h
target_link_libraries(LVKWebGPU PUBLIC LVKLibrary)
# the shader compilers are linked directly, without the Vulkan backend's helpers
target_link_libraries(LVKWebGPU PRIVATE spirv-cross-wgsl spirv-cross-glsl spirv-cross-core)
target_link_libraries(LVKWebGPU PRIVATE glslang::glslang glslang::SPIRV glslang::glslang-default-resource-limits)

if(LVK_WITH_SLANG)
  target_compile_definitions(LVKWebGPU PRIVATE "LVK_WITH_SLANG=1")
  target_link_libraries(LVKWebGPU PRIVATE slang)
  target_include_directories(LVKWebGPU PRIVATE "${LVK_ROOT_DIR}/third-party/deps/src/slang/include")
endif()

# executables using LVKWebGPU need webgpu_dawn.dll and the Slang DLLs next to them on Windows
function(lvk_copy_webgpu_runtime_dlls target)
  if(NOT WIN32)
    return()
  endif()
  add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${target}>
                                                         $<TARGET_FILE_DIR:${target}> COMMAND_EXPAND_LISTS)
  if(LVK_WITH_SLANG)
    # the whole set: TARGET_RUNTIME_DLLS misses some, and mixing Slang DLLs from different builds crashes
    add_custom_command(TARGET ${target} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE_DIR:slang>/slang.dll" "$<TARGET_FILE_DIR:slang>/slang-compiler.dll"
                               "$<TARGET_FILE_DIR:slang>/slang-glsl-module.dll" "$<TARGET_FILE_DIR:slang>/slang-rt.dll" $<TARGET_FILE_DIR:${target}>)
  endif()
endfunction()
