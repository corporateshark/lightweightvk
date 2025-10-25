# LightweightVK
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

project(LVKVulkan CXX C)

file(GLOB SRC_FILES LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp *.c)
file(GLOB HEADER_FILES LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)

add_library(LVKVulkan STATIC ${SRC_FILES} ${HEADER_FILES})

lvk_setup_target(LVKVulkan)
lvk_set_folder(LVKVulkan "LVK")

lvk_setup_groups("${SRC_FILES}")
lvk_setup_groups("${HEADER_FILES}")

# glslang
# cmake-format: off
set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "")
set(ENABLE_HLSL             OFF CACHE BOOL "")
set(ENABLE_CTEST            OFF CACHE BOOL "")
set(ENABLE_OPT              OFF CACHE BOOL "")
set(ENABLE_SPVREMAPPER      OFF CACHE BOOL "")
set(SKIP_GLSLANG_INSTALL    ON  CACHE BOOL "")
add_subdirectory(${LVK_ROOT_DIR}/third-party/deps/src/glslang "glslang")
lvk_set_folder(GenericCodeGen     "third-party/glslang")
lvk_set_folder(glslang            "third-party/glslang")
lvk_set_folder(MachineIndependent "third-party/glslang")
lvk_set_folder(OSDependent        "third-party/glslang")
lvk_set_folder(SPIRV              "third-party/glslang")
lvk_set_folder(glslang-default-resource-limits "third-party/glslang")
# cmake-format: on

# slang
# cmake-format: off
if(LVK_WITH_SLANG)
  set(SLANG_ENABLE_CUDA          OFF CACHE BOOL "")
  set(SLANG_ENABLE_OPTIX         OFF CACHE BOOL "")
  set(SLANG_ENABLE_NVAPI         OFF CACHE BOOL "")
  set(SLANG_ENABLE_XLIB          OFF CACHE BOOL "")
  set(SLANG_ENABLE_AFTERMATH     OFF CACHE BOOL "")
  set(SLANG_ENABLE_DX_ON_VK      OFF CACHE BOOL "")
  set(SLANG_ENABLE_GFX           OFF CACHE BOOL "")
  set(SLANG_ENABLE_SLANGC        OFF CACHE BOOL "")
  set(SLANG_ENABLE_SLANGRT       ON  CACHE BOOL "")
  set(SLANG_ENABLE_SLANG_GLSLANG OFF CACHE BOOL "")
  set(SLANG_ENABLE_TESTS         OFF CACHE BOOL "")
  set(SLANG_ENABLE_EXAMPLES      OFF CACHE BOOL "")
  set(SLANG_ENABLE_REPLAYER      OFF CACHE BOOL "")
  set(SLANG_ENABLE_PREBUILT_BINARIES OFF CACHE BOOL "")
  set(SLANG_EMBED_CORE_MODULE_SOURCE OFF CACHE BOOL "")
  set(SLANG_ENABLE_DXIL              OFF CACHE BOOL "")
  set(SLANG_RHI_ENABLE_D3D11         OFF CACHE BOOL "")
  set(SLANG_RHI_ENABLE_D3D12         OFF CACHE BOOL "")
  set(SLANG_RHI_ENABLE_METAL         OFF CACHE BOOL "")
  set(SLANG_RHI_ENABLE_CUDA          OFF CACHE BOOL "")
  set(SLANG_RHI_ENABLE_WGPU          OFF CACHE BOOL "")
  add_subdirectory(${LVK_ROOT_DIR}/third-party/deps/src/slang "slang")
  lvk_set_folder(miniz                  "third-party")
  lvk_set_folder(lz4_static             "third-party")
  lvk_set_folder(compiler-core          "third-party/slang")
  lvk_set_folder(core                   "third-party/slang")
  lvk_set_folder(slang                  "third-party/slang")
  lvk_set_folder(slangd                 "third-party/slang")
  lvk_set_folder(slang-rt               "third-party/slang")
  lvk_set_folder(slang-rhi              "third-party/slang")
  lvk_set_folder(slang-rhi-copy-files   "third-party/slang")
  lvk_set_folder(slang-rhi-resources    "third-party/slang")
  lvk_set_folder(slang-rhi-tests        "third-party/slang")
endif()
# cmake-format: on

# SPIRV-Reflect
set(SPIRV_REFLECT_EXECUTABLE OFF CACHE BOOL "")
set(SPIRV_REFLECT_STATIC_LIB ON  CACHE BOOL "")
add_subdirectory(${LVK_ROOT_DIR}/third-party/deps/src/SPIRV-Reflect "SPIRV-Reflect")
lvk_set_folder(spirv-reflect-static "third-party")

if(NOT LVK_USE_CUSTOM_MOLTENVK)
  find_package(Vulkan REQUIRED)
endif()

if(LVK_WITH_VULKAN_PORTABILITY)
  add_definitions("-DLVK_WITH_VULKAN_PORTABILITY")
endif()

target_link_libraries(LVKVulkan INTERFACE LVKLibrary)
target_link_libraries(LVKVulkan PRIVATE glslang SPIRV glslang-default-resource-limits)
target_link_libraries(LVKVulkan PRIVATE spirv-reflect-static)
target_link_libraries(LVKVulkan INTERFACE minilog)
if(LVK_WITH_SLANG)
  target_link_libraries(LVKVulkan PRIVATE slang)
  target_link_libraries(LVKVulkan PRIVATE slang-rt)
endif()

if(LVK_USE_CUSTOM_MOLTENVK)
  target_include_directories(LVKVulkan PUBLIC "${LVK_CUSTOM_MOLTENVK_PATH}/include")
  target_link_libraries(LVKVulkan PUBLIC "${LVK_CUSTOM_MOLTENVK_PATH}/dylib/macOS/libMoltenVK.dylib")
else()
  target_link_libraries(LVKVulkan PUBLIC Vulkan::Vulkan)
endif()

target_include_directories(LVKVulkan PUBLIC "${LVK_ROOT_DIR}/third-party/deps/src/")
target_include_directories(LVKVulkan PUBLIC "${LVK_ROOT_DIR}/third-party/deps/src/volk")
target_include_directories(LVKVulkan PUBLIC "${LVK_ROOT_DIR}/third-party/deps/src/vma/include")

if(WIN32)
  add_definitions("-DVK_USE_PLATFORM_WIN32_KHR=1")
  add_definitions("-DNOMINMAX")
elseif(ANDROID)
  add_definitions("-DVK_USE_PLATFORM_ANDROID_KHR=1")
elseif(UNIX AND NOT APPLE)
  if(WAYLAND_FOUND)
    add_definitions("-DVK_USE_PLATFORM_WAYLAND_KHR=1")
  else()
    add_definitions("-DVK_USE_PLATFORM_XLIB_KHR=1")
  endif()
elseif(APPLE)
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_definitions("-DVK_USE_PLATFORM_MACOS_MVK=1")
  endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(LVKVulkan PRIVATE "-Wno-nullability-completeness")
endif()
