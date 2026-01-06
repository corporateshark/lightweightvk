# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.22)

include(CMakeDependentOption)
include(FetchContent)

project("LVK"
  LANGUAGES CXX C
  HOMEPAGE_URL "https://github.com/corporateshark/lightweightvk"
)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# cmake-format: off
option(LVK_DEPLOY_DEPS             "Deploy dependencies via CMake"           ON)
option(LVK_WITH_GLFW               "Enable GLFW"                             ON)
option(LVK_WITH_SAMPLES            "Enable sample demo apps"                 ON)
option(LVK_WITH_SAMPLES_ANDROID    "Generate Android projects for demo apps" OFF)
option(LVK_WITH_TRACY              "Enable Tracy profiler"                   ON)
option(LVK_WITH_TRACY_GPU          "Enable Tracy GPU profiler"               OFF)
option(LVK_WITH_WAYLAND            "Enable Wayland"                          OFF)
option(LVK_WITH_IMPLOT             "Enable ImPlot"                           ON)
option(LVK_WITH_OPENXR             "Enable OpenXR"                           OFF)
option(LVK_WITH_ANDROID_VALIDATION "Enable validation layers on Android"     ON)
option(LVK_WITH_SLANG              "Enable Slang compiler"                   OFF)

cmake_dependent_option(LVK_WITH_VULKAN_PORTABILITY "Enable portability extension" ON "APPLE" OFF)

set(LVK_ANDROID_ABI "arm64-v8a" CACHE STRING "Enabled ABI on Android")
set(LVK_CUSTOM_MOLTENVK_PATH "" CACHE STRING "Path to custom MoltenVK")
# cmake-format: on

if(LVK_WITH_SAMPLES AND NOT LVK_WITH_GLFW)
  message(STATUS "WARNING: Setting LVK_WITH_SAMPLES=ON forces LVK_WITH_GLFW=ON")
  set(LVK_WITH_GLFW ON)
endif()

if(ANDROID)
  message(STATUS "WARNING: LVK_WITH_GLFW and LVK_WITH_SLANG were set to OFF for Android")
  set(LVK_WITH_GLFW OFF)
  set(LVK_WITH_SLANG OFF)
endif()

if(LVK_WITH_TRACY_GPU AND NOT LVK_WITH_TRACY)
  message(STATUS "WARNING: Setting LVK_WITH_TRACY_GPU=ON forces LVK_WITH_TRACY=ON")
  set(LVK_WITH_TRACY ON)
endif()

if(LVK_WITH_WAYLAND AND (ANDROID OR APPLE OR WIN32))
  message(FATAL_ERROR "LVK_WITH_WAYLAND=ON can be set only on Linux")
endif()

if(APPLE)
  if(EXISTS ${LVK_CUSTOM_MOLTENVK_PATH})
    message("Custom MoltenVK found at ${LVK_CUSTOM_MOLTENVK_PATH}")
    set(LVK_USE_CUSTOM_MOLTENVK ON)
  else()
    if(NOT LVK_WITH_VULKAN_PORTABILITY)
      message(FATAL_ERROR "LVK_WITH_VULKAN_PORTABILITY=ON must be set on macOS if Vulkan SDK is used")
    endif()
    set(LVK_USE_CUSTOM_MOLTENVK OFF)
  endif()
endif()

include(cmake/CommonMacros.txt)

if(NOT LVK_USE_CUSTOM_MOLTENVK AND DEFINED ENV{VULKAN_SDK})
  message(STATUS "VULKAN_SDK=$ENV{VULKAN_SDK}")
  if(NOT EXISTS $ENV{VULKAN_SDK})
    message(FATAL_ERROR "$ENV{VULKAN_SDK} does not exist.")
  endif()
endif()

function(lvk_set_folder target folder_name)
  set_property(TARGET ${target} PROPERTY FOLDER ${folder_name})
endfunction()

function(lvk_set_cxxstd target cpp_version)
  set_property(TARGET ${target} PROPERTY CXX_STANDARD ${cpp_version})
  set_property(TARGET ${target} PROPERTY CXX_STANDARD_REQUIRED ON)
endfunction()

# cmake-format: off
message(STATUS "LVK_DEPLOY_DEPS             = ${LVK_DEPLOY_DEPS}")
message(STATUS "LVK_WITH_GLFW               = ${LVK_WITH_GLFW}")
message(STATUS "LVK_WITH_SAMPLES            = ${LVK_WITH_SAMPLES}")
message(STATUS "LVK_WITH_SAMPLES_ANDROID    = ${LVK_WITH_SAMPLES_ANDROID}")
message(STATUS "LVK_WITH_TRACY              = ${LVK_WITH_TRACY}")
message(STATUS "LVK_WITH_TRACY_GPU          = ${LVK_WITH_TRACY_GPU}")
message(STATUS "LVK_WITH_VULKAN_PORTABILITY = ${LVK_WITH_VULKAN_PORTABILITY}")
message(STATUS "LVK_WITH_WAYLAND            = ${LVK_WITH_WAYLAND}")
message(STATUS "LVK_WITH_IMPLOT             = ${LVK_WITH_IMPLOT}")
message(STATUS "LVK_WITH_OPENXR             = ${LVK_WITH_OPENXR}")
# cmake-format: on

# cmake-format: off
if(LVK_WITH_SAMPLES_ANDROID)
  message(STATUS "LVK_ANDROID_ABI             = ${LVK_ANDROID_ABI}")
  message(STATUS "LVK_WITH_ANDROID_VALIDATION = ${LVK_WITH_ANDROID_VALIDATION}")
endif()
# cmake-format: on

if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")

message(STATUS "Generator : " ${CMAKE_GENERATOR})
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

set(LVK_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(LVK_DEPS_CACHE_FILE "${LVK_ROOT_DIR}/third-party/.bootstrap-deps.json")
find_package(Python3 COMPONENTS Interpreter)

message(STATUS "LVK_ROOT_DIR = " ${LVK_ROOT_DIR})

# enable multithreaded builds for Visual Studio
add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)

if(LVK_DEPLOY_DEPS)
  # deploy dependencies during configuration
  execute_process(COMMAND ${Python3_EXECUTABLE} deploy_deps.py WORKING_DIRECTORY ${LVK_ROOT_DIR} COMMAND_ERROR_IS_FATAL ANY)

  if(NOT EXISTS ${LVK_DEPS_CACHE_FILE})
    message(FATAL_ERROR "Cannot deploy dependencies.")
  endif()

  # regenerate dependencies when .bootstrap-deps.json is older than bootstrap-deps.json
  add_custom_command(COMMAND ${Python3_EXECUTABLE} deploy_deps.py OUTPUT ${LVK_DEPS_CACHE_FILE}
                     DEPENDS "${LVK_ROOT_DIR}/third-party/bootstrap-deps.json" WORKING_DIRECTORY ${LVK_ROOT_DIR})
  add_custom_target(LVKDependencies DEPENDS ${LVK_DEPS_CACHE_FILE})

  lvk_set_folder(LVKDependencies "LVK")

  if(LVK_WITH_SLANG)
    FetchContent_Populate(
      slang
      GIT_REPOSITORY https://github.com/shader-slang/slang
      GIT_TAG        v2025.24.3
      SOURCE_DIR     ${LVK_ROOT_DIR}/third-party/deps/src/slang
    )
  endif()
endif()

if(LVK_WITH_WAYLAND)
  include(${LVK_ROOT_DIR}/third-party/deps/src/cmake-wayland/FindWayland.cmake)
  if(NOT WAYLAND_FOUND)
    message(FATAL_ERROR "Wayland is not found in this OS")
  endif()
endif()

if(LVK_WITH_TRACY)
  add_definitions("-DTRACY_ENABLE=1")
  add_subdirectory(third-party/deps/src/tracy)
  lvk_set_folder(TracyClient "third-party")
endif()

if(LVK_WITH_OPENXR)
  if(NOT WIN32)
    message(FATAL_ERROR "OpenXR can be enabled only on Windows.")
  endif()
  add_subdirectory(third-party/deps/src/openxr-sdk)
  # cmake-format: off
  lvk_set_folder(openxr_loader             "third-party/OpenXR")
  lvk_set_folder(generate_openxr_header    "third-party/OpenXR")
  lvk_set_folder(xr_common_generated_files "third-party/OpenXR")
  lvk_set_folder(xr_global_generated_files "third-party/OpenXR")
  # cmake-format: on
endif()

# cmake-format: off
set(MINILOG_BUILD_EXAMPLE OFF CACHE BOOL "")
set(MINILOG_RAW_OUTPUT     ON CACHE BOOL "")
# cmake-format: on
add_subdirectory(third-party/deps/src/minilog)
lvk_set_folder(minilog "third-party")
target_compile_definitions(minilog PUBLIC "-DMINILOG_ENABLE_VA_LIST=1")

# temporary
if(NOT LVK_USE_CUSTOM_MOLTENVK AND NOT LVK_WITH_SLANG)
  find_package(Vulkan REQUIRED)
endif()

include_directories(.)
include_directories(src)

add_subdirectory(lvk)

target_include_directories(LVKLibrary PUBLIC ${LVK_ROOT_DIR})
target_include_directories(LVKLibrary PUBLIC ${LVK_ROOT_DIR}/src)

if(LVK_WITH_TRACY)
  target_link_libraries(LVKVulkan PUBLIC TracyClient)
  target_compile_definitions(LVKVulkan PUBLIC "LVK_WITH_TRACY=1")
  if(LVK_WITH_TRACY_GPU)
    target_compile_definitions(LVKVulkan PUBLIC "LVK_WITH_TRACY_GPU=1")
  endif()
endif()

if(LVK_WITH_GLFW)
  target_compile_definitions(LVKLibrary PUBLIC "LVK_WITH_GLFW=1")
endif()

if(LVK_WITH_IMPLOT)
  target_compile_definitions(LVKLibrary PUBLIC "LVK_WITH_IMPLOT=1")
endif()

if(LVK_DEPLOY_DEPS)
  add_dependencies(LVKLibrary LVKDependencies)
endif()

add_library(LVKstb third-party/deps/patches/stb_impl/stb_image.c third-party/deps/patches/stb_impl/stb_image_resize2.c
                   third-party/deps/patches/stb_impl/stb_image_write.c)
target_include_directories(LVKstb PUBLIC "third-party/deps/src/stb")
if(ANDROID)
  target_compile_definitions(LVKstb PUBLIC "STBI_NO_SIMD=1")
  target_compile_definitions(LVKstb PUBLIC "STBIR_NO_SIMD=1")
endif()
lvk_set_folder(LVKstb "LVK")

target_link_libraries(LVKLibrary PUBLIC minilog)
target_include_directories(LVKLibrary PUBLIC "third-party/deps/src")

if(LVK_WITH_GLFW)
  # cmake-format: off
  set(GLFW_BUILD_DOCS     OFF CACHE BOOL "")
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "")
  set(GLFW_BUILD_TESTS    OFF CACHE BOOL "")
  set(GLFW_INSTALL        OFF CACHE BOOL "")
  set(GLFW_VULKAN_STATIC  OFF CACHE BOOL "")
  if(WAYLAND_FOUND)
    set(GLFW_BUILD_WAYLAND  ON)
    set(GLFW_BUILD_X11     OFF)
  else()
    set(GLFW_BUILD_WAYLAND OFF)
    set(GLFW_BUILD_X11      ON)
  endif()
  add_subdirectory(third-party/deps/src/glfw)
  lvk_set_folder(glfw            "third-party/GLFW3")
  lvk_set_folder(update_mappings "third-party/GLFW3")
  # cmake-format: on
endif()
  
if(LVK_WITH_SAMPLES)
  include_directories("third-party/deps/src")
  include_directories("third-party/deps/src/glm")
  include_directories("third-party/deps/src/stb")
  include_directories("third-party/deps/src/taskflow")
  include_directories("third-party/deps/src/3D-Graphics-Rendering-Cookbook")
  if(UNIX AND NOT ANDROID)
    find_package(OpenGL REQUIRED)
  endif()
  # cmake-format: off
  set(LIBKTX_VERSION_READ_ONLY    OFF CACHE BOOL "")
  set(LIBKTX_VERSION_FULL          ON CACHE BOOL "")
  set(LIBKTX_FEATURE_KTX1          ON CACHE BOOL "")
  set(LIBKTX_FEATURE_KTX2          ON CACHE BOOL "")
  set(LIBKTX_FEATURE_VK_UPLOAD    OFF CACHE BOOL "")
  set(LIBKTX_FEATURE_GL_UPLOAD    OFF CACHE BOOL "")
  set(LIBKTX_FEATURE_ETC_UNPACK    ON CACHE BOOL "")
  # cmake-format: on
  add_subdirectory(third-party/deps/src/ldrutils)
  lvk_set_folder(LUtils "third-party")
  add_subdirectory(third-party/deps/src/ktx-software/lib)
  lvk_set_folder(ktx "third-party/ktx-software")
  lvk_set_folder(ktx_version "third-party/ktx-software")
  lvk_set_folder(basisu_encoder "third-party/ktx-software")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(basisu_encoder PUBLIC "-Wno-deprecated-anon-enum-enum-conversion")
  endif()
  if(TARGET astcenc-avx2-static)
    lvk_set_folder(astcenc-avx2-static "third-party/ktx-software")
  endif()
  add_subdirectory(third-party/deps/src/meshoptimizer)
  add_subdirectory(third-party/deps/src/fast_obj)
  add_subdirectory(samples)
  # cmake-format: off
  lvk_set_folder(meshoptimizer "third-party")
  lvk_set_folder(fast_obj_lib  "third-party")
  # cmake-format: on
endif()

if(LVK_WITH_OPENXR)
  target_compile_definitions(LVKVulkan PRIVATE "XR_USE_GRAPHICS_API_VULKAN=1")
  target_compile_definitions(LVKLibrary PUBLIC "LVK_WITH_OPENXR=1")
  target_link_libraries(LVKLibrary PUBLIC OpenXR::openxr_loader)
  target_include_directories(LVKLibrary PUBLIC "${LVK_ROOT_DIR}/third-party/deps/src/openxr-sdk/include")
endif()

# ImGui
target_include_directories(LVKLibrary PUBLIC "${LVK_ROOT_DIR}/third-party/deps/src/imgui")
