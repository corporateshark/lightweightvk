# Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.22)

include(CMakeDependentOption)

project("LVK"
  LANGUAGES CXX C
  HOMEPAGE_URL "https://github.com/corporateshark/lightweightvk"
)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# cmake-format: off
option(LVK_DEPLOY_DEPS             "Deploy dependencies via CMake"           ON)
option(LVK_DEPLOY_SCREENSHOT_TESTS "Deploy screenshot tests"                 ON)
option(LVK_WITH_GLFW               "Enable GLFW"                             ON)
option(LVK_WITH_SDL3               "Enable SDL3"                             OFF)
option(LVK_WITH_SAMPLES            "Enable sample demo apps"                 ON)
option(LVK_WITH_SAMPLES_ANDROID    "Generate Android projects for demo apps" OFF)
option(LVK_WITH_TRACY              "Enable Tracy profiler"                   ON)
option(LVK_WITH_TRACY_GPU          "Enable Tracy GPU profiler"               OFF)
option(LVK_WITH_WAYLAND            "Enable Wayland"                          OFF)
option(LVK_WITH_OPENXR             "Enable OpenXR"                           OFF)
option(LVK_WITH_ANDROID_VALIDATION "Enable validation layers on Android"     ON)
option(LVK_WITH_MINILOG            "Enable Minilog"                          ON)
option(LVK_WITH_SLANG              "Enable Slang compiler"                   OFF)
option(LVK_WITH_SPIRV_OPT          "Run SPIRV optimization on shaders"       OFF)
option(LVK_WITH_IMPLOT             "Enable ImPlot"                           ON)
option(LVK_IMGUI_EXTERNAL          "Use external ImGui library"              OFF)
option(LVK_IMPLOT_EXTERNAL         "Use external ImPlot library"             OFF)
option(LVK_GLSLANG_EXTERNAL        "Use external glslang and SPIRV-Tools"    OFF)
option(LVK_SPIRV_REFLECT_EXTERNAL  "Use external SPIRV-Reflect library"      OFF)
option(LVK_ENABLE_LONG_PATHS_WIN32 "Enable long paths on Windows"            ON)
option(LVK_WITH_ML_EMULATION_LAYER "Build the Arm ML Emulation Layer for Vulkan (VK_ARM_tensors/VK_ARM_data_graph)" OFF)

set(LVK_ANDROID_ABI "arm64-v8a" CACHE STRING "Enabled ABI on Android")
# cmake-format: on

if(LVK_WITH_GLFW AND LVK_WITH_SDL3)
  message(FATAL_ERROR "LVK_WITH_GLFW and LVK_WITH_SDL3 are mutually exclusive. Enable only one.")
endif()

if(LVK_WITH_SAMPLES AND NOT LVK_WITH_GLFW AND NOT LVK_WITH_SDL3)
  message(STATUS "WARNING: Setting LVK_WITH_SAMPLES=ON forces LVK_WITH_GLFW=ON")
  set(LVK_WITH_GLFW ON)
endif()

if(ANDROID)
  message(STATUS "WARNING: LVK_WITH_GLFW, LVK_WITH_SDL3, and LVK_WITH_SLANG were set to OFF for Android")
  set(LVK_WITH_GLFW  OFF)
  set(LVK_WITH_SDL3  OFF)
  set(LVK_WITH_SLANG OFF)
endif()

if(LVK_WITH_TRACY_GPU AND NOT LVK_WITH_TRACY)
  message(STATUS "WARNING: Setting LVK_WITH_TRACY_GPU=ON forces LVK_WITH_TRACY=ON")
  set(LVK_WITH_TRACY ON)
endif()

if(LVK_WITH_WAYLAND AND (ANDROID OR APPLE OR WIN32))
  message(FATAL_ERROR "LVK_WITH_WAYLAND=ON can be set only on Linux")
endif()

if(LVK_ENABLE_LONG_PATHS_WIN32 AND WIN32)
  execute_process(
    COMMAND reg query "HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem" /v LongPathsEnabled
    OUTPUT_VARIABLE LONG_PATHS_OUTPUT
    ERROR_VARIABLE LONG_PATHS_ERROR
    RESULT_VARIABLE LONG_PATHS_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(LONG_PATHS_RESULT EQUAL 0 AND LONG_PATHS_OUTPUT MATCHES "0x1")
    message(STATUS "Windows long paths enabled")
  else()
    message(FATAL_ERROR "Windows long paths not enabled. Enable via PowerShell (admin): New-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem' -Name 'LongPathsEnabled' -Value 1 -PropertyType DWORD -Force")
  endif()
endif()

include(cmake/CommonMacros.txt)

if(DEFINED ENV{VULKAN_SDK})
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
message(STATUS "LVK_DEPLOY_SCREENSHOT_TESTS = ${LVK_DEPLOY_SCREENSHOT_TESTS}")
message(STATUS "LVK_WITH_GLFW               = ${LVK_WITH_GLFW}")
message(STATUS "LVK_WITH_SDL3               = ${LVK_WITH_SDL3}")
message(STATUS "LVK_WITH_SAMPLES            = ${LVK_WITH_SAMPLES}")
message(STATUS "LVK_WITH_SAMPLES_ANDROID    = ${LVK_WITH_SAMPLES_ANDROID}")
message(STATUS "LVK_WITH_TRACY              = ${LVK_WITH_TRACY}")
message(STATUS "LVK_WITH_TRACY_GPU          = ${LVK_WITH_TRACY_GPU}")
message(STATUS "LVK_WITH_WAYLAND            = ${LVK_WITH_WAYLAND}")
message(STATUS "LVK_WITH_OPENXR             = ${LVK_WITH_OPENXR}")
message(STATUS "LVK_WITH_MINILOG            = ${LVK_WITH_MINILOG}")
message(STATUS "LVK_WITH_SLANG              = ${LVK_WITH_SLANG}")
message(STATUS "LVK_WITH_SPIRV_OPT          = ${LVK_WITH_SPIRV_OPT}")
message(STATUS "LVK_WITH_IMPLOT             = ${LVK_WITH_IMPLOT}")
message(STATUS "LVK_WITH_RAW_VULKAN         = ${LVK_WITH_RAW_VULKAN}")
message(STATUS "LVK_IMGUI_EXTERNAL          = ${LVK_IMGUI_EXTERNAL}")
message(STATUS "LVK_IMPLOT_EXTERNAL         = ${LVK_IMPLOT_EXTERNAL}")
message(STATUS "LVK_GLSLANG_EXTERNAL        = ${LVK_GLSLANG_EXTERNAL}")
message(STATUS "LVK_SPIRV_REFLECT_EXTERNAL  = ${LVK_SPIRV_REFLECT_EXTERNAL}")
message(STATUS "LVK_ENABLE_LONG_PATHS_WIN32 = ${LVK_ENABLE_LONG_PATHS_WIN32}")
message(STATUS "LVK_WITH_ML_EMULATION_LAYER = ${LVK_WITH_ML_EMULATION_LAYER}")
# cmake-format: on

# cmake-format: off
if(LVK_WITH_SAMPLES_ANDROID)
  message(STATUS "LVK_ANDROID_ABI             = ${LVK_ANDROID_ABI}")
  message(STATUS "LVK_WITH_ANDROID_VALIDATION = ${LVK_WITH_ANDROID_VALIDATION}")
endif()
# cmake-format: on

if(NOT CMAKE_BUILD_TYPE)
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
  # pass CMake options to Bootstrap predicates via env variables
  if(LVK_DEPLOY_SCREENSHOT_TESTS)
    set(ENV{LVK_DEPLOY_SCREENSHOT_TESTS} "ON")
  else()
    set(ENV{LVK_DEPLOY_SCREENSHOT_TESTS} "OFF")
  endif()
  if(LVK_WITH_SLANG)
    set(ENV{LVK_WITH_SLANG} "ON")
  else()
    set(ENV{LVK_WITH_SLANG} "OFF")
  endif()
  if(LVK_WITH_ML_EMULATION_LAYER)
    set(ENV{LVK_WITH_ML_EMULATION_LAYER} "ON")
  else()
    set(ENV{LVK_WITH_ML_EMULATION_LAYER} "OFF")
  endif()
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

endif()

if(LVK_WITH_WAYLAND)
  include(${LVK_ROOT_DIR}/third-party/deps/src/cmake-wayland/FindWayland.cmake)
  if(NOT WAYLAND_FOUND)
    message(FATAL_ERROR "Wayland is not found in this OS")
  endif()
endif()

if(LVK_WITH_TRACY)
  set(TRACY_ENABLE ON CACHE BOOL "")
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

if(LVK_WITH_MINILOG)
  # cmake-format: off
  set(MINILOG_BUILD_EXAMPLE OFF CACHE BOOL "")
  set(MINILOG_RAW_OUTPUT     ON CACHE BOOL "")
  # cmake-format: on
  add_subdirectory(third-party/deps/src/minilog)
  lvk_set_folder(minilog "third-party")
endif()

# temporary
if(NOT LVK_WITH_SLANG)
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

if(LVK_WITH_SDL3)
  target_compile_definitions(LVKLibrary PUBLIC "LVK_WITH_SDL3=1")
endif()

if(LVK_WITH_IMPLOT)
  target_compile_definitions(LVKLibrary PUBLIC "LVK_WITH_IMPLOT=1")
  if(NOT LVK_IMPLOT_EXTERNAL)
    target_include_directories(LVKLibrary PUBLIC "${LVK_ROOT_DIR}/third-party/deps/src/implot")
  endif()
endif()

if(NOT LVK_IMGUI_EXTERNAL)
  target_include_directories(LVKLibrary PUBLIC "${LVK_ROOT_DIR}/third-party/deps/src/imgui")
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

if(LVK_WITH_MINILOG)
  target_link_libraries(LVKLibrary PUBLIC minilog)
  target_compile_definitions(LVKLibrary PUBLIC LVK_WITH_MINILOG)
endif()
target_include_directories(LVKLibrary PUBLIC "third-party/deps/src")

if(LVK_WITH_ML_EMULATION_LAYER AND NOT ANDROID)
  include(ExternalProject)
  set(LVK_DEPS_SRC_DIR "${LVK_ROOT_DIR}/third-party/deps/src")
  set(LVK_ML_SDK_DIR "${LVK_DEPS_SRC_DIR}/ml-sdk")
  set(LVK_ML_EMULATION_LAYER_PREFIX "${CMAKE_BINARY_DIR}/ml-emulation-layer")
  find_program(LVK_GLSLANG_EXECUTABLE NAMES glslang glslangValidator HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
  # glslang, SPIRV-Headers, SPIRV-Tools and Vulkan-Headers are shared with LVK
  # cmake-format: off
  set(LVK_ML_EMULATION_LAYER_CMAKE_ARGS
      "-DCMAKE_BUILD_TYPE=Release"
      "-DCMAKE_INSTALL_PREFIX=${LVK_ML_EMULATION_LAYER_PREFIX}"
      "-DVULKAN_HEADERS_PATH=${LVK_DEPS_SRC_DIR}/vulkan-headers"
      "-DSPIRV_HEADERS_PATH=${LVK_DEPS_SRC_DIR}/SPIRV-Headers"
      "-DSPIRV_TOOLS_PATH=${LVK_DEPS_SRC_DIR}/SPIRV-Tools"
      "-DGLSLANG_PATH=${LVK_DEPS_SRC_DIR}/glslang"
      "-DSPIRV_CROSS_PATH=${LVK_ML_SDK_DIR}/SPIRV-Cross"
      "-DJSON_PATH=${LVK_ML_SDK_DIR}/json"
      "-DVMEL_TESTS_ENABLE=OFF"
      "-DVMEL_USE_FLOAT_AS_DOUBLE=OFF")
  # cmake-format: on
  if(LVK_GLSLANG_EXECUTABLE)
    list(APPEND LVK_ML_EMULATION_LAYER_CMAKE_ARGS "-DGLSLANG_EXECUTABLE=${LVK_GLSLANG_EXECUTABLE}")
  endif()
  if(WIN32)
    set(LVK_ML_EMULATION_LAYER_PATH "${LVK_ML_EMULATION_LAYER_PREFIX}/bin")
    set(LVK_ML_EMULATION_LAYER_LIB_DIRS "${LVK_ML_EMULATION_LAYER_PREFIX}/bin")
  else()
    set(LVK_ML_EMULATION_LAYER_PATH "${LVK_ML_EMULATION_LAYER_PREFIX}/share/vulkan/explicit_layer.d")
    set(LVK_ML_EMULATION_LAYER_LIB_DIRS "${LVK_ML_EMULATION_LAYER_PREFIX}/lib|${LVK_ML_EMULATION_LAYER_PREFIX}/lib64")
  endif()
  ExternalProject_Add(MLEmulationLayer
    SOURCE_DIR "${LVK_ML_SDK_DIR}/emulation-layer"
    BINARY_DIR "${LVK_ML_EMULATION_LAYER_PREFIX}/build"
    CMAKE_ARGS ${LVK_ML_EMULATION_LAYER_CMAKE_ARGS}
    BUILD_ALWAYS ON
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release --parallel
    INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR> --config Release
    COMMAND ${CMAKE_COMMAND} "-DMANIFEST_DIR=${LVK_ML_EMULATION_LAYER_PATH}" "-DLIBRARY_DIRS=${LVK_ML_EMULATION_LAYER_LIB_DIRS}"
            -P "${LVK_ROOT_DIR}/cmake/MLEmulationLayerManifests.cmake"
    USES_TERMINAL_BUILD ON)
  lvk_set_folder(MLEmulationLayer "third-party/ml-sdk")
  message(STATUS "Arm ML Emulation Layer manifests: ${LVK_ML_EMULATION_LAYER_PATH}")
  if(LVK_WITH_SAMPLES_ANDROID)
    if(DEFINED ENV{ANDROID_NDK})
      cmake_path(SET LVK_ANDROID_NDK $ENV{ANDROID_NDK})
    elseif(DEFINED ENV{NDK_ROOT})
      cmake_path(SET LVK_ANDROID_NDK $ENV{NDK_ROOT})
    else()
      message(FATAL_ERROR "ANDROID_NDK (or NDK_ROOT) environment variable is not set")
    endif()
    if(CMAKE_HOST_WIN32)
      set(LVK_ANDROID_NDK_HOST windows-x86_64)
    elseif(CMAKE_HOST_APPLE)
      set(LVK_ANDROID_NDK_HOST darwin-x86_64)
    else()
      set(LVK_ANDROID_NDK_HOST linux-x86_64)
    endif()
    file(GLOB LVK_ANDROID_SDK_CMAKE_BIN "$ENV{ANDROID_HOME}/cmake/*/bin" "$ENV{ANDROID_SDK_ROOT}/cmake/*/bin")
    find_program(LVK_NINJA_EXECUTABLE ninja HINTS ${LVK_ANDROID_SDK_CMAKE_BIN})
    find_program(LVK_ANDROID_STRIP llvm-strip HINTS "${LVK_ANDROID_NDK}/toolchains/llvm/prebuilt/${LVK_ANDROID_NDK_HOST}/bin")
    if(NOT LVK_NINJA_EXECUTABLE OR NOT LVK_ANDROID_STRIP)
      message(FATAL_ERROR "The Arm ML Emulation Layer for Android needs ninja and llvm-strip: ${LVK_NINJA_EXECUTABLE} ${LVK_ANDROID_STRIP}")
    endif()
    set(LVK_ML_EMULATION_LAYER_ANDROID_PREFIX "${CMAKE_BINARY_DIR}/ml-emulation-layer-android")
    set(LVK_ANDROID_JNI_LIBS "${LVK_ML_EMULATION_LAYER_ANDROID_PREFIX}/jniLibs")
    set(LVK_ML_EMULATION_LAYER_ANDROID_CMAKE_ARGS ${LVK_ML_EMULATION_LAYER_CMAKE_ARGS})
    list(FILTER LVK_ML_EMULATION_LAYER_ANDROID_CMAKE_ARGS EXCLUDE REGEX "^-DCMAKE_INSTALL_PREFIX=")
    foreach(abi ${LVK_ANDROID_ABI})
      set(libs "${LVK_ANDROID_JNI_LIBS}/${abi}")
      ExternalProject_Add(MLEmulationLayerAndroid_${abi}
        SOURCE_DIR "${LVK_ML_SDK_DIR}/emulation-layer"
        BINARY_DIR "${LVK_ML_EMULATION_LAYER_ANDROID_PREFIX}/build/${abi}"
        CMAKE_GENERATOR Ninja
        CMAKE_ARGS ${LVK_ML_EMULATION_LAYER_ANDROID_CMAKE_ARGS}
                   "-DCMAKE_MAKE_PROGRAM=${LVK_NINJA_EXECUTABLE}"
                   "-DCMAKE_TOOLCHAIN_FILE=${LVK_ANDROID_NDK}/build/cmake/android.toolchain.cmake"
                   "-DANDROID_ABI=${abi}"
                   "-DANDROID_PLATFORM=android-33"
        BUILD_ALWAYS ON
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --parallel
        INSTALL_COMMAND ${CMAKE_COMMAND} -E make_directory "${libs}"
        COMMAND ${CMAKE_COMMAND} -E copy <BINARY_DIR>/graph/libVkLayer_Graph.so <BINARY_DIR>/tensor/libVkLayer_Tensor.so "${libs}"
        COMMAND ${LVK_ANDROID_STRIP} --strip-unneeded "${libs}/libVkLayer_Graph.so" "${libs}/libVkLayer_Tensor.so"
        USES_TERMINAL_BUILD ON)
      lvk_set_folder(MLEmulationLayerAndroid_${abi} "third-party/ml-sdk")
    endforeach()
    message(STATUS "Arm ML Emulation Layer for Android: ${LVK_ANDROID_JNI_LIBS}")
  endif()
endif()

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

if(LVK_WITH_SDL3)
  # cmake-format: off
  set(SDL_AUDIO        OFF CACHE BOOL "")
  set(SDL_DIRECTX      OFF CACHE BOOL "")
  set(SDL_EXAMPLES     OFF CACHE BOOL "")
  set(SDL_FRIBIDI      OFF CACHE BOOL "")
  set(SDL_INSTALL      OFF CACHE BOOL "")
  set(SDL_LIBTHAI      OFF CACHE BOOL "")
  set(SDL_LIBUDEV      OFF CACHE BOOL "")
  set(SDL_METAL        OFF CACHE BOOL "")
  set(SDL_OPENGL       OFF CACHE BOOL "")
  set(SDL_OPENGLES     OFF CACHE BOOL "")
  set(SDL_RENDER_D3D   OFF CACHE BOOL "")
  set(SDL_RENDER_D3D11 OFF CACHE BOOL "")
  set(SDL_RENDER_D3D12 OFF CACHE BOOL "")
  set(SDL_RENDER_GPU   OFF CACHE BOOL "")
  set(SDL_RENDER_METAL OFF CACHE BOOL "")
  set(SDL_SHARED       OFF CACHE BOOL "")
  set(SDL_STATIC       ON  CACHE BOOL "")
  set(SDL_TEST_LIBRARY OFF CACHE BOOL "")
  if(WAYLAND_FOUND)
    set(SDL_WAYLAND  ON)
    set(SDL_X11     OFF)
  else()
    set(SDL_WAYLAND OFF)
    set(SDL_X11      ON)
  endif()
  add_subdirectory(third-party/deps/src/SDL)
  lvk_set_folder(SDL3-static  "third-party/SDL3")
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
  set(BUILD_SHARED_LIBS           OFF CACHE BOOL "")
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
  # patch LIBKTX: this file interferes with normal Vulkan headers
  list(REMOVE_ITEM LIBKTX_MAIN_SRC "../external/dfdutils/vulkan/vulkan_core.h")
  lvk_set_folder(ktx "third-party/ktx-software")
  lvk_set_folder(ktx_version "third-party/ktx-software")
  lvk_set_folder(basisu_encoder "third-party/ktx-software")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(basisu_encoder PUBLIC "-Wno-deprecated-anon-enum-enum-conversion" "-Wno-deprecated-enum-enum-conversion")
    target_compile_options(ktx            PRIVATE "-Wno-deprecated-anon-enum-enum-conversion" "-Wno-deprecated-enum-enum-conversion")
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
