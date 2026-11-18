# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.27)

set(PROJECT_NAME "LVK Samples")

if(MSVC)
  add_definitions(-D_CONSOLE)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_compile_options(-Wno-deprecated-volatile)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-Wno-init-list-lifetime)
endif()

if(WIN32)
  add_definitions("-DVK_USE_PLATFORM_WIN32_KHR=1")
  add_definitions("-DNOMINMAX")
endif()

if(ANDROID)
  if(DEFINED ENV{ANDROID_NDK})
    cmake_path(SET NDK_PATH $ENV{ANDROID_NDK})
  else()
    cmake_path(SET NDK_PATH $ENV{NDK_ROOT})
  endif()
endif()

macro(ADD_VULKAN_APP_LIBRARY name)
  add_library(${name}
    VulkanApp.cpp
    VulkanApp.h
    "${LVK_ROOT_DIR}/third-party/deps/src/3D-Graphics-Rendering-Cookbook/shared/UtilsCubemap.cpp"
  )
  lvk_set_cxxstd(${name} 20)
  lvk_set_folder(${name} "LVK")
  target_link_libraries(${name} PUBLIC LVKLibrary)
  target_link_libraries(${name} PUBLIC LVKstb)
  target_link_libraries(${name} PUBLIC LUtils)
  target_link_libraries(${name} PUBLIC ktx)
endmacro()

if(NOT EMSCRIPTEN)
  ADD_VULKAN_APP_LIBRARY(LVKVulkanApp)
endif()

# the same VulkanApp running on the WebGPU backend (lvk/webgpu)
if(LVK_WITH_WEBGPU)
  ADD_VULKAN_APP_LIBRARY(LVKVulkanAppWebGPU)
  target_link_libraries(LVKVulkanAppWebGPU PUBLIC LVKWebGPU)
  target_compile_definitions(LVKVulkanAppWebGPU PUBLIC "LVK_DEMO_WITH_WEBGPU=1")
endif()

if(ANDROID)
  target_sources(LVKVulkanApp PRIVATE "${NDK_PATH}/sources/android/native_app_glue/android_native_app_glue.c")
  target_include_directories(LVKVulkanApp PUBLIC "${NDK_PATH}/sources/android/native_app_glue")
  target_link_libraries(LVKVulkanApp PUBLIC android log)
  target_link_options(LVKVulkanApp INTERFACE "LINKER:-u,ANativeActivity_onCreate")
endif()
if(LVK_WITH_ML_EMULATION_LAYER AND NOT ANDROID)
  target_compile_definitions(LVKVulkanApp PUBLIC "LVK_ML_EMULATION_LAYER_PATH=\"${LVK_ML_EMULATION_LAYER_PATH}\"")
  add_dependencies(LVKVulkanApp MLEmulationLayer)
endif()

macro(ADD_DEMO_TARGET app target_name defs)
  add_executable(${target_name} "${app}.cpp")
  lvk_set_cxxstd(${target_name} 20)
  lvk_set_folder(${target_name} ${PROJECT_NAME})
  target_link_libraries(${target_name} PRIVATE LVKLibrary)
  target_link_libraries(${target_name} PRIVATE meshoptimizer)
  target_link_libraries(${target_name} PRIVATE fast_obj_lib)
  if("${defs}" MATCHES "LVK_DEMO_WITH_WEBGPU")
    target_link_libraries(${target_name} PRIVATE LVKVulkanAppWebGPU)
    lvk_copy_webgpu_runtime_dlls(${target_name})
    if(EMSCRIPTEN)
      # a web page; VulkanApp needs `third-party/content` to exist and the ImGui font
      set(LVK_WEB_FONT "3D-Graphics-Rendering-Cookbook/data/OpenSans-Light.ttf")
      file(WRITE "${CMAKE_BINARY_DIR}/web-content/README.txt" "The sample content is not deployed to the web.")
      set_target_properties(${target_name} PROPERTIES SUFFIX ".html")
      # some samples use std::terminate() without <exception>, which other standard libraries include transitively
      target_compile_options(${target_name} PRIVATE "SHELL:-include exception")
      target_link_options(${target_name} PRIVATE "-sALLOW_MEMORY_GROWTH=1" "-sSTACK_SIZE=8MB"
                          "--preload-file=${CMAKE_BINARY_DIR}/web-content@/third-party/content"
                          "--preload-file=${LVK_ROOT_DIR}/third-party/deps/src/${LVK_WEB_FONT}@/third-party/deps/src/${LVK_WEB_FONT}")
    endif()
  else()
    target_link_libraries(${target_name} PRIVATE LVKVulkanApp)
  endif()
  if(NOT "${defs}" STREQUAL "")
    target_compile_definitions(${target_name} PRIVATE ${defs})
  endif()
endmacro()

# `${app}_WebGPU` (and `${app}_WebGPU_Slang`) on the WebGPU backend, after ADD_DEMO()
macro(ADD_WEBGPU_DEMO app)
  if(LVK_WITH_WEBGPU)
    ADD_DEMO_TARGET(${app} ${app}_WebGPU "LVK_DEMO_WITH_WEBGPU=1")
    if(LVK_WITH_SLANG)
      ADD_DEMO_TARGET(${app} ${app}_WebGPU_Slang "LVK_DEMO_WITH_SLANG=1;LVK_DEMO_WITH_WEBGPU=1")
    endif()
  endif()
endmacro()

macro(ADD_DEMO app) # optional args: `demo_has_slang` `demo_has_openxr`
  if(NOT ANDROID AND NOT EMSCRIPTEN)
    # default values for optional args
    set(demo_has_slang FALSE)
    set(demo_has_openxr FALSE)

    if(${ARGC} GREATER 1)
      set(demo_has_slang ${ARGV1})
    endif()
    if(${ARGC} GREATER 2)
      set(demo_has_openxr ${ARGV2})
    endif()

    # GLSL
    ADD_DEMO_TARGET(${app} ${app} "")

    # Slang
    if(LVK_WITH_SLANG AND demo_has_slang)
      ADD_DEMO_TARGET(${app} ${app}_Slang "LVK_DEMO_WITH_SLANG=1")
    endif()

    # OpenXR
    if(LVK_WITH_OPENXR AND demo_has_openxr)
      ADD_DEMO_TARGET(${app} ${app}_OpenXR "LVK_DEMO_WITH_OPENXR=1")
      lvk_set_folder(${app}_OpenXR "${PROJECT_NAME} OpenXR")
    endif()

    # Slang + OpenXR
    if(LVK_WITH_SLANG AND demo_has_slang AND LVK_WITH_OPENXR AND demo_has_openxr)
      ADD_DEMO_TARGET(${app} ${app}_Slang_OpenXR "LVK_DEMO_WITH_SLANG=1;LVK_DEMO_WITH_OPENXR=1")
      lvk_set_folder(${app}_Slang_OpenXR "${PROJECT_NAME} OpenXR")
    endif()
  endif()

  # Android
  if(LVK_WITH_SAMPLES_ANDROID)
    if((NOT DEFINED ENV{ANDROID_NDK}) AND (NOT DEFINED ENV{NDK_ROOT}))
      message(FATAL_ERROR "ANDROID_NDK (or NDK_ROOT) environment variable is not set")
    endif()

    # One CMakeLists file is used for generation Android Project and for building native library.
    # The native library is built only if ANDROID is defined, otherwise Android project is generated.
    if(NOT ANDROID)
      file(COPY "${LVK_ROOT_DIR}/android/build.gradle"
           DESTINATION "${CMAKE_BINARY_DIR}/android/${app}")
      file(COPY "${LVK_ROOT_DIR}/android/gradle.properties"
           DESTINATION "${CMAKE_BINARY_DIR}/android/${app}")
      file(COPY "${LVK_ROOT_DIR}/android/gradlew"
           DESTINATION "${CMAKE_BINARY_DIR}/android/${app}")
      file(COPY "${LVK_ROOT_DIR}/android/gradlew.bat"
           DESTINATION "${CMAKE_BINARY_DIR}/android/${app}")
      file(COPY "${LVK_ROOT_DIR}/android/gradle/wrapper/gradle-wrapper.jar"
           DESTINATION "${CMAKE_BINARY_DIR}/android/${app}/gradle/wrapper")
      file(COPY "${LVK_ROOT_DIR}/android/gradle/wrapper/gradle-wrapper.properties"
           DESTINATION "${CMAKE_BINARY_DIR}/android/${app}/gradle/wrapper")
      set(APP_NAME ${app})

      set(ABI_STRING "")
      foreach(str ${LVK_ANDROID_ABI})
        if(NOT ${ABI_STRING} STRLESS "")
          set(ABI_STRING "${ABI_STRING}, ")
        endif()
        set(ABI_STRING "${ABI_STRING}'${str}'")
      endforeach()
      set(ANDROID_ABI_FILTERS ${ABI_STRING})
      set(ANDROID_JNI_LIBS "")
      if(LVK_ANDROID_JNI_LIBS)
        set(ANDROID_JNI_LIBS "'${LVK_ANDROID_JNI_LIBS}'")
      endif()

      configure_file("${LVK_ROOT_DIR}/android/settings.gradle.in"
                     "${CMAKE_BINARY_DIR}/android/${app}/settings.gradle")
      configure_file("${LVK_ROOT_DIR}/android/app/build.gradle.in"
                     "${CMAKE_BINARY_DIR}/android/${app}/app/build.gradle")
      configure_file("${LVK_ROOT_DIR}/android/app/AndroidManifest.xml.in"
                     "${CMAKE_BINARY_DIR}/android/${app}/app/src/main/AndroidManifest.xml")
      configure_file("${LVK_ROOT_DIR}/android/app/MainActivity.java.in"
                     "${CMAKE_BINARY_DIR}/android/${app}/app/src/main/java/org/lvk/samples/MainActivity.java")

      set_property(GLOBAL APPEND PROPERTY LVK_ANDROID_PACKAGES "org.lvk.samples.lvk_${app}")

      if(LVK_WITH_ANDROID_VALIDATION)
        list(APPEND abdroid_abi arm64-v8a armeabi-v7a x86 x86_64)
        foreach(abi ${abdroid_abi})
          string(FIND ${ABI_STRING} "${abi}" abi_substr_index)
          if(abi_substr_index GREATER -1)
            file(COPY "${LVK_ROOT_DIR}/third-party/deps/src/android-validation-layers/${abi}/libVkLayer_khronos_validation.so"
                 DESTINATION "${CMAKE_BINARY_DIR}/android/${app}/app/src/main/jniLibs/${abi}")
          endif()
        endforeach()
      endif()
    endif()

    if(ANDROID)
      add_library(lvk_android_native_${app} SHARED "${app}.cpp")
      lvk_set_cxxstd(lvk_android_native_${app} 20)
      target_link_libraries(lvk_android_native_${app} PUBLIC
                            LVKVulkanApp LVKLibrary LVKVulkan LVKstb meshoptimizer fast_obj_lib)
    endif()
  endif()
endmacro()

macro(ADD_DEMO_SOURCES app srcs)
  if(ANDROID)
    target_sources(lvk_android_native_${app} PRIVATE ${srcs})
  else()
    foreach(suffix "" "_Slang" "_OpenXR" "_Slang_OpenXR" "_WebGPU" "_WebGPU_Slang")
      if(TARGET ${app}${suffix})
        target_sources(${app}${suffix} PRIVATE ${srcs})
      endif()
    endforeach()
  endif()
endmacro()

macro(ADD_DEMO_LINK_LIBRARIES app libs)
  if(ANDROID)
    target_link_libraries(lvk_android_native_${app} PRIVATE ${libs})
  else()
    target_link_libraries(${app} PRIVATE ${libs})
  endif()
endmacro()

ADD_DEMO("001_HelloTriangle" LVK_WITH_SLANG)
ADD_DEMO("002_RenderToCubeMap" LVK_WITH_SLANG)
ADD_DEMO("003_RenderToCubeMapSinglePass" LVK_WITH_SLANG)
ADD_DEMO("004_YUV" LVK_WITH_SLANG)
if(WIN32 OR UNIX AND NOT (APPLE))
  # Windows and Linux
  ADD_DEMO("005_MeshShaders" LVK_WITH_SLANG)
  ADD_DEMO("008_MeshShaderFireworks" LVK_WITH_SLANG LVK_WITH_OPENXR)
endif()
ADD_DEMO("006_SwapchainHDR" LVK_WITH_SLANG)
ADD_DEMO("007_DynamicRenderingLocalRead" LVK_WITH_SLANG)
ADD_DEMO("009_TriplanarMapping" LVK_WITH_SLANG)
ADD_DEMO("010_OmniShadows" LVK_WITH_SLANG)
ADD_DEMO("011_VariableRateShading" LVK_WITH_SLANG)
if((WIN32 OR UNIX AND NOT (APPLE)) OR LVK_WITH_SAMPLES_ANDROID)
  # Windows, Linux and Android
  ADD_DEMO("RTX_001_Hello" LVK_WITH_SLANG)
  ADD_DEMO("RTX_002_AO" LVK_WITH_SLANG)
  ADD_DEMO("RTX_003_Pipeline" LVK_WITH_SLANG)
  ADD_DEMO("RTX_004_Textures" LVK_WITH_SLANG)
endif()

ADD_DEMO("DEMO_001_SolarSystem" LVK_WITH_SLANG)
ADD_DEMO("Tiny_MeshLarge" LVK_WITH_SLANG)

# the samples which also run on the WebGPU backend
ADD_WEBGPU_DEMO("001_HelloTriangle")
ADD_WEBGPU_DEMO("002_RenderToCubeMap")
ADD_WEBGPU_DEMO("003_RenderToCubeMapSinglePass")
ADD_WEBGPU_DEMO("009_TriplanarMapping")
ADD_WEBGPU_DEMO("010_OmniShadows")
ADD_WEBGPU_DEMO("DEMO_001_SolarSystem")
ADD_WEBGPU_DEMO("Tiny_MeshLarge")
if(EMSCRIPTEN)
  # the web pages pack the content they read
  foreach(pair "009_TriplanarMapping:src/bistro/BuildingTextures/wood_polished_01_diff.png" "DEMO_001_SolarSystem:src/solarsystem")
    string(REPLACE ":" ";" pair "${pair}")
    list(GET pair 0 app)
    list(GET pair 1 path)
    target_link_options(${app}_WebGPU PRIVATE "--preload-file=${LVK_ROOT_DIR}/third-party/content/${path}@/third-party/content/${path}")
  endforeach()
  # the Bistro textures Tiny_MeshLarge uses (the mesh cache comes from a native run of Tiny_MeshLarge)
  set(LVK_WEB_BISTRO_RSP "${CMAKE_BINARY_DIR}/web-content/Tiny_MeshLarge.rsp")
  if(EXISTS "${LVK_ROOT_DIR}/third-party/content/cache.data")
    execute_process(COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/web_preload_bistro.py" "${LVK_ROOT_DIR}/third-party/content"
                            "${LVK_WEB_BISTRO_RSP}" COMMAND_ERROR_IS_FATAL ANY)
    target_link_options(Tiny_MeshLarge_WebGPU PRIVATE "@${LVK_WEB_BISTRO_RSP}")
  else()
    message(WARNING "third-party/content/cache.data is missing: run Tiny_MeshLarge natively once before the web build of Tiny_MeshLarge_WebGPU")
  endif()
endif()

ADD_DEMO_SOURCES("Tiny_MeshLarge" "${LVK_ROOT_DIR}/third-party/deps/src/imgui/imgui_demo.cpp")

if(LVK_WITH_SAMPLES_ANDROID AND LVK_ANDROID_OBB_CONTENT AND NOT ANDROID)
  get_property(LVK_ANDROID_PACKAGES GLOBAL PROPERTY LVK_ANDROID_PACKAGES)
  if(LVK_ANDROID_PACKAGES)
    add_custom_target(LVKAndroidContent
                      COMMAND ${Python3_EXECUTABLE} "${LVK_ROOT_DIR}/deploy_content_android.py" --obb ${LVK_ANDROID_PACKAGES}
                      WORKING_DIRECTORY "${LVK_ROOT_DIR}"
                      COMMENT "Installing the sample content as an OBB for every Android demo"
                      USES_TERMINAL)
    lvk_set_folder(LVKAndroidContent ${PROJECT_NAME})
  endif()
endif()
