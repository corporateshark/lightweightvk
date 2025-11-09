# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)

set(PROJECT_NAME "LVK Samples")

if(MSVC)
  add_definitions(-D_CONSOLE)
else()
  add_compile_options(-Wno-deprecated-volatile)
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

add_library(LVKVulkanApp
  VulkanApp.cpp
  VulkanApp.h
  "${LVK_ROOT_DIR}/third-party/deps/src/3D-Graphics-Rendering-Cookbook/shared/UtilsCubemap.cpp"
  "${LVK_ROOT_DIR}/third-party/deps/src/imgui/imgui_demo.cpp"
)
lvk_set_cxxstd(LVKVulkanApp 20)
lvk_set_folder(LVKVulkanApp "LVK")
target_link_libraries(LVKVulkanApp PUBLIC LVKLibrary)
target_link_libraries(LVKVulkanApp PUBLIC LVKstb)
target_link_libraries(LVKVulkanApp PUBLIC LUtils)
target_link_libraries(LVKVulkanApp PUBLIC ktx)
if(ANDROID)
  target_include_directories(LVKVulkanApp PUBLIC "${NDK_PATH}/sources/android/native_app_glue")
  target_link_libraries(LVKVulkanApp PUBLIC android)
endif()

macro(ADD_DEMO app) # optional arg `demo_has_slang`
  if(NOT ANDROID)
    add_executable(${app} "${app}.cpp")
    lvk_set_cxxstd(${app} 20)
    lvk_set_folder(${app} ${PROJECT_NAME})
    target_link_libraries(${app} PRIVATE LVKLibrary)
    target_link_libraries(${app} PRIVATE meshoptimizer)
    target_link_libraries(${app} PRIVATE fast_obj_lib)
    target_link_libraries(${app} PRIVATE LVKVulkanApp)

    # default value for the optional arg
    set(demo_has_slang FALSE)

    # check if a second arg was provided
    if(${ARGC} GREATER 1)
        set(demo_has_slang ${ARGV1})
    endif()

    if(LVK_WITH_SLANG AND demo_has_slang)
      add_executable(${app}_Slang "${app}.cpp")
      lvk_set_cxxstd(${app}_Slang 20)
      lvk_set_folder(${app}_Slang ${PROJECT_NAME})
      target_link_libraries(${app}_Slang PRIVATE LVKLibrary)
      target_link_libraries(${app}_Slang PRIVATE meshoptimizer)
      target_link_libraries(${app}_Slang PRIVATE fast_obj_lib)
      target_link_libraries(${app}_Slang PRIVATE LVKVulkanApp)
      target_compile_definitions(${app}_Slang PRIVATE "LVK_DEMO_WITH_SLANG=1")
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

      configure_file("${LVK_ROOT_DIR}/android/settings.gradle.in"
                     "${CMAKE_BINARY_DIR}/android/${app}/settings.gradle")
      configure_file("${LVK_ROOT_DIR}/android/app/build.gradle.in"
                     "${CMAKE_BINARY_DIR}/android/${app}/app/build.gradle")
      configure_file("${LVK_ROOT_DIR}/android/app/AndroidManifest.xml.in"
                     "${CMAKE_BINARY_DIR}/android/${app}/app/src/main/AndroidManifest.xml")
      configure_file("${LVK_ROOT_DIR}/android/app/MainActivity.java.in"
                     "${CMAKE_BINARY_DIR}/android/${app}/app/src/main/java/org/lvk/samples/MainActivity.java")

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
      add_library(lvk_android_native_${app} SHARED
                  "${app}.cpp"
                  "${NDK_PATH}/sources/android/native_app_glue/android_native_app_glue.c")
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
    target_sources(${app} PRIVATE ${srcs})
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
endif()
ADD_DEMO("006_SwapchainHDR" LVK_WITH_SLANG)
ADD_DEMO("009_TriplanarMapping" LVK_WITH_SLANG)
if((WIN32 OR UNIX AND NOT (APPLE)) OR LVK_WITH_SAMPLES_ANDROID)
  # Windows, Linux and Android
  ADD_DEMO("RTX_001_Hello" LVK_WITH_SLANG)
  ADD_DEMO("RTX_002_AO")
  ADD_DEMO("RTX_003_Pipeline" LVK_WITH_SLANG)
  ADD_DEMO("RTX_004_Textures" LVK_WITH_SLANG)
endif()

ADD_DEMO("DEMO_001_SolarSystem")
ADD_DEMO("Tiny_MeshLarge" LVK_WITH_SLANG)
