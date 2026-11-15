# LightweightVK
#
# Copyright (c) 2023-2026 Sergey Kosarevsky and contributors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.27)

# built from the LVK tree with LVK_WITH_TENSORS, or standalone by `make_nss_model.py`
if(TARGET vgf)
  set(LVK_TOOLS_EXCLUDE EXCLUDE_FROM_ALL)
else()
  project(LVKTools CXX)
  set(LVK_TOOLS_EXCLUDE "")
  set(LVK_ML_SDK_DIR "${CMAKE_CURRENT_LIST_DIR}/../third-party/deps/src/ml-sdk")
  if(NOT EXISTS "${LVK_ML_SDK_DIR}/vgf-library")
    message(FATAL_ERROR "${LVK_ML_SDK_DIR}/vgf-library is missing. Run `deploy_deps.py` first.")
  endif()
  # cmake-format: off
  set(FLATBUFFERS_PATH "${LVK_ML_SDK_DIR}/flatbuffers" CACHE PATH "" FORCE)
  set(ML_SDK_VGF_LIB_BUILD_TOOLS  OFF CACHE BOOL "" FORCE)
  set(ML_SDK_VGF_LIB_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
  set(ML_SDK_VGF_LIB_BUILD_PYLIB  OFF CACHE BOOL "" FORCE)
  set(ML_SDK_VGF_LIB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
  # cmake-format: on
  add_subdirectory("${LVK_ML_SDK_DIR}/vgf-library" vgf-library EXCLUDE_FROM_ALL)
endif()

add_executable(vgf_respecialize ${LVK_TOOLS_EXCLUDE} vgf_respecialize.cpp)
target_link_libraries(vgf_respecialize PRIVATE vgf)
set_property(TARGET vgf_respecialize PROPERTY CXX_STANDARD 20)
if(COMMAND lvk_set_folder)
  lvk_set_folder(vgf_respecialize "LVK")
endif()
